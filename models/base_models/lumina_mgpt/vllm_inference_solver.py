import argparse
import copy
import math
from typing import List, Optional, Union
from tqdm import tqdm

from PIL import Image
import torch
import transformers
from transformers import GenerationConfig, TextStreamer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList, LogitsWarper
from vllm import LLM, SamplingParams

from .item_processor import FlexARItemProcessor

class LLMImageStartTriggeredUnbatchedClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
    r"""
    Logits processor for Classifier-Free Guidance (CFG). The processors computes a weighted average across scores
    from prompt conditional and prompt unconditional (or negative) logits, parameterized by the `guidance_scale`.
    The unconditional scores are computed internally by prompting `model` with the `unconditional_ids` branch.

    See [the paper](https://arxiv.org/abs/2306.17806) for more information.
    """

    def __init__(
        self,
        guidance_scale: float,
        model,
        image_start_token_id,
        image_end_token_id,
        image_next_line_token_id,
        patch_size,
        use_cache: Optional[bool] = True,
        **kwargs,
    ):
        self.guidance_scale = guidance_scale
        self.nums_image_start_tokens = None

        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_next_line_token_id = image_next_line_token_id
        self.image_start_token_id_index = None
        self.patch_size = patch_size
        self.h_latent_dim = None
        self.w_latent_dim = None

    def __call__(self, prompt_token_ids: torch.LongTensor, past_token_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Sanity Check for two consecutive inputs are well-paired
        assert past_token_ids[0] == past_token_ids[1], "Past token ids are not equal. Something wrong in the query scheduling process."
        assert len(prompt_token_ids[1]) == 1, "The second prompt token ids should be a single token, but it is not."

        input_ids = torch.LongTensor([prompt_token_ids[0] + past_token_ids[0]])

        num_image_start_tokens = (input_ids[0] == self.image_start_token_id).sum()
        num_image_end_tokens = (input_ids[0] == self.image_end_token_id).sum()

        if num_image_start_tokens == num_image_end_tokens:
            # Text generation
            self.h_latent_dim, self.w_latent_dim = None, None
            self.image_start_token_id_index = None

            return torch.stack([scores[0], scores[0]], dim=0)

        elif num_image_start_tokens == num_image_end_tokens + 1:
            # Image generation
            self.image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0][-1].item()
            
            new_token_num = len(input_ids[0][self.image_start_token_id_index + 1 :])
            if new_token_num >= 2:
                if self.h_latent_dim is None or self.w_latent_dim is None:
                    h_grids, w_grids = (
                        input_ids[0][self.image_start_token_id_index + 1] - 8804,
                        input_ids[0][self.image_start_token_id_index + 2] - 8804,
                    )
                    self.h_latent_dim, self.w_latent_dim = h_grids * 2, w_grids * 2

                if self.guidance_scale == 1.0:
                    return scores

                conditional_logits = scores[0]
                unconditional_logits = scores[1]

                cfg_logits = self.guidance_scale * (conditional_logits - unconditional_logits) + unconditional_logits
                return torch.stack([cfg_logits, cfg_logits], dim=0)

        else:
            print("Something wrong in the decoding process.")

        return torch.stack([scores[0], scores[0]], dim=0)


class MultiModalLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        image_start_token_id=None,
        image_end_token_id=None,
        image_next_line_token_id=None,
        patch_size=None,
        voc_size=None,
    ):
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_next_line_token_id = image_next_line_token_id
        self.image_start_token_id_index = None
        self.patch_size = patch_size
        self.h_latent_dim = None
        self.w_latent_dim = None

        self.vocab_list = [i for i in range(voc_size)]
        self.image_token_list = [i for i in range(4, 8195 + 1)]
        self.suppress_tokens = torch.tensor(
            [x for x in self.vocab_list if x not in self.image_token_list], device="cuda"
        )

        self.vocab_tensor = torch.arange(voc_size, device="cuda")
        self.suppress_token_mask = torch.isin(self.vocab_tensor, self.suppress_tokens)
        self.new_line_force_token_mask = torch.isin(
            self.vocab_tensor, torch.tensor([self.image_next_line_token_id], device="cuda")
        )
        self.eos_image_force_token_mask = torch.isin(
            self.vocab_tensor, torch.tensor([self.image_end_token_id], device="cuda")
        )

        self.flag = False
        self.num_image_start_tokens = None
        self.num_image_end_tokens = None

    def helper(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        self.num_image_start_tokens = (input_ids[0] == self.image_start_token_id).sum()
        self.num_image_end_tokens = (input_ids[0] == self.image_end_token_id).sum()

        # print(self.num_image_start_tokens, self.num_image_end_tokens)

        if self.num_image_start_tokens == self.num_image_end_tokens:
            self.h_latent_dim, self.w_latent_dim = None, None
            self.image_start_token_id_index = None
            return scores

        elif self.num_image_start_tokens == self.num_image_end_tokens + 1:
            self.image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0]
            self.image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0][-1].item()
            
            new_token_num = len(input_ids[0][self.image_start_token_id_index + 1 :])
            if new_token_num >= 2:
                if self.h_latent_dim is None or self.w_latent_dim is None:
                    h_grids, w_grids = (
                        input_ids[0][self.image_start_token_id_index + 1] - 8804,
                        input_ids[0][self.image_start_token_id_index + 2] - 8804,
                    )
                    # print(f"h_grids: {h_grids}, w_grids: {w_grids}")
                    self.h_latent_dim, self.w_latent_dim = h_grids * 2, w_grids * 2
                    print(f"h_latent_dim: {self.h_latent_dim}, w_latent_dim: {self.w_latent_dim}")

                tokens = input_ids[0][self.image_start_token_id_index + 3 :]
                if (len(tokens) + 1) % (self.w_latent_dim + 1) == 0:
                    new_line_constrained_scores = torch.full_like(scores, -math.inf)
                    new_line_constrained_scores[self.image_next_line_token_id] = 0
                    print(f"new line: {len(tokens)+1}")
                    return new_line_constrained_scores
                elif (len(tokens) + 1) == (self.w_latent_dim + 1) * self.h_latent_dim + 1:
                    eos_image_constrained_scores = torch.full_like(scores, -math.inf)
                    eos_image_constrained_scores[self.image_end_token_id] = 0
                    print(f"eos image: {len(tokens)+1}")
                    return eos_image_constrained_scores
                elif (len(tokens) + 1) % (self.w_latent_dim + 1) != 0:
                    # force to generate image tokens only
                    image_constrained_scores = torch.where(self.suppress_token_mask, -float("inf"), scores)
                    return image_constrained_scores
        else:
            print("Something wrong in the decoding process.")

        return scores

    # @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, prompt_token_ids: torch.LongTensor, past_token_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        input_ids = torch.LongTensor([prompt_token_ids[0] + past_token_ids[0]])
        scores_processed = self.helper(input_ids, scores[0])
        return torch.stack([scores_processed, scores_processed], dim=0)
        
class InterleavedTopKLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements. Often used together
    with [`TemperatureLogitsWarper`] and [`TopPLogitsWarper`].
    """

    def __init__(
        self,
        image_top_k: int,
        text_top_k: int,
        image_start_token_id=None,
        image_end_token_id=None,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        if not isinstance(text_top_k, int) or text_top_k <= 0:
            raise ValueError(f"`text_top_k` has to be a strictly positive integer, but is {text_top_k}")
        if not isinstance(image_top_k, int) or text_top_k <= 0:
            raise ValueError(f"`image_top_k` has to be a strictly positive integer, but is {image_top_k}")

        self.image_top_k = max(image_top_k, min_tokens_to_keep)
        self.text_top_k = max(text_top_k, min_tokens_to_keep)
        self.filter_value = filter_value

        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id

        self.flag = False
        self.num_image_start_tokens = None
        self.num_image_end_tokens = None

    def helper(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        self.num_image_start_tokens = (input_ids[0] == self.image_start_token_id).sum()
        self.num_image_end_tokens = (input_ids[0] == self.image_end_token_id).sum()

        if self.num_image_start_tokens == self.num_image_end_tokens + 1:
            # Image generation
            top_k = min(self.image_top_k, scores.size(-1))
        else:
            # Text generation
            top_k = min(self.text_top_k, scores.size(-1))  # Safety check
        
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)

        return scores_processed

    def __call__(self, prompt_token_ids: torch.LongTensor, past_token_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        input_ids = torch.LongTensor([prompt_token_ids[0] + past_token_ids[0]])
        scores_processed = self.helper(input_ids, scores[0])
        return torch.stack([scores_processed, scores_processed], dim=0)

class FlexARInferenceSolver:
    @classmethod
    def get_args_parser(cls):
        parser = argparse.ArgumentParser("xllmx Inference", add_help=False)
        parser.add_argument("--model_path", type=str)
        parser.add_argument("--precision", type=str, choices=["fp16", "bf16", "tf32"], default="bf16")

        return parser

    def __init__(self, model_path, precision, target_size=512, max_num_seqs=24):
        self.max_num_seqs = max_num_seqs
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        self.item_processor = FlexARItemProcessor(target_size=target_size)

        self.model = LLM(
            model=model_path,
            dtype=self.dtype,
            max_num_seqs=self.max_num_seqs,
        )

    def get_streamer(self):
        return TextStreamer(self.item_processor.tokenizer)

    @torch.no_grad()
    def generate(
        self,
        images: Image.Image | str | List[Union[Image.Image, str]],
        qas,
        max_gen_len,
        temperature,
        logits_processor=None,
        streamer=None,
        return_images=False,
    ):

        conversations = []
        for q, a in qas:
            entry = []
            entry.append(
                {
                    "from": "human",
                    "value": q,
                }
            )
            entry.append(
                {
                    "from": "gpt",
                    "value": a,
                }
            )
            conversations.append(entry)
        
        prompts = []
        for entry in conversations:
            item = {"image": [], "conversations": entry}
            _prompt = self.item_processor.process_item(item)
            prompt = []
            for value in _prompt:
                if isinstance(value, int):
                    prompt.append(value)
                else:
                    prompt += value["input_ids"]
            prompts.append(torch.tensor(prompt, dtype=torch.int64).tolist())
            prompts.append([0])

        if logits_processor is None:
            logits_processor = self.create_logits_processor()
        
        assert self.max_num_seqs % 2 == 0, "max_num_seqs should be an even number."

        results = []
        missing_indices = []
        prompts_processed = 0

        with tqdm(total=len(prompts)) as pbar:
            while prompts_processed < len(prompts):
                prompts_batch = prompts[prompts_processed : prompts_processed + self.max_num_seqs]

                try:
                    generation_result = self.model.generate(
                        prompt_token_ids=prompts_batch,
                        sampling_params=SamplingParams(
                            logits_processors=logits_processor,
                            max_tokens=max_gen_len,
                            temperature=temperature,
                            stop_token_ids=[8710],
                        ),
                    )
                except Exception as e:
                    print(f"Exception caught: {e}")
                    missing_indices.append(list(range(prompts_processed, prompts_processed + self.max_num_seqs)))
                    continue

                for i in range(0, len(generation_result), 2):
                    if len(generation_result[i].outputs[0].token_ids) > 0 and generation_result[i].outputs[0].token_ids[-1] == 8710:
                        generation_result[i].outputs[0].token_ids = generation_result[i].outputs[0].token_ids[:-1]
                    
                    if return_images:
                        raise NotImplementedError("return_images is not implemented yet. Please use decode_image.py to decode the images.")
                    else:
                        results.append({
                            "prompt": qas[i//2][0],
                            "prompt_token_ids": generation_result[i].prompt_token_ids,
                            "out_token_ids": generation_result[i].outputs[0].token_ids
                        })

                prompts_processed += self.max_num_seqs
                pbar.update(self.max_num_seqs)
        
        return results, missing_indices
        
    def decode_ids(self, tokens: List[int]):
        generated_images = []
        generation_result_processed = []
        i = 0
        while i < len(tokens):
            token_id = tokens[i]
            if token_id == self.item_processor.token2id(self.item_processor.image_start_token):
                cache = []
                for j in range(i + 1, len(tokens)):
                    if tokens[j] != self.item_processor.token2id(self.item_processor.image_end_token):
                        cache.append(tokens[j])
                        i = j + 1
                    else:
                        image = self.decode_image(cache)
                        generated_images.append(image)
                        generation_result_processed.append(self.item_processor.token2id("<|image|>"))
                        i = j + 1
                        break
            else:
                generation_result_processed.append(token_id)
                i += 1

        generated = self.item_processor.tokenizer.decode(generation_result_processed)

        return generated, generated_images

    def decode_image(self, tokens: List[int]):
        return self.item_processor.decode_image(tokens)

    @staticmethod
    def create_image_grid(images, rows, cols):
        width, height = images[0].size

        grid_img = Image.new("RGB", (cols * width, rows * height))

        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            grid_img.paste(img, (col * width, row * height))

        return grid_img

    def create_logits_processor(self, cfg=3.0, image_top_k=2000, text_top_k=10):
        logits_processor = LogitsProcessorList()

        cfg_processor = LLMImageStartTriggeredUnbatchedClassifierFreeGuidanceLogitsProcessor(
            guidance_scale=cfg,
            model=self.model,
            image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
            image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
            image_next_line_token_id=self.item_processor.token2id(self.item_processor.new_line_token),
            patch_size=32,
        )

        candidate_processor = MultiModalLogitsProcessor(
            image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
            image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
            image_next_line_token_id=self.item_processor.token2id(self.item_processor.new_line_token),
            patch_size=32,
            voc_size=65536,
            # voc_size=self.model.config.vocab_size,
        )

        topk_processor = InterleavedTopKLogitsWarper(
            image_top_k=image_top_k,
            text_top_k=text_top_k,
            image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
            image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
        )

        logits_processor.append(cfg_processor)
        logits_processor.append(candidate_processor)
        logits_processor.append(topk_processor)

        return logits_processor


if __name__ == "__main__":
    parser = FlexARInferenceSolver.get_args_parser()
    args = parser.parse_args()
    solver = FlexARInferenceSolver(**vars(args))
