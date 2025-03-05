import re
import os
import json, csv
import random
import argparse
import traceback

import torch
import numpy as np

import models.drafters.choices as choices

from itertools import product
from torchvision.utils import save_image

from tqdm import tqdm

USE_EXPERIMENTAL_FEATURES = os.getenv("USE_EXPERIMENTAL_FEATURES", "0") == "1"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model to use for image generation",
                        default="lumina_mgpt")
    parser.add_argument("--model_type", type=str, help="Model type; choices: ['eagle', 'base', 'vllm']",
                        default="eagle")
    parser.add_argument("--model_path", type=str, help="Path to the model",
                        default="Alpha-VLLM/Lumina-mGPT-7B-768")
    parser.add_argument("--drafter_path", type=str, help="Path to the drafter model",
                        default="ckpts/lumina_mgpt/trained_drafters/lumina_mgpt_7b_768_lr0.0001_p_w0.1"
                        "_bsz2_gradacc_8_epochs20_noise_uniform_std0.2_coupled_False_cfgloss_False"
                        "_cfgscale_3.0_embed_upscale1.0_mscoco2017train30k/state_20")
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--target_size", type=int, default=768)

    parser.add_argument("--prompt", type=str, help="Prompt for image generation",
                        default="PartiPrompts")
    parser.add_argument("--num_images", type=int, help="Number of images to generate",
                        default=10)
    parser.add_argument("--slice", type=str, help="Slice of prompts to use; format: 'start-end'",
                        default=None)
    parser.add_argument("--output_dir", type=str, help="Output directory for generated images",
                        default="generated_images")
    
    parser.add_argument("--set_seed", action="store_true", help="Set random seed for reproducibility")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation")
    parser.add_argument("--top_k", type=int, default=2000, help="Top-k for generation")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for generation")
    parser.add_argument("--cfg", type=float, default=3.0, help="CFG for generation")
    
    # LANTERN-specific arguments
    parser.add_argument("--lantern", action="store_true", help="Use LANTERN for image generation")
    parser.add_argument("--lantern_k", type=int, default="1000", help="Value of k for LANTERN")
    parser.add_argument("--lantern_delta", type=float, default="0.1", help="Value of delta for LANTERN")
    parser.add_argument("--grid_search", action="store_true", help="Run grid search for LANTERN hyperparameters")

    parser.add_argument("--static_tree", action="store_true", help="Use static tree based drafting")
    parser.add_argument("--eagle_version", type=int, default=1, help="EAGLE version")
    parser.add_argument("--cfg_mode", type=str, default="sequential", help="CFG mode")

    # Experimental arguments
    parser.add_argument("--tree_choices", type=str, help="Tree choice for LANTERN",
                        default="mc_sim_7b_63")
    parser.add_argument("--drafter_top_k", type=int, default=None, help="Top-k for drafter")

    # legacy arguments
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for image generation")
    parser.add_argument("--end_idx", type=int, default=10000, help="End index for image generation")

    return parser

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model(args):
    if args.model == "lumina_mgpt":
        if args.model_type == "vllm":
            from models.base_models.lumina_mgpt.vllm_inference_solver import FlexARInferenceSolver
            model = FlexARInferenceSolver(
                model_path=args.model_path,
                precision=args.precision,
                target_size=args.target_size,
                max_num_seqs=24,
            )

        elif args.model_type == "base":
            from models.base_models.lumina_mgpt.inference_solver import FlexARInferenceSolver
            model = FlexARInferenceSolver(
                model_path=args.model_path,
                precision=args.precision,
                target_size=args.target_size,
                cfg_mode=args.cfg_mode,
            )

        elif args.model_type == "eagle":
            from models.base_models.lumina_mgpt.eagle_inference_solver import FlexARInferenceSolver
            model = FlexARInferenceSolver(
                model_path=args.model_path,
                drafter_path=args.drafter_path,
                precision=args.precision,
                target_size=args.target_size,
                cfg_mode=args.cfg_mode,
                eagle_version=args.eagle_version,
            )
        else:
            raise ValueError(f"Model type {args.model_type} is not supported for model {args.model}")
    elif args.model == "anole":
        if args.model_type == "vllm":
            raise NotImplementedError("VLLM model is not supported for Anole model")
        
        elif args.model_type == 'base':
            from models.kv_variants.modeling_anole_kv import ChameleonForConditionalGeneration
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
            model = ChameleonForConditionalGeneration.from_pretrained(args.model_path).to(dtype=dtype, device='cuda')
            model.eval()
        
        elif args.model_type == 'eagle':
            from models.ea_model_anole import EaModel
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
            model = EaModel.from_pretrained(base_model_path=args.model_path, ea_model_path=args.drafter_path).to(dtype=dtype, device='cuda')
            model.eval()
        
        else:
            raise ValueError(f"Model type {args.model_type} is not supported for model {args.model}")
    elif "llamagen" in args.model:
        if args.model_type == "vllm":
            raise NotImplementedError("VLLM model is not supported for LLAMAGen model")
        
        elif args.model_type == 'base':
            from models.kv_variants.modeling_llamagen_kv import LlamaForCausalLM
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
            model = LlamaForCausalLM.from_pretrained(args.model_path).to(dtype=dtype, device='cuda')
            model.eval()
        
        elif args.model_type == 'eagle':
            from models.ea_model_llamagen import EaModel
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
            model = EaModel.from_pretrained(base_model_path=args.model_path, ea_model_path=args.drafter_path).to(dtype=dtype, device='cuda')
            model.eval()
        else:
            raise ValueError(f"Model type {args.model_type} is not supported for model {args.model}")
    else:
        raise NotImplementedError(f"Model {args.model} is not supported")
    
    return model

def load_prompts(args):
    prompts = []
    if args.prompt == "PartiPrompts":
        with open('data/prompts/PartiPrompts.tsv', 'r') as f:
            tsv_reader = csv.DictReader(f, delimiter='\t')
            for row in tsv_reader:
                prompts.append(row['Prompt'])
    elif args.prompt == "MSCOCO2017Val":
        with open('data/prompts/captions_val2017_longest.json', 'r') as f:
            captions = json.load(f)
            for caption in captions:
                prompts.append(caption)
    elif args.prompt == "MSCOCO2014Val":
        with open('data/prompts/captions_val_2014.json', 'r') as f:
            captions = json.load(f)
            for caption in captions:
                prompts.append(caption)
    elif args.prompt == "MSCOCO2017Train":
        with open('data/prompts/captions_train2017_extracted.json', 'r') as f:
            captions = json.load(f)
            for caption in captions:
                prompts.append(caption['caption'])
    elif args.prompt == "SJDPrompts":
        with open('data/prompts/SJDPrompts.tsv', 'r') as f:
            tsv_reader = csv.DictReader(f, delimiter='\t')
            for row in tsv_reader:
                prompts.append(row['Prompt'])
    else:
        # Single prompt input
        prompts = [args.prompt] * args.num_images

    if args.slice is not None:
        assert re.match(r'^\d+-\d+$', args.slice), f"Invalid format: '{args.slice}'. Expected format is 'start-end'."

        start, end = map(int, args.slice.split('-'))
        assert start < end, f"Invalid range: '{args.slice}'. Start value must be less than end value."
        assert start >= 0 and end >= 0, "Slice values must be non-negative."

        prompts = prompts[start:end]
    
    if args.num_images < len(prompts):
        print(f"Number of images to generate is less than the number of prompts. Sampling {args.num_images} prompts.")
        prompts = random.sample(prompts, args.num_images)
    else:
        print(f"Number of images to generate is greater than the number of prompts. Generating only {len(prompts)} images and no sampling.")
        pass
    
    return prompts

def generate_and_save_image(model, model_name, prompt, img_save_path, **kwargs):
    # print(f"Generating image for prompt: {prompt}")
    if model_name == "lumina_mgpt":
        generate_params = {
            "images": [],
            "qas": [[prompt, None]],
            "max_gen_len": 2354,
            "temperature": kwargs["temperature"],
            "top_k": kwargs["top_k"],
            "cfg_scale": kwargs["cfg"],
        }
    elif model_name in ["anole", "llamagen", "llamagen2"]:
        if model_name == "llamagen":
            max_gen_len = 256
        else:
            max_gen_len = 1024
        generate_params = {
            "prompt": [prompt],
            "max_length": max_gen_len,
            "temperature": kwargs["temperature"],
            "top_k": kwargs["top_k"],
            "top_p": kwargs["top_p"],
            "cfg": kwargs["cfg"],
            "static_tree": kwargs["static_tree"],
        }
    else:
        raise NotImplementedError(f"Model {model_name} is not supported for image generation.")
    
    if "lantern" in kwargs:
        generate_params["lantern"] = kwargs.get("lantern", False)
        generate_params["lantern_k"] = kwargs.get("lantern_k", 1000)
        generate_params["lantern_delta"] = kwargs.get("lantern_delta", 0.1)

    if USE_EXPERIMENTAL_FEATURES:
        generate_params["tree_choices"] = kwargs["tree_choices"]
        generate_params["drafter_top_k"] = kwargs["drafter_top_k"]

    generated_tokens, step_compression, latency = model.generate(**generate_params)
    _, generated_image = model.decode_ids(generated_tokens)

    if model_name in ["lumina_mgpt", "anole"]:
        generated_image[0].save(img_save_path, "png")
    elif "llamagen" in model_name:
        save_image(generated_image, img_save_path, normalize=True, value_range=(-1, 1))

    return step_compression, latency

def run_generate_image(args):
    assert args.model_type != "vllm", "VLLM model is not supported for single image generation"

    if args.set_seed:
        set_seed(args.random_seed)
    
    model = load_model(args)
    prompts = load_prompts(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    global_statistics = {}  
    for idx, prompt in tqdm(enumerate(prompts), total=len(prompts)):
        if idx < args.start_idx or idx >= args.end_idx:
            continue
        if args.model == "lumina_mgpt":
            q1 = f"Generate an image of 768x768 according to the following prompt:\n{prompt}"
        else:
            q1 = prompt

        generate_image_kwargs = {
            "model" : model,
            "model_name" : args.model,
            "prompt" : q1,
            "temperature" : args.temperature,
            "top_k" : args.top_k,
            "top_p" : args.top_p,
            "cfg" : args.cfg,
            "lantern": args.lantern,
            "lantern_k": args.lantern_k,
            "lantern_delta": args.lantern_delta,
            "img_save_path": f"{args.output_dir}/prompt_{idx}.png",
            "static_tree": args.static_tree,
        }

        if USE_EXPERIMENTAL_FEATURES:
            try:
                tree_choices = getattr(choices, args.tree_choices)
            except AttributeError:
                print(f"Tree choices {args.tree_choices} is not a valid choice")
                return
            
            generate_image_kwargs["tree_choices"] = tree_choices
            generate_image_kwargs["drafter_top_k"] = args.drafter_top_k
        
        step_compression, latency = generate_and_save_image(**generate_image_kwargs)

        statistics = {
            "prompt": prompt,
            "step_compression": step_compression,
            "latency": latency
        }

        global_statistics[f"prompt_{idx}"] = statistics

    with open(f"{args.output_dir}/global_statistics_{args.start_idx}_{args.end_idx}.json", "w") as f:
        json.dump(global_statistics, f, indent=4)

    with open(f"{args.output_dir}/generation_configs.json", "w") as f:
        json.dump(vars(args), f, indent=4)

if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

    run_generate_image(args)