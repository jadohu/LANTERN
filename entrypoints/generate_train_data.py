import os
import json
import numpy as np
import argparse

import torch

from typing import Dict, Optional, Sequence
from tqdm import tqdm

from torch.utils.data import Dataset
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Generate data for drafter training')
    
    parser.add_argument('--model', type=str, help="model type",
                        default="lumina_mgpt")
    parser.add_argument('--data_path', type=str, help="data path for image tokens",
                        default="data/self_distilled_data/lumina_mgpt_vllm_generated_20000-40000.json")
    parser.add_argument('--output_dir', type=str, default='data/drafter_train_data/lumina_mgpt')
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument("--precision", type=str, default="bf16")

    return parser

class SupervisedDataset(Dataset):
    def __init__(self, data_path, model, uncond_embedding=None):
        super(SupervisedDataset, self).__init__()
        self.model = model

        if model == "lumina_mgpt" or model == "anole":
            """
                self.dataset = [
                    {"prompt": ..., "token_ids": ...},
                    {"prompt": ..., "token_ids": ...},
                    ...
                ]
            """
            with open(data_path, 'r') as f:
                self.dataset = json.load(f)
        elif "llamagen" in self.model:
            self.code_data = sorted(os.listdir(os.path.join(data_path, "codes")))
            self.text_data = sorted(os.listdir(os.path.join(data_path, "text_features")))
            self.code_base_path = os.path.join(data_path, "codes")
            self.text_base_path = os.path.join(data_path, "text_features")
            self.uncond_embedding = uncond_embedding
            self.cond_length = uncond_embedding.shape[0]
            input_ids = np.load(os.path.join(self.code_base_path, self.code_data[0]))
            input_ids = torch.from_numpy(input_ids).long()
            self.input_length = input_ids.shape[1]
        else:
            raise NotImplementedError(f"Model {model} not supported")
        
    def __len__(self):
        if self.model == "lumina_mgpt" or self.model == "anole":
            return len(self.dataset)
        elif "llamagen" in self.model:
            return len(self.code_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.model == "lumina_mgpt" or self.model == "anole":
            return {"prompt_token_ids" : torch.tensor(self.dataset[i]["prompt_token_ids"]).long().unsqueeze(0),
                    "out_token_ids": torch.tensor(self.dataset[i]["out_token_ids"]).long().unsqueeze(0)}
        elif "llamagen" in self.model:
            assert os.path.basename(self.code_data[i]) == os.path.basename(self.text_data[i])
            input_ids = np.load(os.path.join(self.code_base_path, self.code_data[i]))
            input_ids = torch.from_numpy(input_ids).long()
            cond_idx = np.load(os.path.join(self.text_base_path, self.text_data[i]))
            if random.random() < 0.1:
                cond_idx = self.uncond_embedding.clone().detach().unsqueeze(0)
                attention_mask = torch.ones((1, self.cond_length+self.input_length))
            else:
                cond_idx = torch.from_numpy(cond_idx)
                attention_mask = torch.ones((1, self.cond_length+self.input_length))
                attention_mask[0, :self.cond_length - cond_idx.shape[1]] = 0
                cond_padding = torch.zeros((1, self.cond_length - cond_idx.shape[1], cond_idx.shape[2]))
                cond_idx = torch.cat([cond_padding, cond_idx], dim=1)
            loss_mask = torch.ones((1, self.cond_length+self.input_length))
            loss_mask[:, :self.cond_length] = 0
            
            return dict(
                cond_idx=cond_idx,
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
            )

    def shuffle(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        if self.model == "lumina_mgpt" or self.model == "anole":
            perm = np.random.permutation(len(self.dataset))
            self.dataset = [self.dataset[i] for i in perm]
        elif "llamagen" in self.model:
            perm = np.random.permutation(len(self.code_data))
            self.code_data = [self.code_data[i] for i in perm]
            self.text_data = [self.text_data[i] for i in perm]
        else:
            raise NotImplementedError(f"Model {self.model} not supported")
        return self

    def select(self, indices: Sequence[int]):
        if self.model == "lumina_mgpt" or self.model == "anole":
            self.dataset = [self.dataset[i] for i in indices]
        elif "llamagen" in self.model:
            self.code_data = [self.code_data[i] for i in indices]
            self.text_data = [self.text_data[i] for i in indices]
        return self

@torch.no_grad()
def generate_data(model, data, model_type):
    if model_type == "lumina_mgpt":
        if (data['out_token_ids'][0][:3] == torch.tensor([8197, 8828, 8828])).sum().item() != 3:
            print(data['out_token_ids'][0][:3])
            return None

        prompt_token_ids = data["prompt_token_ids"]
        out_token_ids = data["out_token_ids"]

        cond_input_ids = torch.cat([prompt_token_ids, out_token_ids], dim=-1)
        cond_outputs = model(input_ids=cond_input_ids.cuda(), output_hidden_states=True)

        uncond_input_ids = out_token_ids
        uncond_outputs = model(input_ids=uncond_input_ids.cuda(), output_hidden_states=True)

        return {"cond_input_ids": cond_input_ids.cpu()[0], "cond_hidden_states": cond_outputs.hidden_states[-1].cpu()[0],
                "uncond_input_ids": uncond_input_ids.cpu()[0], "uncond_hidden_states": uncond_outputs.hidden_states[-1].cpu()[0]}
    elif model_type == "anole":
        prompt_token_ids = data["prompt_token_ids"]
        out_token_ids = data["out_token_ids"]

        cond_input_ids = torch.cat([prompt_token_ids, torch.tensor([[8710, 8197]]), out_token_ids], dim=-1)
        cond_outputs = model(input_ids=cond_input_ids.cuda(), output_hidden_states=True)

        uncond_input_ids = torch.cat([torch.tensor([[0, 8197]]), out_token_ids], dim=-1)
        uncond_outputs = model(input_ids=uncond_input_ids.cuda(), output_hidden_states=True)

        return {"cond_input_ids": cond_input_ids.cpu()[0], "cond_hidden_states": cond_outputs.hidden_states[-1].cpu()[0],
                "uncond_input_ids": uncond_input_ids.cpu()[0], "uncond_hidden_states": uncond_outputs.hidden_states[-1].cpu()[0]}
    elif "llamagen" in model_type:
        input_ids=data["input_ids"]
        cond_idx=data["cond_idx"].to(model.dtype)
        loss_mask=data["loss_mask"]
        attention_mask = data["attention_mask"]
        outs_big = model(cond_idx=cond_idx.cuda(), input_ids=input_ids.cuda(),attention_mask=attention_mask.cuda(), output_hidden_states=True)
        hidden_state_big = outs_big.hidden_states[-1]
        return {"cond_idx":cond_idx, "input_ids":input_ids.cpu()[0],"hidden_state":hidden_state_big.cpu()[0],
                "loss_mask":loss_mask.cpu()[0],'attention_mask':attention_mask[0]}
    else:
        raise NotImplementedError(f"Model {model_type} not supported")
def writedata(name, data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(name))
    idx=current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')

def run_generate_data(args):
    
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    
    if args.model == "lumina_mgpt":
        from models.base_models.lumina_mgpt.modeling_lumina_mgpt import ChameleonForConditionalGeneration
        model = ChameleonForConditionalGeneration.from_pretrained(
            "Alpha-VLLM/Lumina-mGPT-7B-768",
            device_map="auto",
            torch_dtype=dtype
        )
        uncond_embedding = None
    elif args.model == "anole":
        from models.kv_variants.modeling_anole_kv import ChameleonForConditionalGeneration
        model = ChameleonForConditionalGeneration.from_pretrained(
            "ckpts/anole/Anole-7b-v0.1-hf",
        ).to(dtype=dtype, device='cuda')
        uncond_embedding = None
    elif args.model == "llamagen":
        from models.kv_variants.modeling_llamagen_kv import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained('ckpts/llamagen/LlamaGen-T2I').to(dtype=dtype, device='cuda')
        uncond_embedding = model.model.cls_embedding.uncond_embedding.to(dtype=dtype, device='cuda')
    elif args.model == "llamagen2":
        from models.kv_variants.modeling_llamagen_kv import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained('ckpts/llamagen/LlamaGen-T2I-2').to(dtype=dtype, device='cuda')
        uncond_embedding = model.model.cls_embedding.uncond_embedding.to(dtype=dtype, device='cuda')
    else:
        raise NotImplementedError(f"Model {args.model} not supported")
    model.eval()
    
    ds = SupervisedDataset(args.data_path, args.model, uncond_embedding)
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(min(args.num_samples, len(ds))))
    

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for data in tqdm(ds):
        outdata = generate_data(model, data, args.model)
        if outdata is not None:
            writedata(args.output_dir, outdata)

if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()
    
    run_generate_data(args)
