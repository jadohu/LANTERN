import os
import yaml
import argparse

import torch
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Generate codebook')
    parser.add_argument('--model', type=str, default="lumina_mgpt", help="Model type; choices: ['lumina_mgpt']")
    parser.add_argument('--save_path', type=str, default="ckpts/lumina_mgpt/vq_distances", help="Path to save the codebook")

    return parser

def run_generate_codebook(args):
    if args.model == "lumina_mgpt":
        from models.base_models.lumina_mgpt.chameleon_vae_ori.vqgan import VQModel
        
        cfg_path = "ckpts/lumina_mgpt/chameleon/tokenizer/vqgan.yaml"
        ckpt_path = "ckpts/lumina_mgpt/chameleon/tokenizer/vqgan.ckpt"
        with open(cfg_path) as f:
            config = yaml.safe_load(f)
        
        params = config["model"]["params"]
        if "lossconfig" in params:
            del params["lossconfig"]
        params["ckpt_path"] = ckpt_path

        vq_model= VQModel(**params)
    elif args.model == "anole":
        from models.base_models.anole.chameleon_vae_ori.vqgan import VQModel
        
        cfg_path = "ckpts/anole/chameleon/tokenizer/vqgan.yaml"
        ckpt_path = "ckpts/anole/chameleon/tokenizer/vqgan.ckpt"
        with open(cfg_path) as f:
            config = yaml.safe_load(f)
            
        params = config["model"]["params"]
        if "lossconfig" in params:
            del params["lossconfig"]
        params["ckpt_path"] = ckpt_path
        
        vq_model= VQModel(**params)
    elif args.model == "llamagen":
        from models.base_models.llamagen.vq_model import VQ_16
        vq_model = VQ_16(codebook_size=16384, codebook_embed_dim=8)
        checkpoint = torch.load('ckpts/llamagen/vq_ds16_t2i.pt')
        vq_model.load_state_dict(checkpoint['model'])

    else:
        raise NotImplementedError(f"Model {args.model} not implemented yet")

    latents = vq_model.quantize.embedding.weight # (8192, 256)

    distances = torch.cdist(latents, latents, p=2) # (8192, 8192)
    distances.fill_diagonal_(float('inf'))

    k = latents.shape[0] - 1 # k-nearest neighbors
    _, topk_indices = torch.topk(distances, k, dim=-1, largest=False)
    topk_indices_uint16 = topk_indices.to(torch.uint16)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    np.save(os.path.join(args.save_path, f"top_{k}_indices.npy"), topk_indices_uint16.cpu().detach().numpy())

if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    
    run_generate_codebook(args)