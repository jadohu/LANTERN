import os
import json
import numpy as np
import argparse
from torchvision import transforms

import torch

from typing import Dict, Optional, Sequence
from tqdm import tqdm

from torch.utils.data import Dataset
import random
from PIL import Image
from models.base_models.llamagen.vq_model import VQ_16
from models.base_models.llamagen.t5 import T5Embedder
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Generate data for drafter training')
    parser.add_argument('--model', type=str, default="llamagen", help="Model type; choices: ['llamagen', 'llamagen2', 'anole']")
    parser.add_argument('--data_path', type=str, help="data path for image and caption files",
                        default="data/laion_coco")
    parser.add_argument('--output_dir', type=str, default='data/extracted_code/llamagen')
    parser.add_argument('--num_samples', type=int, default=1000000)

    return parser

def apply_mask(tensor, mask):
    """
    Apply a 1D mask to remove irrelevant rows from a 3D tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape (1, 120, 2048).
        mask (torch.Tensor): Binary mask of shape (1, 120), indicating which rows to keep.

    Returns:
        torch.Tensor: Masked tensor with irrelevant rows removed.
    """
    mask = mask.bool()  # Convert to boolean
    masked_tensor = tensor[:, mask.squeeze(0), :]  # Apply the mask along dimension 1
    return masked_tensor

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class SupervisedDataset(Dataset):
    def __init__(self, data_path, transform=None):
        super(SupervisedDataset, self).__init__()
        self.images = sorted([d for d in os.listdir(data_path) if d.endswith(".jpg")])
        self.captions = sorted([d for d in os.listdir(data_path) if d.endswith(".txt")])
        self.base_path = data_path
        self.transform = transform
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        assert self.images[i].split(".")[0] == self.captions[i].split(".")[0]
        img = Image.open(os.path.join(self.base_path, self.images[i])).convert("RGB")
        caption = open(os.path.join(self.base_path, self.captions[i])).read().strip()
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "caption": caption}
        

    def shuffle(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        perm = np.random.permutation(len(self.images))
        self.images = [self.images[i] for i in perm]
        self.captions = [self.captions[i] for i in perm]
        return self

    def select(self, indices: Sequence[int]):
        self.images = [self.images[i] for i in indices]
        self.captions = [self.captions[i] for i in indices]
        return self

@torch.no_grad()
def generate_data_llamagen(vq_model, t5_model, data):
    caption_embs, emb_masks = t5_model.get_text_embeddings([data['caption']])
    caption_embs = apply_mask(caption_embs, emb_masks)
    caption_embs = caption_embs.detach().cpu().numpy()
    img = data['image'].unsqueeze(0)
    img = img.to(t5_model.device)
    _, _, [_, _, indices] = vq_model.encode(img)
    codes = indices.reshape(img.shape[0], -1)
    codes = codes.detach().cpu().numpy()
    return {
        'caption_emb': caption_embs,
        'codes': codes
    } 

@torch.no_grad()
def generate_data_anole(vq_model, tokenizer, data, device):
    caption = data['caption']
    caption = tokenizer(caption, return_tensors="pt", padding="max_length")
    input_ids = caption['input_ids'][0].tolist()
    img = data['image'].unsqueeze(0)
    img = img.to(device)
    _, _, [_, _, indices] = vq_model.encode(img)
    indices = indices.tolist()
    return {
        "prompt_token_ids": input_ids,
        "out_token_ids": indices
    }
    
def writedata(name, data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(os.path.join(name, "codes")))
    idx=current_length
    np.save(os.path.join(name, os.path.join("text_features", f"{idx}.npy")), data_point['caption_emb'])
    np.save(os.path.join(name, os.path.join("codes", f"{idx}.npy")), data_point['codes'])


def run_extract_code(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model == "llamagen":
        image_size = 256
    else:
        image_size = 512
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    ds = SupervisedDataset(args.data_path, transform)
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(min(len(ds), args.num_samples)))
    if "llamagen" in args.model:
        vq_model = VQ_16(codebook_size=16384, codebook_embed_dim=8)
        vq_model.to(device)
        checkpoint = torch.load('ckpts/llamagen/vq_ds16_t2i.pt')
        vq_model.load_state_dict(checkpoint['model'])
        vq_model.eval()
        del checkpoint

        t5_model = T5Embedder(
            device = device,
            local_cache=True,
            cache_dir='ckpts/llamagen/t5',
            dir_or_name='flan-t5-xl',
            torch_dtype=torch.float16,
            model_max_length=120
        )
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            os.makedirs(os.path.join(args.output_dir, 'codes'))
            os.makedirs(os.path.join(args.output_dir, 'text_features'))
        for data in tqdm(ds):
            outdata = generate_data_llamagen(vq_model, t5_model, data)
            if outdata is not None:
                writedata(args.output_dir, outdata)
    elif args.model == "anole":
        from models.base_models.anole.chameleon_vae_ori.vqgan import VQModel
        from transformers import AutoTokenizer
        
        cfg_path = "ckpts/anole/chameleon/tokenizer/vqgan.yaml"
        ckpt_path = "ckpts/anole/chameleon/tokenizer/vqgan.ckpt"
        with open(cfg_path) as f:
            config = yaml.safe_load(f)
            
        params = config["model"]["params"]
        if "lossconfig" in params:
            del params["lossconfig"]
        params["ckpt_path"] = ckpt_path
        
        vq_model= VQModel(**params)
        vq_model.to(device)
        vq_model.eval()
        tokenizer = AutoTokenizer.from_pretrained("ckpts/anole/Anole-7b-v0.1-hf")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        outdata_list = []
        for data in tqdm(ds):
            outdata = generate_data_anole(vq_model, tokenizer, data, device)
            outdata_list.append(outdata)
        with open(os.path.join(args.output_dir, "data.json"), "w") as f:
            json.dump(outdata_list, f)
    else:
        raise NotImplementedError(f"Model {args.model} not implemented yet")

    

    

if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()
    
    run_extract_code(args)
