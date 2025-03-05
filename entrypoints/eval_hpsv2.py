import argparse, os, csv, json, re, time
from glob import glob
from tqdm import tqdm

from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import numpy as np

import hpsv2
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--prompt_path', type=str, required=True)
    return parser

def run_eval_hpsv2(args):

    image_fnames = glob(os.path.join(args.image_path, '**', '*.jpg'), recursive=True) + \
                    glob(os.path.join(args.image_path, '**', '*.png'), recursive=True)

    prompts = []
    if args.prompt_path.endswith('.tsv'):
        with open(args.prompt_path, 'r') as f:
            tsv_reader = csv.DictReader(f, delimiter='\t')
            for row in tsv_reader:
                prompts.append(row['Prompt'])
    elif args.prompt_path.endswith('.json'):
        with open(args.prompt_path, 'r') as f:
            captions = json.load(f)
            for caption in captions:
                prompts.append(caption)
    elif args.prompt_path.endswith('.csv'):
        with open(args.prompt_path, 'r') as f:
            csv_reader = csv.DictReader(f)  # Defaults to ',' as delimiter
            for row in csv_reader:
                prompts.append(row['Prompt'])
    else:
        raise ValueError("Prompt file should be either .tsv or .json")

    hpsv2_scores = []
    for image_fname in tqdm(image_fnames):
        start = 0
        match = re.search(r"(prompt|image)_(\d{1,4})\.png", image_fname)
        if match is None:
            match = re.search(r"(\d{1,6})\.png", image_fname)
            idx = int(match.group(1))
        else:
            idx = int(match.group(2))
        if idx >= len(prompts):
            continue

        pil_image = Image.open(image_fname)
        prompt = prompts[start+idx]
        hpsv2_scores.append(hpsv2.score(pil_image, prompt, hps_version="v2.1"))

    print("Image Path:", args.image_path)
    print(np.mean(hpsv2_scores))

if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

    run_eval_hpsv2(args)
