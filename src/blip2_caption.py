import argparse
import json
import os
from glob import glob

import torch
from lavis.models import load_model_and_preprocess
from PIL import Image

args = argparse.ArgumentParser()
args.add_argument('--input', type=str, default='/path/to/underwater/folder')
args.add_argument('--output', type=str, default='/path/to/triplets/folder')
args = args.parse_args()

paths = glob(os.path.join(args.input, '*.png'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="caption_coco_opt6.7b", is_eval=True, device=device)
# model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="coco_base", is_eval=True, device=device)

results = []

for p in paths:
    # load sample image
    raw_image = Image.open(p).convert("RGB")    
    
    # preprocess the image
    # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    # generate caption
    caption = model.generate({"image": image})[0]

    entry = {}
    entry['text'] = caption
    entry['image'] = os.path.basename(p)
    entry['conditioning_image'] = os.path.basename(p)

    print(entry)
    results.append(entry)

# save results
with open(os.path.join(args.output, 'metadata.jsonl'), 'a') as j:
    for r in results:
        j.write(json.dumps(r) + '\n')