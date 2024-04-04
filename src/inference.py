import argparse
import os
from glob import glob

import torch

from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                       UniPCMultistepScheduler)
from diffusers.utils import load_image

args = argparse.ArgumentParser()
args.add_argument('--base_model_path', type=str, default='runwayml/stable-diffusion-v1-5')
args.add_argument('--controlnet_path', type=str, default='/path/to/depth2underwater/controlnet')
args.add_argument('--depth_dir', type=str, default='/path/to/terrestrial/depth')
args.add_argument('--output_dir', type=str, default='/path/to/output/folder')
args.add_argument('--sample_num', type=int, default=5)
args = args.parse_args()

controlnet = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=torch.float16, use_safetensors=True, local_file_only=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    args.base_model_path, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# comment following line if xformers is not installed
# pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()


paths = glob(os.path.join(args.depth_dir, '*.png'))

prompts = ["an underwater view of Atlantis", "a corner of lost Atlantis"]
generator = torch.manual_seed(0)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)

for p in paths:
    filename = os.path.basename(p)
    print(filename)
    control_image = load_image(p)
    # generate image
    index = 0
    for prompt in prompts:
        for i in range(args.sample_num):
            image = pipe(prompt, num_inference_steps=20, generator=generator, image=control_image, guidance_scale=5).images[0]
            image.save(os.path.join(args.output_dir, "{:s}_{:02d}.png".format(filename[:-4], index)))
            index += 1