import argparse
import logging
import math
import os
import shutil
import json
from glob import glob
from einops import rearrange
from omegaconf import OmegaConf
import numpy as np
from tqdm import tqdm
from packaging import version
from functools import partial
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet3DConditionModel, UNetSpatioTemporalConditionModel, DDIMScheduler

from echosyn.common.datasets import instantiate_dataset
from echosyn.common import (
    padf, unpadf, 
    load_model
)

"""
python echosyn/lidm/sample.py \
    --config echosyn/lidm/configs/dynamic.yaml \
    --unet experiments/lidm_dynamic/checkpoint-500000/unet_ema \
    --vae models/vae \
    --output samples/dynamic \
    --num_samples 50000 \
    --batch_size 128 \
    --num_steps 64 \
    --save_latent \
    --seed 0
"""

if __name__ == "__main__":
    # 1 - Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    parser.add_argument("--unet", type=str, default=None, help="Path unet checkpoint.")
    parser.add_argument("--vae", type=str, default=None, help="Path vae checkpoint.")
    parser.add_argument("--output", type=str, default='.', help="Output directory.")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples to generate.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--num_steps", type=int, default=128, help="Number of steps.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--save_latent", action="store_true", help="Save latents.")
    parser.add_argument("--ddim", action="store_true", help="Save video.")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    # 2 - Load models
    unet = load_model(args.unet)
    vae = load_model(args.vae)

    # 3 - Load scheduler
    scheduler_kwargs = OmegaConf.to_container(config.noise_scheduler)
    scheduler_klass_name = scheduler_kwargs.pop("_class_name")
    if args.ddim:
        print("Using DDIMScheduler")
        scheduler_klass_name = "DDIMScheduler"
        scheduler_kwargs.pop("variance_type")
    scheduler_klass = getattr(diffusers, scheduler_klass_name, None)
    assert scheduler_klass is not None, f"Could not find scheduler class {scheduler_klass_name}"
    scheduler = scheduler_klass(**scheduler_kwargs)
    scheduler.set_timesteps(args.num_steps)
    timesteps = scheduler.timesteps

    # 5 - Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    generator = torch.Generator(device=device).manual_seed(config.seed) if config.seed is not None else None
    unet = unet.to(device, dtype)
    vae = vae.to(device, torch.float32)
    unet.eval()
    vae.eval()

    format_input = padf
    format_output = unpadf

    B, C, H, W = args.batch_size, config.unet.out_channels, config.unet.sample_size, config.unet.sample_size

    forward_kwargs = {
        "timestep": -1,
    }
    
    sample_index = 0

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "images"), exist_ok=True)
    if args.save_latent:
        os.makedirs(os.path.join(args.output, "latents"), exist_ok=True)
    finished = False

    # 6 - Generate samples
    with torch.no_grad():
        for _ in tqdm(range(int(np.ceil(args.num_samples/args.batch_size)))):
            if finished:
                break

            latents = torch.randn((B, C, H, W), device=device, dtype=dtype, generator=generator)

            with torch.autocast("cuda"):
                for t in timesteps:
                    forward_kwargs["timestep"] =  t
                    latent_model_input = latents
                    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
                    latent_model_input, padding = format_input(latent_model_input, mult=3)
                    noise_pred = unet(latent_model_input, **forward_kwargs).sample
                    noise_pred = format_output(noise_pred, pad=padding)
                    latents = scheduler.step(noise_pred, t, latents).prev_sample
            
            if args.save_latent:
                latents_clean = latents.clone()

            # VAE decode
            rep = [1, 3, 1, 1]
            latents = latents / vae.config.scaling_factor
            images = vae.decode(latents.float()).sample
            images = (images + 1) * 128 # [-1, 1] -> [0, 256]

            # grayscale
            images = images.mean(1).unsqueeze(1).repeat(*rep)

            images = images.clamp(0, 255).to(torch.uint8).cpu()
            images = rearrange(images, 'b c h w -> b h w c')

            # 7 - Save samples
            images = images.numpy()
            for j in range(B):
                
                Image.fromarray(images[j]).save(os.path.join(args.output, "images", f"sample_{sample_index:06d}.jpg"))
                if args.save_latent:
                    torch.save(latents_clean[j].clone(), os.path.join(args.output, "latents", f"sample_{sample_index:06d}.pt"))

                sample_index += 1
                if sample_index >= args.num_samples:
                    finished = True
                    break

    print(f"Finished generating {sample_index} samples.")











