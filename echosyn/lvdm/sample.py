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
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet3DConditionModel, UNetSpatioTemporalConditionModel

from echosyn.common.datasets import TensorSet, ImageSet
from echosyn.common import (
        pad_reshape, unpad_reshape, padf, unpadf, 
        load_model, save_as_mp4, save_as_gif, save_as_img, save_as_avi,
        parse_formats,
    )

"""
python echosyn/lvdm/sample.py \
    --config echosyn/lvdm/configs/default.yaml \
    --unet experiments/lvdm/checkpoint-500000/unet_ema \
    --vae model/vae \
    --conditioning samples/dynamic/privacy_compliant_latents \
    --output samples/dynamic/privacy_compliant_samples \
    --num_samples 2048 \
    --batch_size 8 \
    --num_steps 64 \
    --min_lvef 10 \
    --max_lvef 90 \
    --save_as avi \
    --frames 192
"""

if __name__ == "__main__":
    # 1 - Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    parser.add_argument("--unet", type=str, default=None, help="Path unet checkpoint.")
    parser.add_argument("--vae", type=str, default=None, help="Path vae checkpoint.")
    parser.add_argument("--conditioning", type=str, default=None, help="Path to the folder containing the conditionning latents.")
    parser.add_argument("--output", type=str, default='.', help="Output directory.")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples to generate.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--num_steps", type=int, default=64, help="Number of steps.")
    parser.add_argument("--min_lvef", type=int, default=10, help="Minimum LVEF.")
    parser.add_argument("--max_lvef", type=int, default=90, help="Maximum LVEF.")
    parser.add_argument("--save_as", type=parse_formats, default=None, help="Save formats separated by commas (e.g., avi,jpg). Available: avi, mp4, gif, jpg, png, pt")
    parser.add_argument("--frames", type=int, default=192, help="Number of frames to generate. Must be a multiple of 32")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    # 2 - Load models
    unet = load_model(args.unet)
    vae = load_model(args.vae)

    # 3 - Load scheduler
    scheduler_kwargs = OmegaConf.to_container(config.noise_scheduler)
    scheduler_klass_name = scheduler_kwargs.pop("_class_name")
    scheduler_klass = getattr(diffusers, scheduler_klass_name, None)
    assert scheduler_klass is not None, f"Could not find scheduler class {scheduler_klass_name}"
    scheduler = scheduler_klass(**scheduler_kwargs)
    scheduler.set_timesteps(args.num_steps)
    timesteps = scheduler.timesteps

    # 4 - Load dataset
    ## detect type of conditioning:
    file_ext = os.listdir(args.conditioning)[0].split(".")[-1].lower()
    assert file_ext in ["pt", "jpg", "png"], f"Conditioning files must be either .pt, .jpg or .png, not {file_ext}"
    if file_ext == "pt":
        dataset = TensorSet(args.conditioning)
    else:
        dataset = ImageSet(args.conditioning, ext=file_ext)
    assert len(dataset) > 0, f"No files found in {args.conditioning} with extension {file_ext}"

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    # 5 - Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    generator = torch.Generator(device=device).manual_seed(config.seed) if config.seed is not None else None
    unet = unet.to(device, dtype)
    vae = vae.to(device, torch.float32)
    unet.eval()
    vae.eval()

    format_input = pad_reshape if config.unet._class_name == "UNetSpatioTemporalConditionModel" else padf
    format_output = unpad_reshape if config.unet._class_name == "UNetSpatioTemporalConditionModel" else unpadf

    B, C, T, H, W = args.batch_size, config.unet.out_channels, config.unet.num_frames, config.unet.sample_size, config.unet.sample_size
    fps = config.globals.target_fps
    # Stitching parameters
    args.frames = int(np.ceil(args.frames/32) * 32)
    if args.frames > T:
        OT = T//2 # overlap 64//2
        TR = (args.frames - T) / 32 # total frames (192 - 64) / 32 = 4
        TR = int(TR + 1) # total repetitions
        NT = (T-OT) * TR + OT # = args.frame
    else:
        OT = 0
        TR = 1
        NT = T

    forward_kwargs = {
        "timestep": -1,
    }

    if config.unet._class_name == "UNetSpatioTemporalConditionModel":
        dummy_added_time_ids = torch.zeros((B*TR, config.unet.addition_time_embed_dim), device=device, dtype=dtype)
        forward_kwargs["added_time_ids"] = dummy_added_time_ids
    
    sample_index = 0

    filelist = []

    os.makedirs(args.output, exist_ok=True)
    for ext in args.save_as:
        os.makedirs(os.path.join(args.output, ext), exist_ok=True)
    finished = False

    pbar = tqdm(total=args.num_samples)

    # 6 - Generate samples
    with torch.no_grad():
        while not finished:
            for cond in dataloader:
                if finished:
                    break

                # Prepare latent noise
                latents = torch.randn((B, C, NT, H, W), device=device, dtype=dtype, generator=generator)

                # Prepare conditioning - lvef
                lvefs = torch.randint(args.min_lvef, args.max_lvef+1, (B,), device=device, dtype=dtype, generator=generator)
                lvefs = lvefs / 100.0
                lvefs = lvefs[:, None, None]
                lvefs = lvefs.repeat_interleave(TR, dim=0)
                forward_kwargs["encoder_hidden_states"] =  lvefs
                # Prepare conditioning - reference frames
                latent_cond_images = cond.to(device, torch.float32)
                if file_ext != "pt":
                    # project image to latent space
                    latent_cond_images = vae.encode(latent_cond_images).latent_dist.sample()
                    latent_cond_images = latent_cond_images * vae.config.scaling_factor
                latent_cond_images = latent_cond_images[:,:,None,:,:].repeat(1,1,NT,1,1) # B x C x T x H x W

                # Denoise the latent
                with torch.autocast("cuda"):
                    for t in timesteps:
                        forward_kwargs["timestep"] = t
                        latent_model_input = scheduler.scale_model_input(latents, timestep=t)
                        latent_model_input = torch.cat((latent_model_input, latent_cond_images), dim=1) # B x 2C x T x H x W
                        latent_model_input, padding = format_input(latent_model_input, mult=3) # B x T x 2C x H+P x W+P

                        # Stitching
                        inputs = torch.cat([latent_model_input[:,r*(T-OT):r*(T-OT)+T] for r in range(TR)], dim=0) # B*TR x T x 2C x H+P x W+P
                        noise_pred = unet(inputs, **forward_kwargs).sample
                        outputs = torch.chunk(noise_pred, TR, dim=0) # TR x B x T x C x H x W
                        noise_predictions = []
                        for r in range(TR):
                            noise_predictions.append(outputs[r] if r == 0 else outputs[r][:,OT:])
                        noise_pred = torch.cat(noise_predictions, dim=1) # B x NT x C x H x W

                        noise_pred = unpad_reshape(noise_pred, pad=padding)
                        latents = scheduler.step(noise_pred, t, latents).prev_sample

                # VAE decode
                latents = rearrange(latents, "b c t h w -> (b t) c h w").cpu()
                latents = latents / vae.config.scaling_factor

                # Decode in chunks to save memory
                chunked_latents = torch.split(latents, args.batch_size, dim=0)
                decoded_chunks = []
                for chunk in chunked_latents:
                    decoded_chunks.append(vae.decode(chunk.float().cuda()).sample.cpu())
                video = torch.cat(decoded_chunks, dim=0) # (B*T) x H x W x C

                # format output
                video = rearrange(video, "(b t) c h w -> b t h w c", b=B)
                video = (video + 1) * 128
                video = video.clamp(0, 255).to(torch.uint8)

                print(video.shape, video.dtype, video.min(), video.max())
                file_lvefs = lvefs.squeeze()[::TR].mul(100).to(torch.int).tolist()
                # save samples
                for j in range(B):
                    # FileName,EF,ESV,EDV,FrameHeight,FrameWidth,FPS,NumberOfFrames,Split
                    filelist.append([f"sample_{sample_index:06d}", file_lvefs[j], 0, 0, video.shape[1], video.shape[2], fps, video.shape[0], "TRAIN"])
                    if "mp4" in args.save_as:
                        save_as_mp4(video[j], os.path.join(args.output, "mp4", f"sample_{sample_index:06d}.mp4"))
                    if "avi" in args.save_as:
                        save_as_avi(video[j], os.path.join(args.output, "avi", f"sample_{sample_index:06d}.avi"))
                    if "gif" in args.save_as:
                        save_as_gif(video[j], os.path.join(args.output, "gif", f"sample_{sample_index:06d}.gif"))
                    if "jpg" in args.save_as:
                        save_as_img(video[j], os.path.join(args.output, "jpg", f"sample_{sample_index:06d}"), ext="jpg")
                    if "png" in args.save_as:
                        save_as_img(video[j], os.path.join(args.output, "png", f"sample_{sample_index:06d}"), ext="png")
                    if "pt" in args.save_as:
                        torch.save(video[j].clone(), os.path.join(args.output, "pt", f"sample_{sample_index:06d}.pt"))
                    sample_index += 1
                    pbar.update(1)
                    if sample_index >= args.num_samples:
                        finished = True
                        break

    df = pd.DataFrame(filelist, columns=["FileName", "EF", "ESV", "EDV", "FrameHeight", "FrameWidth", "FPS", "NumberOfFrames", "Split"])
    df.to_csv(os.path.join(args.output, "FileList.csv"), index=False)
    print(f"Generated {sample_index} samples.")




