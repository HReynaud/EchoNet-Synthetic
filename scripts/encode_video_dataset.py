import argparse
import os
from pathlib import Path
from glob import glob
from einops import rearrange
import numpy as np
from tqdm.auto import tqdm
import cv2, json
import shutil
import pandas as pd

import torch

from diffusers import AutoencoderKL
from echosyn.common import loadvideo, load_model

"""
usage example:

python scripts/encode_video_dataset.py \
    -m models/vae \
    -i datasets/EchoNet-Dynamic \
    -o data/latents/dynamic \
    -g
"""

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.videos = sorted(glob(os.path.join(folder, "*.avi")))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video, fps = loadvideo(self.videos[idx], return_fps=True)
        return video, self.videos[idx], fps

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to model folder")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input folder")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output folder")
    parser.add_argument("-g", "--gray_scale", action="store_true", help="Convert to gray scale", default=False)
    parser.add_argument("-f", "--force_overwrite", action="store_true", help="Overwrite existing latents", default=False)
    args = parser.parse_args()

    # Prepare
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)
    video_in_folder = os.path.abspath(os.path.join(args.input, "Videos"))
    video_out_folder = os.path.abspath(os.path.join(args.output, "Latents"))

    df = pd.read_csv(os.path.join(args.input, "FileList.csv"))
    needs_df_update = False
    if df['Split'].dtype == int:
        print("Updating Split column to string")
        needs_df_update = True
        df['Fold'] = df['Split']

        def split_set(row):
            if row['Fold'] in range(8):
                return 'TRAIN'
            elif row['Fold'] == 8:
                return 'VAL'
            else:
                return 'TEST'

        df['Split'] = df.apply(split_set, axis=1)
        df['FileName'] = df['FileName'].apply(lambda x: x.split('.')[0])

    if not needs_df_update:
        df.to_csv(os.path.join(args.output, "FileList.csv"), index=False)

    print("Loading videos from ", video_in_folder)
    print("Saving latents to ", video_out_folder)

    # Load VAE
    vae = load_model(args.model)
    vae = vae.to(device)
    vae.eval()

    # Load Dataset
    ds = VideoDataset(video_in_folder)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=8)
    print(f"Found {len(ds)} videos")

    batch_size = 32 # number of frames to encode simultaneously

    # for vpath in tqdm(videos):
    for video, vpath, fps in tqdm(dl):

        video = video[0]
        vpath = vpath[0]
        fps = fps[0]

        # output path
        opath = vpath.replace(video_in_folder, "")[1:] # retrieve relative path to input folder, similar to basename but keeps the folders
        opath = opath.replace(".avi", f".pt") # change extension
        opath = os.path.join(video_out_folder, opath) # add output folder

        # check if already exists
        if os.path.exists(opath) and not args.force_overwrite:
            print(f"Skipping {vpath} as {opath} already exists")
            continue

        # load video
        # video = loadvideo(vpath) # B H W C
        video = rearrange(video, "t h w c-> t c h w") # B C H W
        video = video.to(device)
        video = video.float() / 128.0 -1 # normalize to [-1, 1]
        if args.gray_scale:
            video = video.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1) # B C H W

        # encode video
        all_latents = []
        for i in range(0, len(video), batch_size):
            batch = video[i:i+batch_size]
            with torch.no_grad():
                latents = vae.encode(batch).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            latents = latents.detach().cpu()
            all_latents.append(latents)
        
        all_latents = torch.cat(all_latents, dim=0)

        # save
        os.makedirs(os.path.dirname(opath), exist_ok=True)
        torch.save(all_latents, opath)

        if needs_df_update:
            fname = os.path.basename(opath).split('.')[0]
            df.loc[df['FileName'] == fname, ['FileName', 'FrameHeight','FrameWidth','FPS','NumberOfFrames']] = [fname, 112, 112, fps, len(video)]

    if needs_df_update:
        df.to_csv(os.path.join(args.output, "FileList.csv"), index=False)
    print("Done")
