import os
import argparse
import json
import random
import shutil

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from PIL import Image

import matplotlib.pyplot as plt
from functools import partial
from echosyn.common.datasets import SimaseUSVideoDataset
from echosyn.privacy.shared import SiameseNetwork


"""
This script is used to apply the privacy filter to a set of synthetic latents.
Example usage:
python echosyn/privacy/apply.py \
    --model experiments/reidentification_dynamic \
    --synthetic samples/dynamic/latents \
    --reference data/latents/dynamic \
    --output samples/dynamic/privatised_latents
"""

def first_frame(vid): 
    return vid[0:1]

def subsample(vid, every_nth_frame): 
    frames = np.arange(0, len(vid), step=every_nth_frame)
    return vid[frames]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Privacy Filter')
    parser.add_argument('--model', type=str, help='Path to the model folder.')
    parser.add_argument('--synthetic', type=str, help='Path to the synthetic latents folder.')
    parser.add_argument('--reference', type=str, help='Path to the real latents folder.')
    parser.add_argument('--output', type=str, help='Path to the output folder.')
    parser.add_argument('--cutoff_precentile', type=float, default=95, help='Cutoff percentile for privacy threshold.')

    args = parser.parse_args()

    # Load real and synthetic latents
    training_latents_csv = os.path.join(args.reference, "FileList.csv")
    training_latents_basepath = os.path.join(args.reference, "Latents")
    normalization =lambda x: (x  - x.min())/(x.max() - x.min()) * 2 - 1  # should be -1 to 1 due to way we trained the model
    ds_train = SimaseUSVideoDataset(phase="training", transform=normalization, latents_csv=training_latents_csv, training_latents_base_path=training_latents_basepath, in_memory=True)
    ds_test = SimaseUSVideoDataset(phase="testing", transform=normalization, latents_csv=training_latents_csv, training_latents_base_path=training_latents_basepath, in_memory=True)

    synthetic_images_paths = os.listdir(args.synthetic)
    synthetic_images = [torch.load(os.path.join(args.synthetic, x)) for x in synthetic_images_paths]

    # convert images to 1 x C x H x W to be consistent in case we want to check videos 
    for i in range(len(synthetic_images)): 
        if len(synthetic_images[i].size()) == 3: 
            synthetic_images[i] = synthetic_images[i].unsqueeze(dim=0)

    synthetic_images = normalization(torch.cat(synthetic_images))
    print(f"Number of synthetic images found: {len(synthetic_images)}")

    # Prepare the images
    train_vid_to_img = first_frame
    test_vid_to_img = first_frame
    train_images = torch.cat([train_vid_to_img(x) for x in tqdm(ds_train)])
    test_images = torch.cat([test_vid_to_img(x) for x in tqdm(ds_test)])
    print(f"Number of real train frames: {len(train_images)}")
    print(f"Number of real test frames: {len(test_images)}")


    # Load Model
    with open(os.path.join(args.model, "config.json")) as config:
        config = config.read()

    config = json.loads(config)
    net = SiameseNetwork(network=config['siamese_architecture'], in_channels=config['n_channels'], n_features=config['n_features'])
    net.load_state_dict(torch.load(os.path.join(args.model, os.path.basename(args.model) + "_best_network.pth")))
    

    print("Sanity Check")
    print(f"Train - Min: {train_images.min()} - Max: {train_images.max()} - shape: {train_images.size()}")
    print(f"Test - Min: {test_images.min()} - Max: {test_images.max()} - shape: {test_images.size()}")
    print(f"Train - Min: {synthetic_images.min()} - Max: {synthetic_images.max()} - shape: {synthetic_images.size()}")

    # Compute the embeddings
    net.eval()
    net = net.cuda()
    bs = 256
    latents_train = []
    latents_test = []
    latents_synth = []
    with torch.no_grad():
        for i in tqdm(np.arange(0, len(train_images), bs), "Computing Train Embeddings"):
            batch = train_images[i:i+bs].cuda()
            latents_train.append(net.forward_once(batch))

        for i in tqdm(np.arange(0, len(test_images), bs), "Computing Test Embeddings"):
            batch = test_images[i:i+bs].cuda()
            latents_test.append(net.forward_once(batch))

        for i in tqdm(np.arange(0, len(synthetic_images), bs), "Computing Synthetic Embeddings"):
            batch = synthetic_images[i:i+bs].cuda()
            latents_synth.append(net.forward_once(batch))

    latents_train = torch.cat(latents_train)
    latents_test = torch.cat(latents_test)
    latents_synth = torch.cat(latents_synth)


    # Automatically determine the privacy threshold
    train_val_corr = torch.corrcoef(torch.cat([latents_train, latents_test])).cpu()
    print(train_val_corr.size())

    closest_train = []
    for i in range(len(train_images)): 
        val_matches = train_val_corr[i, len(train_images):]
        closest_train.append(val_matches.max().cpu())

    tau = np.percentile(torch.stack(closest_train).numpy(), args.cutoff_precentile)
    print(f"Privacy threshold tau: {tau}")

    # Compute the closest matches between synthetic and real images
    closest_test = []
    batch_size = 10000
    latents_synth.cpu()
    closest_loc = []
    for l in np.arange(0, len(latents_synth), batch_size): 
        synth = latents_synth[l:l + batch_size].cuda()
        train_synth_corr = torch.corrcoef(torch.cat([latents_train, synth]))
        for i in range(len(synth)): 
            synth_matches = train_synth_corr[len(train_images)+i, :len(train_images)]
            closest_test.append(synth_matches.max().cpu())
            closest_loc.append(synth_matches.argmax().cpu())
        synth = synth.cpu()

    # Plot the results
    fig, ax = plt.subplots()
    nt, bins, patches = ax.hist(closest_train, 100, density=True, label="Train-Val", alpha=0.5, color="blue")
    ns, bins, patches = ax.hist(closest_test, 100, density=True, label="Train-Synth", alpha=.5, color="orange")
    ax.axvline(tau, 0, max(max(nt), max(ns)), color="black")
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of highest correlation matches')
    ax.text(tau, max(max(nt), max(ns)), f'$tau$ = {tau:0.5f} ', ha='right', va='bottom', rotation=0, color='black')
    plt.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(args.model, "privacy_results.png"))

    closest_test = torch.stack(closest_test)
    is_public = closest_test < tau
    print(f"Number of synthetic images: {len(closest_test)}")
    print(f"Number of non private synthetic images: {sum(is_public)} - memorization rate: {1- is_public.float().mean()}")

    # Save the privacy-compliant latents
    private_latents_output_path = os.path.abspath(args.output)
    os.makedirs(private_latents_output_path, exist_ok=True)

    for path, is_pub in zip(synthetic_images_paths, is_public):
        src = os.path.join(args.synthetic, path)
        tgt = os.path.join(private_latents_output_path, path) 
        if is_pub: 
            shutil.copy(src, tgt)
        else: 
            print(f"Skipping images because it appears memorized: {tgt} ")

    results = {"real_path":[], "tau":[], "synth_path":[], "img_idx": []}
    for clos_idx, tau_val, path in zip(closest_loc, closest_test, synthetic_images_paths):
        results["real_path"].append(ds_train.df.iloc[int(clos_idx)]["FileName"])
        results["tau"].append(float(tau_val))
        results["synth_path"].append(path)
        results["img_idx"].append(int(clos_idx))
    res = pd.DataFrame(results)
    res.to_csv(os.path.join(os.path.dirname(private_latents_output_path), "privacy_scores.csv"), index=False)



