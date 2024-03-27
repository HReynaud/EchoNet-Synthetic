import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
from einops import rearrange

from echosyn.common import load_model, save_as_img


class ImageLoader(Dataset):
    def __init__(self, all_paths):
        self.image_paths = all_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path)

        image = np.array(image)
        image = image / 128.0 - 1 # [-1, 1]
        image = rearrange(image, 'h w c -> c h w')

        return image, self.image_paths[idx]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode and decode images using a trained VAE model.")
    parser.add_argument("-m", "--model", type=str, help="Path to the trained VAE model.")
    parser.add_argument("-i", "--input", type=str, help="Path to the input folder.")
    parser.add_argument("-o", "--output", type=str, help="Path to the output folder.")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="Batch size to use for encoding and decoding.")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = load_model(args.model)
    model.eval()
    model.to(device, torch.float32)

    # Load the videos
    folder_input = os.path.abspath(args.input)
    folder_output = os.path.abspath(args.output)
    all_images = glob(os.path.join(folder_input, "**", "*.jpg"), recursive=True)
    print(f"Found {len(all_images)} images in {folder_input}")

    # prepare output folder
    os.makedirs(folder_output, exist_ok=True)

    # dataset
    dataset = ImageLoader(all_images)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=max(args.batch_size, 32))

    for batch in tqdm(dataloader):
        images, paths = batch
        images = images.to(device, torch.float32)

        # Encode the video
        with torch.no_grad():
            reconstructed_images = model(images).sample
        
        reconstructed_images = rearrange(reconstructed_images, 'b c h w -> b h w c')
        reconstructed_images = (reconstructed_images + 1) * 128.0
        reconstructed_images = reconstructed_images.clamp(0, 255).cpu().to(torch.uint8)

        # Save the reconstructed images
        for i, path in enumerate(paths):
            new_path = path.replace(folder_input, folder_output)
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            save_as_img(reconstructed_images[i], new_path)
        
    print(f"All reconstructed images saved to {folder_output}")