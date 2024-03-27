import argparse
import decord
import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm
from echosyn.common import save_as_img

decord.bridge.set_bridge('torch')

"""
python scripts/create_reference_dataset.py --dataset datasets/EchoNet-Dynamic --output data/reference/dynamic --frames 128
python scripts/create_reference_dataset.py --dataset datasets/EchoNet-Pediatric/A4C --output data/reference/ped_a4c --frames 16
python scripts/create_reference_dataset.py --dataset datasets/EchoNet-Pediatric/PSAX --output data/reference/ped_psax --frames 16
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--frames', type=int, default=128)
    parser.add_argument('--fps', type=float, default=32, help='Target FPS of the video before frame extraction. Should match the diffusion model FPS.')

    args = parser.parse_args()

    csv_path = os.path.join(args.dataset, 'FileList.csv')
    video_path = os.path.join(args.dataset, 'Videos')

    assert os.path.exists(csv_path), f"Could not find FileList.csv at {csv_path}"
    assert os.path.exists(video_path), f"Could not find Videos directory at {video_path}"

    metadata = pd.read_csv(csv_path)
    metadata = metadata.sample(frac=1, random_state=42) # shuffle
    metadata.reset_index(drop=True, inplace=True)

    extracted_videos = 0

    target_count = {16: 3125, 128: 2048}[args.frames]

    threshold_duration = args.frames / args.fps # 128 -> 4 seconds, 16 -> 0.5 seconds

    for row in tqdm(metadata.iterrows(), total=len(metadata)):
        row = row[1]

        video_name = row['FileName'] if row['FileName'].endswith('.avi') else row['FileName'] + '.avi'
        video_path = os.path.join(args.dataset, 'Videos', video_name)

        nframes = row['NumberOfFrames']
        fps = row['FPS']
        duration = nframes / fps

        if duration < threshold_duration:
            # skip videos which are too short (less than 4 seconds / 128 frames)
            continue

        new_frame_count = np.floor(args.fps / fps * nframes).astype(int)
        resample_indices = np.linspace(0, nframes, new_frame_count, endpoint=False).round().astype(int)

        assert len(resample_indices) >= args.frames
        resample_indices = resample_indices[:args.frames]

        reader = decord.VideoReader(video_path, ctx=decord.cpu(), width=112, height=112)
        video = reader.get_batch(resample_indices) # T x H x W x C, uint8 tensor
        video = video.float().mean(axis=-1).clamp(0, 255).to(torch.uint8) # T x H x W
        video = video.unsqueeze(-1).repeat(1, 1, 1, 3) # T x H x W x 3

        folder_name = video_name[:-4] # remove .avi

        if row['Split'] == 'TRAIN':
            # path_all = os.path.join(args.output, "train", folder_name)
            path_all = os.path.join(args.output, folder_name)
            os.makedirs(path_all, exist_ok=True)
            save_as_img(video, path_all)
            extracted_videos += 1

            if extracted_videos >= target_count:
                print("Reached target count, stopping.")
                break

    print(f"Saved {extracted_videos} videos to {args.output}.")
    if extracted_videos < target_count:
        print(f"WARNING: only saved {extracted_videos} videos, which is less than the target count of {target_count}.")



