import argparse
import cv2
import os
import pandas as pd
from tqdm import tqdm

"""
python scripts/complete_pediatrics_filelist.py --dataset datasets/EchoNet-Pediatric/A4C
python scripts/complete_pediatrics_filelist.py --dataset datasets/EchoNet-Pediatric/PSAX
"""

def get_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, nframes, width, height

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset directory')
    args = parser.parse_args()

    csv_path = os.path.join(args.dataset, 'FileList.csv')
    assert os.path.exists(csv_path), f"Could not find FileList.csv at {csv_path}"

    metadata = pd.read_csv(csv_path)
    metadata.to_csv(os.path.join(args.dataset, 'FileList_ORIGINAL.csv'), index=False) # backup

    metadata['FileName'] = metadata['FileName'].apply(lambda x: x.split('.')[0]) # remove extension
    metadata['Fold'] = metadata['Split'] # Copy kfold indices to the Fold column
    metadata['Split'] = metadata['Fold'].apply(lambda x: 'TRAIN' if x in range(8) else 'VAL' if x == 8 else 'TEST')

    # Add columns:
    # df.loc[df['FileName'] == fname, ['FileName', 'FrameHeight','FrameWidth','FPS','NumberOfFrames']] = [fname, 112, 112, fps, len(video)]

    for i, row in tqdm(metadata.iterrows(), total=len(metadata)):
        video_name = row['FileName'] + '.avi'
        video_path = os.path.join(args.dataset, 'Videos', video_name)

        fps, nframes, width, height = get_video_metadata(video_path)

        metadata.loc[i, ['FrameHeight','FrameWidth','FPS','NumberOfFrames']] = [height, width, fps, nframes]
    
    metadata.to_csv(csv_path, index=False)
    print("Updated metadata saved to ", csv_path)


