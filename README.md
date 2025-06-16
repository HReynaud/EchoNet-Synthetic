# EchoNet-Synthetic
<p align="center">
<a href="https://huggingface.co/spaces/HReynaud/EchoNet-Synthetic" alt="Hugging Face Demo"><img src="https://img.shields.io/static/v1?label=HuggingFace&message=Demo&color=yellow&logo=data:image/png;base64,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" /></a>
<a href="https://huggingface.co/HReynaud/EchoNet-Synthetic/tree/main" alt="Weights"><img src="https://img.shields.io/static/v1?label=Download&message=Weights&color=red&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAQAAAC1QeVaAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAkGVYSWZNTQAqAAAACAAGAQYAAwAAAAEAAgAAARIAAwAAAAEAAQAAARoABQAAAAEAAABWARsABQAAAAEAAABeASgAAwAAAAEAAgAAh2kABAAAAAEAAABmAAAAAAAAAEgAAAABAAAASAAAAAEAA6ABAAMAAAABAAEAAKACAAQAAAABAAAADqADAAQAAAABAAAADgAAAAAGqundAAAACXBIWXMAAAsTAAALEwEAmpwYAAADRmlUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iWE1QIENvcmUgNi4wLjAiPgogICA8cmRmOlJERiB4bWxuczpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiPgogICAgICA8cmRmOkRlc2NyaXB0aW9uIHJkZjphYm91dD0iIgogICAgICAgICAgICB4bWxuczp0aWZmPSJodHRwOi8vbnMuYWRvYmUuY29tL3RpZmYvMS4wLyIKICAgICAgICAgICAgeG1sbnM6ZXhpZj0iaHR0cDovL25zLmFkb2JlLmNvbS9leGlmLzEuMC8iPgogICAgICAgICA8dGlmZjpDb21wcmVzc2lvbj4xPC90aWZmOkNvbXByZXNzaW9uPgogICAgICAgICA8dGlmZjpSZXNvbHV0aW9uVW5pdD4yPC90aWZmOlJlc29sdXRpb25Vbml0PgogICAgICAgICA8dGlmZjpYUmVzb2x1dGlvbj43MjwvdGlmZjpYUmVzb2x1dGlvbj4KICAgICAgICAgPHRpZmY6WVJlc29sdXRpb24+NzI8L3RpZmY6WVJlc29sdXRpb24+CiAgICAgICAgIDx0aWZmOlBob3RvbWV0cmljSW50ZXJwcmV0YXRpb24+MjwvdGlmZjpQaG90b21ldHJpY0ludGVycHJldGF0aW9uPgogICAgICAgICA8dGlmZjpPcmllbnRhdGlvbj4xPC90aWZmOk9yaWVudGF0aW9uPgogICAgICAgICA8ZXhpZjpQaXhlbFhEaW1lbnNpb24+OTAwPC9leGlmOlBpeGVsWERpbWVuc2lvbj4KICAgICAgICAgPGV4aWY6Q29sb3JTcGFjZT4xPC9leGlmOkNvbG9yU3BhY2U+CiAgICAgICAgIDxleGlmOlBpeGVsWURpbWVuc2lvbj45MDA8L2V4aWY6UGl4ZWxZRGltZW5zaW9uPgogICAgICA8L3JkZjpEZXNjcmlwdGlvbj4KICAgPC9yZGY6UkRGPgo8L3g6eG1wbWV0YT4KeyhqswAAAPhJREFUGBkFwcFrzQEAwPHv+71nzEmxrbe9eqjJoqUoJEWMg8mKFg7mIKU4cHbwX4yTP8Bf4eTgqnbwL+zg4Kg+Pp/KUAZDeeRpGVdVZVSGKpfA7XK0qiqLZd1lz/303UOrZVKVL/54Z9WS80aOlC1r5ZghB9g388ZVvxxYKTcsV+WM97a9NLUDdmxat+WJs1V5YNdjF5yzURad8BG/q3LXng0XvfXBxNSa1/iWV3544b47tsGeFXPXnTLKIfZN3HTaJ5+r3LJQlV1fXbNp7oqZk+WehTKuqspxM7v455lxGaoyNjFUWXLor3mZVFVVVcZl2bSMq6r+A7pwtCIxZY9XAAAAAElFTkSuQmCC" /></a>
<a href="https://link.springer.com/chapter/10.1007/978-3-031-72104-5_28" alt="MICCAI Proceedings"><img src="https://img.shields.io/static/v1?label=MICCAI&message=2024&color=blue&logo=data:image/png;base64,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" /></a>
<a href="https://arxiv.org/abs/2406.00808" alt="arXiv"><img src="http://img.shields.io/badge/arXiv-2406.00808-B31B1B.svg" /></a>
<a href="https://github.com/HReynaud/EchoNet-Synthetic"><img src="https://img.shields.io/github/stars/HReynaud/EchoNet-Synthetic?style=social" alt="Star this repo!" /></a>
</p>

## Deep-Dive Podcast (AI generated)


https://github.com/user-attachments/assets/06d0811b-f2fc-442f-bde4-0885d8dfd319

*Note: Turn on audio !*

## Introduction

This repository contains the code and model weights for the paper *[EchoNet-Synthetic: Privacy-preserving Video Generation for Safe Medical Data Sharing](https://arxiv.org/abs/2406.00808)*. Hadrien Reynaud, Qingjie Meng, Mischa Dombrowski, Arijit Ghosh, Alberto Gomez, Paul Leeson and Bernhard Kainz. MICCAI 2024.

EchoNet-Synthetic presents a protocol to generate surrogate privacy-compliant datasets that are as valuable as their original counterparts to train downstream models (e.g. regression models).

In this repository, we present the code we use for the experiments in the paper. We provide the code to train the models, generate the synthetic data, and evaluate the quality of the synthetic data.
We also provide all the pre-trained models and release the synthetic datasets we generated.

📜 Read the Paper [on arXiv](https://arxiv.org/abs/2406.00808) <br>
📕 [MICCAI 2024 Proceedings](https://link.springer.com/chapter/10.1007/978-3-031-72104-5_28) <br>
🤗 Try our interactive demo [on HuggingFace](https://huggingface.co/spaces/HReynaud/EchoNet-Synthetic), it contains all the generative pipeline inference code and weights !

![Slim GIF Demo](ressources/mosaic_slim.gif)

*Exemple of synthetic videos generated with EchoNet-Synthetic. First Video is real, others are generated.*

## Table of contents
1. [Environment setup](#environment-setup)
2. [Data preparation](#data-preparation)
3. [The models](#the-models)
4. [Generating EchoNet-Synthetic](#generating-echonet-synthetic)
5. [Evaluation](#evaluation)
6. [Results](#results)
7. [Citation](#citation)

## Environment setup
<!-- <details open id="environment-setup">
<summary style="font-size: 1.5em; font-weight: bold;" >Environment setup<hr></summary> -->

First, we need to set up the environment. We use the following command to create a new conda environment with the required dependencies.

```bash
conda create -y -n echosyn python=3.11
conda activate echosyn
pip install -e .
```
*Note: the exact version of each package can be found in requirements.txt if necessary*

This repository lets you train three models: 
- the Latent Image Diffusion Model (LIDM)
- the Re-Indentification models for privacy checks
- the Latent Video Diffusion Model (LVDM)

We rely on external libraries to:
- train the Variational Auto-Encoder (VAE) (Stable-Diffusion and Taming-Transformers)
- evaluate the generated images and videos (StyleGAN-V)
- evaluate the synthetic data on the Ejection Fraction downstream task (EchoNet-Dynamic)

How to install the external libraries is explained in the [External libraries](external/README.md) section.


<!-- </details> -->

## Data preparation
<!-- <details open id="data-preparation">
<summary style="font-size: 1.5em; font-weight: bold;">Data preparation<hr></summary> -->

### ➡ Original datasets
Download the EchoNet-Dynamic dataset from [here](https://echonet.github.io/dynamic/) and the EchoNet-Pediatric dataset from [here](https://echonet.github.io/pediatric/). The datasets are available for free upon request. Once downloaded, extract the content of the archive in the `datasets` folder. For simplicity and consistency, we structure them like so:
```
datasets
├── EchoNet-Dynamic
│   ├── Videos
│   ├── FileList.csv
│   └── VolumeTracings.csv
└── EchoNet-Pediatric
    ├── A4C
    │   ├── Videos
    │   ├── FileList.csv
    │   └── VolumeTracings.csv
    └── PSAX
        ├── Videos
        ├── FileList.csv
        └── VolumeTracings.csv
```

To harmonize the datasets, we add some information to the `FileList.csv` files of the EchoNet-Pediatric dataset, namely FrameHeight, FrameWidth, FPS, NumberOfFrames. We also arbitrarily set the splits from the 10-fold indices to a simple TRAIN/VAL/TEST split. These updates ares applied with the following command:

```bash
python scripts/complete_pediatrics_filelist.py --dataset datasets/EchoNet-Pediatric/A4C
python scripts/complete_pediatrics_filelist.py --dataset datasets/EchoNet-Pediatric/PSAX
```

This is crucial for the other scripts to work properly.

### ➡ Image datasets for VAE training
See the [VAE training](echosyn/vae/README.md) section To see how to train the VAE.

The VAE needs images only and to keep things simple, we format our video datasets into image datasets.
We do that with:
```bash
bash scripts/extract_frames_from_videos.sh datasets/EchoNet-Dynamic/Videos data/vae_train_images/images/
bash scripts/extract_frames_from_videos.sh datasets/EchoNet-Pediatric/A4C/Videos data/vae_train_images/images/
bash scripts/extract_frames_from_videos.sh datasets/EchoNet-Pediatric/PSAX/Videos data/vae_train_images/images/
```

Note that this will merge all the images in the same folder.

Then, we need to create train.txt file and a val.txt file containing the path to these images.
```bash
find $(cd data/vae_train_images/images && pwd) -type f | shuf > tmp.txt
head -n -1000 tmp.txt > data/vae_train_images/train.txt
tail -n 1000 tmp.txt > data/vae_train_images/val.txt
rm tmp.txt
```

### ➡ Latent Video datasets for LIDM / Privacy / LVDM training

The LIDM, Re-Identification model and LVDM are trained on pre-encoded latent representations of the videos. To encode the videos, we use the image VAE. You can either retrain the VAE or download it from [here](https://huggingface.co/HReynaud/EchoNet-Synthetic/tree/main/vae). Once you have the VAE, you can encode the videos with the following command:

```bash
# For the EchoNet-Dynamic dataset
python scripts/encode_video_dataset.py \
    --model models/vae \
    --input datasets/EchoNet-Dynamic \
    --output data/latents/dynamic \
    --gray_scale
```
```bash
# For the EchoNet-Pediatric datasets
python scripts/encode_video_dataset.py \
    --model models/vae \
    --input datasets/EchoNet-Pediatric/A4C \
    --output data/latents/ped_a4c \
    --gray_scale

python scripts/encode_video_dataset.py \
    --model models/vae \
    --input datasets/EchoNet-Pediatric/PSAX \
    --output data/latents/ped_psax \
    --gray_scale
```

### ➡ Validation datasets

To quantitatively evaluate the quality of the generated images and videos, we use the StyleGAN-V repo.
We cover the evaluation process in the [Evaluation](#evaluation) section.
To enable this evaluation, we need to prepare the validation datasets. We do that with the following command:

```bash
python scripts/create_reference_dataset.py --dataset datasets/EchoNet-Dynamic --output data/reference/dynamic --frames 128
```

```bash
python scripts/create_reference_dataset.py --dataset datasets/EchoNet-Pediatric/A4C --output data/reference/ped_a4c --frames 16
```

```bash
python scripts/create_reference_dataset.py --dataset datasets/EchoNet-Pediatric/PSAX --output data/reference/ped_psax --frames 16
```

Note that the Pediatric datasets do not support 128 frames, preventing the computation of FVD_128, because there are not enough videos lasting more 4 seconds or more. We therefore only extract 16 frames per video for these datasets.

</details>

## The Models
<!-- <details open id="models">
<summary style="font-size: 1.5em; font-weight: bold;">The models<hr></summary> -->

![Models](ressources/models.jpg)

*Our pipeline, using all our models: LIDM, Re-Identification (Privacy), LVDM and VAE*


### The VAE

You can download the pretrained VAE from [here](https://huggingface.co/HReynaud/EchoNet-Synthetic/tree/main/vae) or train it yourself by following the instructions in the [VAE training](echosyn/vae/README.md) section.

### The LIDM

You can download the pretrained LIDMs from [here](https://huggingface.co/HReynaud/EchoNet-Synthetic/tree/main/lidm_dynamic) or train them yourself by following the instructions in the [LIDM training](echosyn/lidm/README.md) section.

### The Re-Identification models

You can download the pretrained Re-Identification models from [here](https://huggingface.co/HReynaud/EchoNet-Synthetic/tree/main/reidentification_dynamic) or train them yourself by following the instructions in the [Re-Identification training](echosyn/privacy/README.md) section.

### The LVDM

You can download the pretrained LVDM from [here](https://huggingface.co/HReynaud/EchoNet-Synthetic/tree/main/lvdm) or train it yourself by following the instructions in the [LVDM training](echosyn/lvdm/README.md) section.

### Structure

The models should be structured as follows:
```
models
├── lidm_dynamic
├── lidm_ped_a4c
├── lidm_ped_psax
├── lvdm
├── regression_dynamic
├── regression_ped_a4c
├── regression_ped_psax
├── reidentification_dynamic
├── reidentification_ped_a4c
├── reidentification_ped_psax
└── vae
```

<!-- </details> -->

## Generating EchoNet-Synthetic
<!-- <details open id="echonet-synthetic">
<summary style="font-size: 1.5em; font-weight: bold;">Generating EchoNet-Synthetic<hr></summary> -->

Now that we have all the necessary models, we can generate the synthetic datasets. The process is the same for all three datasets and involves the following steps:
- Generate a collection of latent heart images with the LIDMs (usually 2x the amount of videos we are targetting)
- Apply the privacy check, which will filter out some of the latent images
- Generate the videos with the LVDM, and decode them with the VAE

#### Dynamic dataset
For the dynamic dataset, we aim for 10,030 videos, so we start by generating 20,000 latent images:
```bash
python echosyn/lidm/sample.py \
    --config models/lidm_dynamic/config.yaml \
    --unet models/lidm_dynamic \
    --vae models/vae \
    --output samples/synthetic/lidm_dynamic \
    --num_samples 20000 \
    --batch_size 128 \
    --num_steps 64 \
    --save_latent \
    --seed 0
```

Then, we filter the latent images with the re-identification model:
```bash
python echosyn/privacy/apply.py \
    --model models/reidentification_dynamic \
    --synthetic samples/synthetic/lidm_dynamic/latents \
    --reference data/latents/dynamic \
    --output samples/synthetic/lidm_dynamic/privatised_latents
```

We generate the synthetic videos with the LVDM:
```bash
python echosyn/lvdm/sample.py \
    --config models/lvdm/config.yaml \
    --unet models/lvdm \
    --vae models/vae \
    --conditioning samples/synthetic/lidm_dynamic/privatised_latents \
    --output samples/synthetic/lvdm_dynamic \
    --num_samples 10030 \
    --batch_size 8 \
    --num_steps 64 \
    --min_lvef 10 \
    --max_lvef 90 \
    --save_as avi \
    --frames 192
```

Finally, update the content of the FileList.csv to respect the right amount of videos in each split, and the correct naming convention.
```bash
mkdir -p datasets/EchoNet-Synthetic/dynamic
mv samples/synthetic/lvdm_dynamic/avi datasets/EchoNet-Synthetic/dynamic/Videos
cp samples/synthetic/lvdm_dynamic/FileList.csv datasets/EchoNet-Synthetic/dynamic/FileList.csv
python scripts/update_split_filelist.py --csv datasets/EchoNet-Synthetic/dynamic/FileList.csv --train 7465 --val 1288 --test 1277
```

You can use the exact same process for the Pediatric datasets, just make sure to use the right models.
The A4C dataset should have 3284 videos, with 2580 train, 336 validation and 368 test videos.
3559 448 519
The PSAX dataset should have 4526 videos, with 3559 train, 448 validation and 519 test videos.
They also have a different number of frames per video, which is 128 insted of 192.

<!-- </details> -->

## Evaluation
<!-- <details open id="evaluation">
<summary style="font-size: 1.5em; font-weight: bold;">Evaluation<hr></summary> -->

As the final step, we evaluate the quality of EchoNet-Synthetic videos by training a Ejection Fraction regression model on the synthetic data and evaluating it on the real data.
To do so, we use the EchoNet-Dynamic repository. 
You will need to clone a slightly modifier version of the repository from [here](https://github.com/HReynaud/echonet), and put it in the `external` folder.

Once that is done, you will need to switch environment to the echonet one.
To create the echonet environment, you can use the following command:
```bash
cd external/echonet
conda create -y -n echonet python=3.10
conda activate echonet
pip install -e .
```

Then, you should double check that everything is working by reproducing the results of the EchoNet-Dynamic repository on the real EchoNet-Dynamic dataset. This should take ~12 hours on a single GPU.
```bash
echonet video --data_dir ../../datasets/EchoNet-Dynamic \
        --output ../../experiments/regression_echonet_dynamic \
        --pretrained \
        --run_test \
        --num_epochs 45
```

Once you have confirmed that everything is working, you can train the same model on the synthetic data, while evaluating on the real data.
```bash
echonet video --data_dir ../../datasets/EchoNet-Synthetic/dynamic \
        --data_dir_real ../../datasets/EchoNet-Dynamic \
        --output ../../experiments/regression_echosyn_dynamic \
        --pretrained \
        --run_test \
        --num_epochs 45 \
        --period 1
```

This will automatically compute the final metrics after training, and store them in the `experiments/regression_echosyn_dynamic/log.csv` folder.

The process is the same for the Pediatric datasets, although they start from a pretrained model, trained on the real EchoNet-Dynamic data, which would look like this:
```bash
echonet video --data_dir ../../datasets/EchoNet-Pediatric/A4C \
        --data_dir_real ../../datasets/EchoNet-Pediatric/A4C \
        --output ../../experiments/regression_echosyn_peda4c \
        --weights ../../experiments/regression_real/best.pt \
        --run_test \
        --num_epochs 45 \
        --period 1
```

<!-- Improving EchoNet-Synthetic -->
Here are two tricks that can improve the quality and alignment of the synthetic data with the real data:
- Re-label the synthetic data with the regression model trained on the real data.
- Use the Ejection Fraction scores from the real dataset to create the synthetic dataset, not uniformaly distributed ones.

</details>

## Results
<!-- <details open id="results">
<summary style="font-size: 1.5em; font-weight: bold;">Results<hr></summary> -->

<p>Here is a side by side comparison between a real video and a synthetic video generated with EchoNet-Synthetic. We use a frame from the real video and its corresponding ejected fraction score to "reproduce" the real video.</p>

<table style="width:auto; border:1px solid black; text-align:center;">
    <tr>
        <th style="padding:10px; border:1px solid black;">Real Video</th>
        <th style="padding:10px; border:1px solid black;">Reproduction</th>
    </tr>
    <tr>
        <td style="padding:10px; border:1px solid black;"><img src="ressources/real.gif" alt="Real Video"></td>
        <td style="padding:10px; border:1px solid black;"><img src="ressources/synt.gif" alt="Reproduction"></td>
    </tr>
</table>

<p>Here we show a collection of synthetic videos generated with EchoNet-Synthetic. We can see that the quality of the videos is excellent, and show a variety of ejection fraction scores.</p>

![Mosaic](ressources/mosaic.gif)

<!-- </details> -->

# Acknowledgements
This work was supported by Ultromics Ltd. and the UKRI Centre for Doctoral Training in Artificial Intelligence for Healthcare (EP/S023283/1).
HPC resources were provided by the Erlangen National High Performance Computing Center (NHR@FAU) of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) under the NHR projects b143dc and b180dc. NHR funding is provided by federal and Bavarian state authorities. NHR@FAU hardware is partially funded by the German Research Foundation (DFG) – 440719683.

<!-- ## Citation -->
## Citation

```
@inproceedings{reynaud2024echonet,
  title={Echonet-synthetic: Privacy-preserving video generation for safe medical data sharing},
  author={Reynaud, Hadrien and Meng, Qingjie and Dombrowski, Mischa and Ghosh, Arijit and Day, Thomas and Gomez, Alberto and Leeson, Paul and Kainz, Bernhard},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={285--295},
  year={2024},
  organization={Springer}
}
```
