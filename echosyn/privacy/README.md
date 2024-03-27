# Re-Identification Model

For this work, we use the [Latent Image Diffusion Model (LIDM)](../lidm/README.md) to generate synthetic echocardiography images. To enforce privacy, as a post-hoc step, we train a re-identification model to project real and generated images into a common latent space. This allows us to compute the similarity between two images, and by extension, to detect any synthetic images that are too similar to real ones.

The re-identification models are trained on the VAE-encoded real images. We train one re-identification model per LIDM. The training takes a few hours on a single A100 GPU.

## 1. Activate the environment

First, activate the echosyn environment.

```bash
conda activate echosyn
```

## 2. Data preparation
Follow the instruction in the [Data preparation](../../README.md#data-preparation) to prepare the data for training. Here, you need the VAE-encoded videos.

## 3. Train the Re-Identification models
Once the environment is set up and the data is ready, you can train the Re-Identification models with the following commands:

```bash
python echosyn/privacy/train.py --config=echosyn/privacy/configs/config_dynamic.json
python echosyn/privacy/train.py --config=echosyn/privacy/configs/config_ped_a4c.json
python echosyn/privacy/train.py --config=echosyn/privacy/configs/config_ped_psax.json
```

## 4. Filter the synthetic images
After training the re-identification models, you can filter the synthetic images, generated with the LIDMs with the following commands:

```bash
python echosyn/privacy/apply.py \
    --model experiments/reidentification_dynamic \
    --synthetic samples/lidm_dynamic/latents \
    --reference data/latents/dynamic \
    --output samples/lidm_dynamic/privatised_latents
```

```bash
python echosyn/privacy/apply.py \
    --model experiments/reidentification_ped_a4c \
    --synthetic samples/lidm_ped_a4d/latents \
    --reference data/latents/ped_a4c \
    --output samples/lidm_ped_a4d/privatised_latents
```

```bash
python echosyn/privacy/apply.py \
    --model experiments/reidentification_ped_psax \
    --synthetic samples/lidm_ped_psax/latents \
    --reference data/latents/ped_psax \
    --output samples/lidm_ped_psax/privatised_latents
```

This script will filter out all latents that are too similar to real latents (encoded images). 
The filtered latents are saved in the directory specified in `output`. 
The similarity threshold is automatically determined by the `apply.py` script.
The latents are all that's required to generate the privacy-compliant synthetic videos, because the Latent Video Diffusion Model (LVDM) is conditioned on the latents, not the images themselves.
To obtain the privacy-compliant images, you can use the provided script like so:

```bash
./scripts/copy_privacy_compliant_images.sh samples/dynamic/images samples/dynamic/privacy_compliant_latents samples/dynamic/privacy_compliant_images
./scripts/copy_privacy_compliant_images.sh samples/ped_a4c/images samples/ped_a4c/privacy_compliant_latents samples/ped_a4c/privacy_compliant_images
./scripts/copy_privacy_compliant_images.sh samples/ped_psax/images samples/ped_psax/privacy_compliant_latents samples/ped_psax/privacy_compliant_images
```

## 5. Evaluate the remaining images

To evaluate the remaining images, we use the same process as for the LIDM, with the following commands:

```bash

cd external/stylegan-v

# For the Dynamic dataset
python src/scripts/calc_metrics_for_dataset.py \
    --real_data_path ../../data/reference/dynamic \
    --fake_data_path ../../samples/lidm_dynamic/privacy_compliant_images \
    --mirror 0 --gpus 1 --resolution 112 \
    --metrics fid50k_full,is50k >> "../../samples/lidm_dynamic/privacy_compliant_metrics.txt"

# For the Pediatric A4C dataset
python src/scripts/calc_metrics_for_dataset.py \
    --real_data_path data/reference/ped_a4c \
    --fake_data_path samples/lidm_ped_a4c/privacy_compliant_images \
    --mirror 0 --gpus 1 --resolution 112 \
    --metrics fid50k_full,is50k >> "samples/lidm_ped_a4c/privacy_compliant_metrics.txt"

# For the Pediatric PSAX dataset
python src/scripts/calc_metrics_for_dataset.py \
    --real_data_path data/reference/ped_psax \
    --fake_data_path samples/lidm_ped_psax/privacy_compliant_images \
    --mirror 0 --gpus 1 --resolution 112 \
    --metrics fid50k_full,is50k >> "samples/lidm_ped_psax/privacy_compliant_metrics.txt
```

It is important that at least 50,000 samples remain in the filtered folders, to ensure that the FID and IS scores are reliable.


## 6. Save the Re-Identification models for later use

The re-identification models can be saved for later use with the following command:

```bash
mkdir -p models/reidentification_dynamic; cp experiments/reidentification_dynamic/reidentification_dynamic_best_network.pth models/reidentification_dynamic/; cp experiments/reidentification_dynamic/config.json models/reidentification_dynamic/
```
```bash
mkdir -p models/reidentification_ped_a4c; cp experiments/reidentification_ped_a4c/reidentification_ped_a4c_best_network.pth models/reidentification_ped_a4c/; cp experiments/reidentification_ped_a4c/config.json models/reidentification_ped_a4c/
```
```bash
mkdir -p models/reidentification_ped_psax; cp experiments/reidentification_ped_psax/reidentification_ped_psax_best_network.pth models/reidentification_ped_psax/; cp experiments/reidentification_ped_psax/config.json models/reidentification_ped_psax/
```
