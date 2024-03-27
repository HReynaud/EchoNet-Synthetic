# Latent Image Diffusion Model

The Latent Image Diffusion Model (LIDM) is the first step of our generative pipeline. It generates a latent representation of a heart, which is then passed to the Latent Video Diffusion Model (LVDM) to generate a video of the heart beating.

Training the LIDM is straightforward and does not require a lot of resouces. In the paper, the LIDMs are trained for ~24h on a single A100 GPU. The batch size can be adjusted to fit smaller GPUs with no noticeable loss of quality.

## 1. Activate the environment

First, activate the echosyn environment.

```bash
conda activate echosyn
```

## 2. Data preparation
Follow the instruction in the [Data preparation](../../README.md#data-preparation) to prepare the data for training. Here, you need the VAE-encoded videos.

## 3. Train the LIDM
Once the environment is set up and the data is ready, you can train the LIDMs with the following commands:

```bash
python echosyn/lidm/train.py --config echosyn/lidm/configs/dynamic.yaml
python echosyn/lidm/train.py --config echosyn/lidm/configs/ped_a4c.yaml
python echosyn/lidm/train.py --config echosyn/lidm/configs/ped_psax.yaml
```

## 4. Sample from the LIDM

Once the LIDMs are trained, you can sample from them with the following command:

```bash
# For the Dynamic dataset
python echosyn/lidm/sample.py \
    --config echosyn/lidm/configs/dynamic.yaml \
    --unet experiments/lidm_dynamic/checkpoint-500000/unet_ema \
    --vae models/vae \
    --output samples/lidm_dynamic \
    --num_samples 50000 \
    --batch_size 128 \
    --num_steps 64 \
    --save_latent \
    --seed 0
```

```bash
# For the Pediatric A4C dataset
python echosyn/lidm/sample.py \
    --config echosyn/lidm/configs/ped_a4c.yaml \
    --unet experiments/lidm_ped_a4c/checkpoint-500000/unet_ema \
    --vae models/vae \
    --output samples/lidm_ped_a4c \
    --num_samples 50000 \
    --batch_size 128 \
    --num_steps 64 \
    --save_latent \
    --seed 0
```

```bash
# For the Pediatric PSAX dataset
python echosyn/lidm/sample.py \
    --config echosyn/lidm/configs/ped_psax.yaml \
    --unet experiments/lidm_ped_psax/checkpoint-500000/unet_ema \
    --vae models/vae \
    --output samples/lidm_ped_psax \
    --num_samples 50000 \
    --batch_size 128 \
    --num_steps 64 \
    --save_latent \
    --seed 0
```

## 5. Evaluate the LIDM

To evaluate the LIDMs, we use the FID and IS scores. 
To do so, we need to generate 50,000 samples, using the command above. 
Note that the privacy step will reject some of these samples. 
It is therefore better to generate 100,000, so we can calculate the FID and IS scores again, on the privacy-compliant samples.
The samples are compared to the real samples, which are generated in the [Data preparation](../../README.md#data-preparation) step.

Then, to evaluate the samples, run the following commands:

```bash
cd external/stylegan-v

# For the Dynamic dataset
python src/scripts/calc_metrics_for_dataset.py \
    --real_data_path ../../data/reference/dynamic \
    --fake_data_path ../../samples/lidm_dynamic/images \
    --mirror 0 --gpus 1 --resolution 112 \
    --metrics fid50k_full,is50k >> "../../samples/lidm_dynamic/metrics.txt"

# For the Pediatric A4C dataset
python src/scripts/calc_metrics_for_dataset.py \
    --real_data_path data/reference/ped_a4c \
    --fake_data_path samples/lidm_ped_a4c/images \
    --mirror 0 --gpus 1 --resolution 112 \
    --metrics fid50k_full,is50k >> "samples/lidm_ped_a4c/metrics.txt"

# For the Pediatric PSAX dataset
python src/scripts/calc_metrics_for_dataset.py \
    --real_data_path data/reference/ped_psax \
    --fake_data_path samples/lidm_ped_psax/images \
    --mirror 0 --gpus 1 --resolution 112 \
    --metrics fid50k_full,is50k >> "samples/lidm_ped_psax/metrics.txt
```

## 6. Save the LIDMs for later use
Once you are satisfied with the performance of the LIDMs, you can save them for later use with the following commands:

```bash
mkdir -p models/lidm_dynamic; cp -r experiments/lidm_dynamic/checkpoint-500000/unet_ema/* models/lidm_dynamic/; cp experiments/lidm_dynamic/config.yaml models/lidm_dynamic/
mkdir -p models/lidm_ped_a4c; cp -r experiments/lidm_ped_a4c/checkpoint-500000/unet_ema/* models/lidm_ped_a4c/; cp experiments/lidm_ped_a4c/config.yaml models/lidm_ped_a4c/
mkdir -p models/lidm_ped_psax; cp -r experiments/lidm_ped_psax/checkpoint-500000/unet_ema/* models/lidm_ped_psax/; cp experiments/lidm_ped_psax/config.yaml models/lidm_ped_psax/
```

This will save the selected ema version of the model, ready to be loaded in any other script as a standalone model.