# Variational Autoencoder (fun fact: it's also a GAN) 

Training the VAE is not a necessity as an open source VAE such as [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) should work well enough for this project.

However, if you would like to train the VAE yourself, follow these steps.

## 1. Clone the necessary repositories
We use the Stable-Diffusion repo to train the VAE model, but the Taming-Transformers repo is a necessary dependency. It is advised to follow these steps exactly to avoid errors later on.

```bash
cd external
git clone https://github.com/CompVis/stable-diffusion

cd stable-diffusion
conda env create -f environment.yaml
conda activate ldm
```

This will download the Stable-Diffusion repository and create the necessary environment to run it.

```bash
git clone https://github.com/CompVis/taming-transformers
cd taming-transformers
pip install -e .
```

This will download the Taming-Transformers repository and install it as a package.

## 2. Prepare the data
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

That's it for the dataset.


## 3. Train the VAE
Now that the data is ready, we can train the VAE. We use the following command to train the VAE on the images we just extracted.

```bash
cd external/stable-diffusion
export DATADIR=$(cd ../../data/vae_train_images && pwd)
python main.py \
    --base ../../echosyn/vae/usencoder_kl_16x16x4.yaml \
    -t True \
    --gpus 0,1,2,3,4,5,6,7 \
    --logdir experiments/vae \
```

This will train the VAE on the images we extracted and save the model in the experiments/vae folder.<br>
For the paper, we train the VAE on 8xA100 GPUs for 5 days.<br>
*Note: if you use a single gpu, you need to leave a comma in the --gpus argument, like so: ```--gpus 0,```*<br>
To resume training from a checkpoint, use the ```--resume``` flag and replace ```EXPERIMENT_NAME``` with the correct experiment name.

```bash
python main.py \
    --base ../../echosyn/vae/usencoder_kl_16x16x4.yaml \
    -t True \
    --gpus 0,1,2,3,4,5,6,7 \
    --logdir experiments/vae \
    --resume experiments/vae/EXPERIMENT_NAME
```

## 4. Export the VAE to Diffusers 🧨
Now that the VAE is trained, we can export it to a Diffuser AutoencoderKL model. This is done with the following command, replacing ```EXPERIMENT_NAME``` with the correct experiment name.

```bash
python scripts/convert_vae_pt_to_diffusers.py
    --vae_pt_path experiments/EXPERIMENT_NAME/checkpoints/last.ckpt
    --dump_path models/vae/
```
The script is taken as-is from the [Diffusers library](https://github.com/huggingface/diffusers/blob/main/scripts/convert_vae_pt_to_diffusers.py).

## 5. Test the model in Diffusers 🧨
That's it! You now have a VAE model trained on your own data and exported to a Diffuser model. 

To test the model, use the echosyn environment and do:
```python
from PIL import Image
import torch
import numpy as np
from diffusers import AutoencoderKL
model = AutoencoderKL.from_pretrained("models/vae")
model.eval()

# Use the model to encode and decode images
img = Image.open("data/vae_train_images/images/0X10A5FC19152B50A5_00001.jpg")

img = torch.from_numpy(np.array(img)).permute(2,0,1).unsqueeze(0).to(torch.float32) / 128 - 1

with torch.no_grad():
    lat = model.encode(img).latent_dist.sample()
    rec = model.decode(lat).sample
# Display the original and reconstructed images
img = img.squeeze(0).permute(1,2,0)
rec = rec.squeeze(0).permute(1,2,0)
display(Image.fromarray(((img + 1) * 128).to(torch.uint8).numpy()))
```

Note that model might not work in mixed precision mode, which would cause nan values. If this happens, force the model dtype to torch.float32

## 6. Evaluate the VAE

To evaluate the VAE, we reconstruct the extracted image datasets and compare the original images to the reconstructed images. We use the following command to recontruct (encode-decode) the images from the dynamic dataset.

```bash
python scripts/vae_reconstruct_image_folder.py \
    -model models/vae \
    -input data/reference/dynamic \
    -output samples/reconstructed/dynamic \
    -batch_size 32
```

This will save the reconstructed images in the data/reconstructed/dynamic folder. To evaluate the VAE, we use the following command to compute the FID score between the original images and the reconstructed images, with the help of the StyleGAN-V repository.

```bash
cd external/stylegan-v

python src/scripts/calc_metrics_for_dataset.py \
    --real_data_path ../../data/reference/dynamic \
    --fake_data_path ../../samples/reconstructed/dynamic \
    --mirror 0 --gpus 1 --resolution 112 \
    --metrics fid50k_full,is50k >> "../../samples/reconstructed/dynamic.txt"
```


