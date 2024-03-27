import argparse
import logging
import math
import os
import shutil
from einops import rearrange
from omegaconf import OmegaConf
import numpy as np
from tqdm.auto import tqdm
from packaging import version
from functools import partial
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet3DConditionModel, UNetSpatioTemporalConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available

from echosyn.common import padf, unpadf, pad_reshape, unpad_reshape, instantiate_from_config
from echosyn.common.datasets import instantiate_dataset

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def log_validation(
        config,
        unet,
        vae,
        scheduler,
        accelerator,
        weight_dtype,
        epoch,
        val_dataset
    ):
    logger.info("Running validation... ")

    val_unet = accelerator.unwrap_model(unet)
    val_vae = vae.to(accelerator.device, dtype=torch.float32)
    scheduler.set_timesteps(config.validation_timesteps)
    timesteps = scheduler.timesteps

    if config.enable_xformers_memory_efficient_attention:
        val_unet.enable_xformers_memory_efficient_attention()

    if config.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(config.seed)

    indices = np.random.choice(len(val_dataset), size=len(config.validation_lvefs), replace=False)
    ref_elements = [val_dataset[i] for i in indices]
    ref_frames = [e['image'] for e in ref_elements]
    ref_videos = [e['video'] for e in ref_elements]
    ref_frames = torch.stack(ref_frames, dim=0) # B x C x H x W
    ref_frames = ref_frames.to(accelerator.device, weight_dtype)
    ref_frames = ref_frames[:,:,None,:,:].repeat(1,1,config.validation_frames,1,1) # B x C x T x H x W

    if config.unet._class_name == "UNetSpatioTemporalConditionModel":
        dummy_added_time_ids = torch.zeros((len(config.validation_lvefs), config.unet.addition_time_embed_dim), device=accelerator.device, dtype=weight_dtype)
        unet = partial(unet, added_time_ids=dummy_added_time_ids)
    
    format_input = pad_reshape if config.unet._class_name == "UNetSpatioTemporalConditionModel" else padf
    format_output = unpad_reshape if config.unet._class_name == "UNetSpatioTemporalConditionModel" else unpadf

    logger.info("Sampling... ")
    with torch.no_grad(), torch.autocast("cuda"):
        # prepare model inputs
        B, C, T, H, W = len(config.validation_lvefs), 4, config.validation_frames, config.unet.sample_size, config.unet.sample_size
        latents = torch.randn((B, C, T, H, W), device=accelerator.device, dtype=weight_dtype, generator=generator)
        lvefs = torch.tensor(config.validation_lvefs, device=accelerator.device, dtype=weight_dtype)
        lvefs = lvefs[:, None, None] # B -> B x 1 x 1

        if config.validation_guidance > 1.0:
            lvefs = torch.cat([lvefs] * 2)
            lvefs_mask = [[True]*B + [False]*B]
            lvefs_mask = torch.tensor(lvefs_mask, device=accelerator.device, dtype=torch.bool)[:,None,None]
            ref_frames = torch.cat([ref_frames] * 2)
        else:
            lvefs_mask = None

        forward_kwargs = {
            "timestep": timesteps,
        }

        if config.unet._class_name == "UNetSpatioTemporalConditionModel":
            dummy_added_time_ids = torch.zeros((B, config.unet.addition_time_embed_dim), device=accelerator.device, dtype=weight_dtype)
            forward_kwargs["added_time_ids"] = dummy_added_time_ids

        # reverse diffusionn loop
        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2) if config.validation_guidance > 1.0 else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
            latent_model_input = torch.cat((latent_model_input, ref_frames), dim=1) # B x 2C x T x H x W
            latent_model_input, padding = format_input(latent_model_input, mult=3)
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=lvefs).sample
            noise_pred = format_output(noise_pred, pad=padding)
            if config.validation_guidance > 1.0:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + config.validation_guidance * (noise_pred_cond - noise_pred_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    # VAE decoding
    with torch.no_grad(): # no autocast
        if val_vae.__class__.__name__ == "AutoencoderKL": # is 2D
            latents = rearrange(latents, "b c t h w -> (b t) c h w")
        latents = latents / val_vae.config.scaling_factor
        videos = val_vae.decode(latents.float()).sample
        videos = (videos + 1) * 128 # [-1, 1] -> [0, 256]
        videos = videos.clamp(0, 255).to(torch.uint8).cpu()
        if val_vae.__class__.__name__ == "AutoencoderKL": # is 2D
            videos = rearrange(videos, "(b t) c h w -> b c t h w", b=B)
    
        ref_frames = ref_frames[:,:,0,:,:]# B x C x H x W
        ref_frames = ref_frames / val_vae.config.scaling_factor
        ref_frames = val_vae.decode(ref_frames.float()).sample
        ref_frames = (ref_frames + 1) * 128 # [-1, 1] -> [0, 256]
        ref_frames = ref_frames.clamp(0, 255).to(torch.uint8).cpu()
        ref_frames = ref_frames[:,:,None,:,:].repeat(1,1,config.validation_frames,1,1) # B x C x T x H x W

        ref_videos = torch.stack(ref_videos, dim=0).to(device=accelerator.device) # B x C x T x H x W
        if val_vae.__class__.__name__ == "AutoencoderKL": # is 2D
            ref_videos = rearrange(ref_videos, "b c t h w -> (b t) c h w")
        ref_videos = ref_videos / val_vae.config.scaling_factor
        ref_videos = val_vae.decode(ref_videos.float()).sample
        ref_videos = (ref_videos + 1) * 128 # [-1, 1] -> [0, 256]
        ref_videos = ref_videos.clamp(0, 255).to(torch.uint8).cpu()
        if val_vae.__class__.__name__ == "AutoencoderKL": # is 2D
            ref_videos = rearrange(ref_videos, "(b t) c h w -> b c t h w", b=B)
        
        videos = torch.cat([ref_frames, ref_videos, videos], dim=3) # B x C x T x (3 H) x W // vertical concat

    # reshape for wandb
    videos = rearrange(videos, "b c t h w -> t c h (b w)") # prepare for wandb
    videos = videos.numpy()

    logger.info("Done sampling... ")
    if config.validation_fps == "original":
        config.validation_fps = 50.0
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log({"validation": wandb.Video(videos, caption=("Lvefs: " + ", ".join([str(e) for e in config.validation_lvefs])), fps=config.validation_fps)})
            logger.info("Samples sent to wandb.")
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del val_unet
    del val_vae
    torch.cuda.empty_cache()

    return videos

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, help="Path to the config file.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    # Setup accelerator
    logging_dir = os.path.join(config.output_dir, config.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if config.seed is not None:
        set_seed(config.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)

    noise_scheduler_kwargs = OmegaConf.to_container(config.noise_scheduler, resolve=True)
    noise_scheduler_klass_name = noise_scheduler_kwargs.pop("_class_name")
    noise_scheduler_klass = globals().get(noise_scheduler_klass_name, None)
    assert noise_scheduler_klass is not None, f"Could not find class {noise_scheduler_klass_name}"
    noise_scheduler = noise_scheduler_klass(**noise_scheduler_kwargs)

    vae = AutoencoderKL.from_pretrained(config.vae_path).cpu()

    # Create the video unet
    unet, unet_klass, unet_kwargs = instantiate_from_config(config.unet, ["diffusers"], return_klass_kwargs=True)

    format_input = pad_reshape if config.unet._class_name == "UNetSpatioTemporalConditionModel" else padf
    format_output = unpad_reshape if config.unet._class_name == "UNetSpatioTemporalConditionModel" else unpadf

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    unet.train()

    # Create EMA for the unet.
    if config.use_ema:
        ema_unet = unet_klass(**unet_kwargs)
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=unet_klass, model_config=ema_unet.config)

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if config.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if config.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), unet_klass)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = unet_klass.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    train_dataset = instantiate_dataset(config.datasets, split=["TRAIN"])
    val_dataset = instantiate_dataset(config.datasets, split=["VAL"])

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.train_batch_size,
        num_workers=config.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if config.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        config.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        config.mixed_precision = accelerator.mixed_precision

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if overrode_max_train_steps:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    config.num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = OmegaConf.to_container(config, resolve=True)
        # tracker_config = dict(vars(tracker_config))
        accelerator.init_trackers(
            config.tracker_project_name, 
            tracker_config,
            init_kwargs={
                "wandb": {
                    "group": config.wandb_group
                },
            },
        )

    # Train!
    total_batch_size = config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps
    model_num_params = sum(p.numel() for p in unet.parameters())
    model_trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    logger.info(f"  U-Net: Total params = {model_num_params} \t Trainable params = {model_trainable_params} ({model_trainable_params/model_num_params*100:.2f}%)")
    global_step = 0
    first_epoch = 0


    # Potentially load in the weights and states from a previous save
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint != "latest":
            path = os.path.basename(config.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            config.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(config.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        # disable=not accelerator.is_local_main_process,
        disable=not accelerator.is_main_process,
    )

    uncond_p = config.get("drop_conditionning", 0.3)

    for epoch in range(first_epoch, config.num_train_epochs):
        train_loss = 0.0
        prediction_mean = 0.0
        prediction_std = 0.0
        target_mean = 0.0
        target_std = 0.0
        mean_losses = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):

                latents = batch['video'] # B x C x T x H x W
                lvef = batch['lvef'] # B
                ref_frame = batch['image'] # B x C x H x W
                padding_indices = batch['padding'] # B

                B, C, T, H, W = latents.shape

                # Loss mask from padding indices
                # Create a tensor of frame indices (0 to T-1)
                frame_indices = torch.arange(0, T, device=accelerator.device).unsqueeze(0).repeat(B, 1)  # Shape (B, T)

                # Get a mask for each video in the batch where frames <= the last non-pad frame are 1, else 0
                mask = (frame_indices <= padding_indices.unsqueeze(1)).float()  # Shape (B, T)

                # Reshape or expand the mask to match the latents' shape for broadcasting
                mask = mask.view(B, 1, T, 1, 1).expand(-1, C, -1, H, W)  # Shape (B, C, T, H, W)

                lvef = lvef[:, None, None] # B -> B x 1 x 1
                lvef_mask = torch.rand_like(lvef, device=accelerator.device, dtype=weight_dtype) > uncond_p

                ref_frame = ref_frame[:,:,None,:,:].repeat(1,1,T,1,1) # B x C x T x H x W

                # Sample a random timestep for each video
                timesteps = torch.randint(0, int(noise_scheduler.config.num_train_timesteps), (B,), device=latents.device).long()

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if config.noise_offset > 0:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += config.noise_offset * torch.randn( (latents.shape[0], latents.shape[1], 1, 1, 1), device=latents.device)
                
                if config.get('input_perturbation', 0) > 0.0:
                    noisy_latents = noise_scheduler.add_noise(latents, noise + config.input_perturbation*torch.rand(1,).item() * torch.randn_like(noise), timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                noisy_latents = torch.cat((noisy_latents, ref_frame), dim=1) # B x 2C x T x H x W

                forward_kwargs = {
                    "timestep": timesteps,
                    "encoder_hidden_states": lvef,
                }

                if config.unet._class_name == "UNetSpatioTemporalConditionModel":
                    dummy_added_time_ids = torch.zeros((B, config.unet.addition_time_embed_dim), device=accelerator.device, dtype=weight_dtype)
                    forward_kwargs["added_time_ids"] = dummy_added_time_ids

                # Predict the noise residual and compute loss
                noisy_latents, padding = format_input(noisy_latents, mult=3)
                model_pred = unet(sample=noisy_latents, **forward_kwargs).sample
                model_pred = format_output(model_pred, pad=padding)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    assert noise_scheduler.config.prediction_type == "sample", f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    target = latents

                # loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss * mask
                loss = loss.mean()
                mean_loss = loss.item()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(config.train_batch_size)).mean()
                train_loss += avg_loss.item() / config.gradient_accumulation_steps
                mean_losses += mean_loss / config.gradient_accumulation_steps
                prediction_mean += model_pred.mean().item() / config.gradient_accumulation_steps
                prediction_std += model_pred.std().item() / config.gradient_accumulation_steps
                target_mean += target.mean().item() / config.gradient_accumulation_steps
                target_std += target.std().item() / config.gradient_accumulation_steps
                

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if config.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({
                    "train_loss": train_loss,
                    "prediction_mean": prediction_mean,
                    "prediction_std": prediction_std,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "mean_losses": mean_losses,
                    }, step=global_step)
                train_loss = 0.0
                prediction_mean = 0.0
                prediction_std = 0.0
                target_mean = 0.0
                target_std = 0.0
                mean_losses = 0.0

                if global_step % config.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if config.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(config.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= config.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - config.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(config.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        OmegaConf.save(config, os.path.join(config.output_dir, "config.yaml"))
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= config.max_train_steps:
                break

            if accelerator.is_main_process:
                if global_step % config.validation_steps == 0:
                    if config.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())

                    log_validation(
                        config,
                        unet,
                        vae,
                        deepcopy(noise_scheduler),
                        accelerator,
                        weight_dtype,
                        epoch,
                        val_dataset,
                    )

                    if config.use_ema:
                        # Switch back to the original UNet parameters.
                        ema_unet.restore(unet.parameters())

    accelerator.end_training()


if __name__ == "__main__":
    main()