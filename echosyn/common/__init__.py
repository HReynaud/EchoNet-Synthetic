import os
import json
import importlib

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
import imageio

import diffusers

def padf(tensor, mult=3):
    """
    Pads a tensor along the last dimension to make its size a multiple of 2^mult.

    Args:
        tensor (torch.Tensor): The tensor to pad.
        mult (int, optional): The power of 2 that the tensor's size should be a multiple of. Defaults to 3.

    Returns:
        torch.Tensor: The padded tensor.
        int: The amount of padding applied.
    """
    pad = 2**mult - (tensor.shape[-1] % 2**mult)
    pad = pad//2
    tensor = F.pad(tensor, (pad, pad, pad, pad, 0, 0), mode='replicate')
    return tensor, pad

def unpadf(tensor, pad=1):
    """
    Removes padding from a tensor along the last two dimensions.

    Args:
        tensor (torch.Tensor): The tensor to unpad.
        pad (int, optional): The amount of padding to remove. Defaults to 1.

    Returns:
        torch.Tensor: The unpadded tensor.
    """
    return tensor[..., pad:-pad, pad:-pad]

def pad_reshape(tensor, mult=3):
    """
    Pads a tensor along the last dimension to make its size a multiple of 2^mult and reshapes it.

    Args:
        tensor (torch.Tensor): The tensor to pad and reshape.
        mult (int, optional): The power of 2 that the tensor's size should be a multiple of. Defaults to 3.

    Returns:
        torch.Tensor: The padded and reshaped tensor.
        int: The amount of padding applied.
    """
    tensor, pad = padf(tensor, mult=mult)
    tensor = rearrange(tensor, "b c t h w -> b t c h w")
    return tensor, pad

def unpad_reshape(tensor, pad=1):
    """
    Reshapes a tensor and removes padding from it along the last two dimensions.

    Args:
        tensor (torch.Tensor): The tensor to reshape and unpad.
        pad (int, optional): The amount of padding to remove. Defaults to 1.

    Returns:
        torch.Tensor: The reshaped and unpadded tensor.
    """
    tensor = rearrange(tensor, "b t c h w -> b c t h w")
    tensor = unpadf(tensor, pad=pad)
    return tensor

def instantiate_from_config(config, scope: list[str], return_klass_kwargs=False, **kwargs):
    """
    Instantiate a class from a config dictionary.

    Args:
        config (dict): The config dictionary.
        scope (list[str]): The scope of the class to instantiate.
        return_klass_kwargs (bool, optional): Whether to return the class and its kwargs. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the class constructor.

    Returns:
        object: The instantiated class.
        (optional) type: The class that was instantiated.
        (optional) dict: The kwargs that were passed to the class constructor.
    """
    okwargs = OmegaConf.to_container(config, resolve=True)
    klass_name = okwargs.pop("_class_name")
    klass = None

    for module_name in scope:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue # Try next module

        klass = getattr(module, klass_name, None)
        if klass is not None:
            break  # Stop when we find a matching class

    assert klass is not None, f"Could not find class {klass_name} in the specified scope"
    instance = klass(**okwargs, **kwargs)

    if return_klass_kwargs:
        return instance, klass, okwargs
    return instance

def load_model(path):
    """
    Loads a model from a checkpoint.

    Args:
        path (str): The path to the checkpoint.

    Returns:
        object: The loaded model.
    """
    # find config.json
    json_path = os.path.join(path, "config.json")
    assert os.path.exists(json_path), f"Could not find config.json at {json_path}"
    with open(json_path, "r") as f:
        config = json.load(f)

    # instantiate class
    klass_name = config["_class_name"]
    klass = getattr(diffusers, klass_name, None)
    if klass is None:
        klass = globals().get(klass_name, None)
    assert klass is not None, f"Could not find class {klass_name} in diffusers or global scope."
    assert getattr(klass, "from_pretrained", None) is not None, f"Class {klass_name} does not support 'from_pretrained'."

    # load checkpoint
    model = klass.from_pretrained(path)

    return model

def save_as_mp4(tensor, filename, fps=30):
    """
    Saves a 4D tensor (nFrames, height, width, channels) as an MP4 video.

    Parameters:
    - tensor: 4D torch.Tensor. Tensor containing the video frames.
    - filename: str. The output filename for the video.
    - fps: int. Frames per second for the output video.

    Returns:
    - None
    """
    import imageio
    # Make sure the tensor is on the CPU and is a numpy array
    np_video = tensor.cpu().numpy()
    
    # Ensure the tensor dtype is uint8
    if np_video.dtype != np.uint8:
        raise ValueError("The tensor has to be of type uint8")
    
    # Write the frames to a video file
    with imageio.get_writer(filename, fps=fps, ) as writer:
        for i in range(np_video.shape[0]):
            writer.append_data(np_video[i])

def save_as_avi(tensor, filename, fps=30):
    """
    Saves a 4D tensor (nFrames, height, width, channels) as an AVI video with reduced compression.
    
    Parameters:
    - tensor: 4D torch.Tensor. Tensor containing the video frames.
    - filename: str. The output filename for the video.
    - fps: int. Frames per second for the output video.
    
    Returns:
    - None
    """
    # Make sure the tensor is on the CPU and is a numpy array
    np_video = tensor.cpu().numpy()
    
    # Ensure the tensor dtype is uint8
    if np_video.dtype != np.uint8:
        raise ValueError("The tensor has to be of type uint8")
    
    # Define codec for reduced compression
    codec = "mjpeg"  # MJPEG codec for AVI files
    quality = 10  # High quality (lower values mean higher quality, but larger file sizes)
    # pixel_format = "yuvj420p"
    # Write the frames to a video file
    with imageio.get_writer(filename, fps=fps, codec=codec, quality=quality) as writer:
        for frame in np_video:
            writer.append_data(frame)

def save_as_gif(tensor, filename, fps=30):
    """
    Saves a 4D tensor (nFrames, height, width, channels) as a GIF.

    Parameters:
    - tensor: 4D torch.Tensor. Tensor containing the video frames.
    - filename: str. The output filename for the GIF.
    - fps: int. Frames per second for the output GIF.

    Returns:
    - None
    """
    import imageio
    # Make sure the tensor is on the CPU and is a numpy array
    np_video = tensor.cpu().numpy()
    
    # Ensure the tensor dtype is uint8
    if np_video.dtype != np.uint8:
        raise ValueError("The tensor has to be of type uint8")
    
    # Write the frames to a GIF file
    imageio.mimsave(filename, np_video, fps=fps)

def save_as_img(tensor, filename, ext="jpg"):
    """
    Saves a 4D tensor (nFrames, height, width, channels) as a series of JPG images.
    OR
    Saves a 3D tensor (height, width, channels) as a single image.

    Parameters:
    - tensor: 4D torch.Tensor. Tensor containing the video frames.
    - filename: str. The output filename for the JPG images.

    Returns:
    - None
    """
    import imageio
    # Make sure the tensor is on the CPU and is a numpy array
    np_video = tensor.cpu().numpy()
    
    # Ensure the tensor dtype is uint8
    if np_video.dtype != np.uint8:
        raise ValueError("The tensor has to be of type uint8")
    
    # Write the frames to a series of JPG files
    if len(np_video.shape) == 3:
        imageio.imwrite(filename, np_video, quality=100)
    else:
        os.makedirs(filename, exist_ok=True)
        for i in range(np_video.shape[0]):
            imageio.imwrite(os.path.join(filename, f"{i:04d}.{ext}"), np_video[i], quality=100)

def loadvideo(filename: str, return_fps=False):
    """
    Loads a video file into a tensor of frames.

    Args:
        filename (str): The path to the video file.
        return_fps (bool, optional): Whether to return the frames per second of the video. Defaults to False.

    Raises:
        FileNotFoundError: If the video file does not exist.

    Returns:
        torch.Tensor: A tensor of the video's frames, with shape (frames, 3, height, width).
        (optional) float: The frames per second of the video. Only returned if return_fps is True.
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)  # type: ignore

    fps = capture.get(cv2.CAP_PROP_FPS) # type: ignore

    frames = []

    while True: # load all frames
        ret, frame = capture.read()
        if not ret:
            break  # Reached end of video
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame)

        frames.append(frame)
    capture.release()

    frames = torch.stack(frames, dim=0) # (frames, 3, height, width)

    if return_fps:
        return frames, fps
    return frames

def parse_formats(s):
    # Split the input string by comma and strip spaces
    formats = [format.strip().lower() for format in s.split(',')]
    # Define the allowed choices
    allowed_formats = ["avi", "mp4", "gif", "jpg", "png", "pt"]
    # Check if all elements in formats are in allowed_formats
    for format in formats:
        if format not in allowed_formats:
            raise argparse.ArgumentTypeError(f"{format} is not a valid format. Choose from {', '.join(allowed_formats)}.")
    return formats






