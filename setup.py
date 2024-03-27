from setuptools import setup, find_packages

setup(
    name='echosyn',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'einops',
        'decord',
        'diffusers',
        'packaging',
        'omegaconf',
        'transformers',
        'accelerate',
        'pandas',
        'wandb',
        'opencv-python',
        'moviepy',
        'imageio',
        'scipy',
        'scikit-learn',
        'matplotlib',
    ]
)