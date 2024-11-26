

# Common imports
import os
import torch
import numpy as np
from tqdm import tqdm

# Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST

# Data Visualization
import matplotlib.pyplot as plt

# Model
from torch import nn
from torch.nn import Module, Sequential
from torch.nn import Conv2d, ConvTranspose2d
from torch.nn import BatchNorm2d, ReLU, LeakyReLU, Tanh

# ENV Constants
DEVICE = "cpu"
IMG_DIMS = 28

# Model Constants
ZDIM = 32
HDIM = 32
IMG_CHANNELS = 1

# Training Constants
LR = 2e-4
EPOCHS = 20                           # Change with respect to your hardware
beta_1 = 0.5
C_LAMBDA = 10
beta_2 = 0.999
BATCH_SIZE = 16
CRITIC_STEPS = 5
DISPLAY_STEP = 500



class Generator(Module):
    """
    Generator for a Spectral Normalization Generative Adversarial Network (SN GAN).

    Parameters:
    - zdims (int): Dimensionality of the input noise vector.
    - hdims (int): Dimensionality of the hidden layers in the generator.
    - img_channels (int): Number of channels in the output synthesized images.

    Attributes:
    - zdims (int): Dimensionality of the input noise vector.
    - generator (Sequential): Sequential model representing the generator architecture.
    """

    def __init__(self, zdims: int = ZDIM, hdims: int = HDIM, img_channels: int = IMG_CHANNELS):
        """
        Initializes the Generator.

        Args:
        - zdims (int): Dimensionality of the input noise vector.
        - hdims (int): Dimensionality of the hidden layers in the generator.
        - img_channels (int): Number of channels in the output synthesized images.
        """

        super(Generator, self).__init__()

        self.zdims = zdims
        self.generator = Sequential(
            self.generator_block(ZDIM, HDIM),
            self.generator_block(HDIM, HDIM * 2, kernel_size=4, stride=1),
            self.generator_block(HDIM * 2, HDIM * 4),
            self.generator_block(HDIM * 4, img_channels, kernel_size=4, output_layer=True),
        )

    def generator_block(self, input_dims: int, output_dims: int, kernel_size: int = 3, stride: int = 2, output_layer: bool = False):
        """
        Defines a generator block.

        Args:
        - input_dims (int): Dimensionality of the input to the block.
        - output_dims (int): Dimensionality of the output from the block.
        - kernel_size (int): Size of the convolutional kernel.
        - stride (int): Stride of the convolutional operation.
        - output_layer (bool): Indicates whether this block is the output layer.

        Returns:
        - Sequential: Generator block as a Sequential model.
        """

        if output_layer:
            return Sequential(
                ConvTranspose2d(input_dims, output_dims, kernel_size, stride),
                Tanh()
            )
        else:
            return Sequential(
                ConvTranspose2d(input_dims, output_dims, kernel_size, stride),
                BatchNorm2d(output_dims),
                ReLU(inplace=True)
            )

    def forward(self, noise):
        """
        Forward pass of the generator.

        Args:
        - noise (Tensor): Input noise tensor.

        Returns:
        - Tensor: Synthesized images.
        """

        noise = noise.view(len(noise), self.zdims, 1, 1)
        synthesized_images = self.generator(noise)

        return synthesized_images