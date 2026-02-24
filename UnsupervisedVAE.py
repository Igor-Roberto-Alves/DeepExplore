import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class UnsupVAE:
    def __init__(self, latent_dim=16):
        super().__init__()

        self.encoder = nn.sequential(
            nn.conv2d(
                1,
                32,
            )
        )
