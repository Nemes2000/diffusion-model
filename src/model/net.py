from torch import nn
from model.block import Block
from config import Config
import torch

class UNet(nn.Module):
    def __init__(self, img_channels = 3, time_embedding_dims = Config.time_embedding_dims, sequence_channels = (64, 128, 256)):
        super().__init__()
        self.time_embedding_dims = time_embedding_dims

        self.downsampling = nn.ModuleList([Block(channels_in, channels_out, time_embedding_dims) for channels_in, channels_out in zip(sequence_channels, sequence_channels[1:])])
        self.upsampling = nn.ModuleList([Block(channels_in, channels_out, time_embedding_dims,downsample=False) for channels_in, channels_out in zip(sequence_channels[::-1], sequence_channels[::-1][1:])])
        self.conv1 = nn.Conv2d(img_channels, sequence_channels[0], 3, padding=1)
        self.conv2 = nn.Conv2d(sequence_channels[0], img_channels, 1)


    def forward(self, x, t, **kwargs):
        residuals = []
        o = self.conv1(x)
        for ds in self.downsampling:
            o = ds(o, t, **kwargs)
            residuals.append(o)
        for us, res in zip(self.upsampling, reversed(residuals)):
            o = us(torch.cat((o, res), dim=1), t, **kwargs)

        return self.conv2(o)