import pytorch_lightning as pl
import torch
import torch.nn as nn
from model.ddpm_v2.block import Block

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

class InferenceDDPMModule(pl.LightningModule):
    def __init__(self, diffusion_model=None, img_channels = 3, time_embedding_dims = 128, labels = False, sequence_channels = (64, 128, 256, 512, 1024), inverse_transform=None):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.time_embedding_dims = time_embedding_dims
        self.inverse_transform = inverse_transform

        self.downsampling = nn.ModuleList([Block(channels_in, channels_out, time_embedding_dims, labels) for channels_in, channels_out in zip(sequence_channels, sequence_channels[1:])])
        self.upsampling = nn.ModuleList([Block(channels_in, channels_out, time_embedding_dims, labels,downsample=False) for channels_in, channels_out in zip(sequence_channels[::-1], sequence_channels[::-1][1:])])
        self.conv1 = nn.Conv2d(img_channels, sequence_channels[0], 3, padding=1)
        self.conv2 = nn.Conv2d(sequence_channels[0], img_channels, 1)
        

    def forward(self, x, t):
        residuals = []
        o = self.conv1(x)
        for ds in self.downsampling:
            o = ds(o, t)
            residuals.append(o)
        for us, res in zip(self.upsampling, reversed(residuals)):
            o = us(torch.cat((o, res), dim=1), t)

        return self.conv2(o)
