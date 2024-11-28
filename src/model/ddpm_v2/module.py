from torchmetrics.regression import MeanSquaredError
import pytorch_lightning as pl
import torch
import torch.nn as nn
from model.ddpm_v2.block import Block
from config import Config

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

class DDPM(pl.LightningModule):
    def __init__(self, diffusion_model, img_channels = 3, time_embedding_dims = 128, labels = False, sequence_channels = (64, 128, 256, 512, 1024)):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.time_embedding_dims = time_embedding_dims
        sequence_channels_rev = reversed(sequence_channels)

        self.downsampling = nn.ModuleList([Block(channels_in, channels_out, time_embedding_dims, labels) for channels_in, channels_out in zip(sequence_channels, sequence_channels[1:])])
        self.upsampling = nn.ModuleList([Block(channels_in, channels_out, time_embedding_dims, labels,downsample=False) for channels_in, channels_out in zip(sequence_channels[::-1], sequence_channels[::-1][1:])])
        self.conv1 = nn.Conv2d(img_channels, sequence_channels[0], 3, padding=1)
        self.conv2 = nn.Conv2d(sequence_channels[0], img_channels, 1)
        
        self.mse = MeanSquaredError()
        

    def forward(self, x, t):
        residuals = []
        o = self.conv1(x)
        for ds in self.downsampling:
            o = ds(o, t)
            residuals.append(o)
        for us, res in zip(self.upsampling, reversed(residuals)):
            o = us(torch.cat((o, res), dim=1), t)

        return self.conv2(o)

    def training_step(self, batch, batch_idx):
        images, _ = batch
        t = torch.randint(0, self.diffusion_model.timesteps, (images.shape[0],)).long().to(device)
        images = images.to(device)
        batch_noisy, noise = self.diffusion_model.forward(images, t, device)
        predicted_noise = self.forward(batch_noisy, t)
        loss = self.mse(noise, predicted_noise)

        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        t = torch.randint(0, self.diffusion_model.timesteps, (images.shape[0],)).long().to(device)
        images = images.to(device)

        batch_noisy, noise = self.diffusion_model.forward(images, t, device)
        predicted_noise = self.forward(batch_noisy, t)

        loss = self.mse(noise, predicted_noise)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, _ = batch
        t = torch.randint(0, self.diffusion_model.timesteps, (images.shape[0],)).long().to(device)
        images = images.to(device)

        batch_noisy, noise = self.diffusion_model.forward(images, t, device)
        predicted_noise = self.forward(batch_noisy, t)

        loss = self.mse(noise, predicted_noise)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)