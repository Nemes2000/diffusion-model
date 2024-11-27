import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from model.scheduler.time_scheduler import TimeScheduler
from config import Config
from util import up_scale_images

class DDPModule(pl.LightningModule):
    def __init__(self, time_scheduler: TimeScheduler, model: nn.Module, inverse_transform=None):
        super().__init__()
        self.save_hyperparameters()
        self.time_scheduler = time_scheduler
        self.inverse_transform = inverse_transform
        self.model = model
        
        # Setup Inception and Frechet Inception Distance scoring for test
        self.inception = InceptionScore()
        self.fid = FrechetInceptionDistance(feature=64)

    def forward(self, x, t):
        out = self.model(x, t)
        return out

    def training_step(self, batch, batch_idx):
        image, _ = batch
        t = torch.randint(0, self.time_scheduler.timesteps, (Config.batch_size,)).long().to(self.device)
        batch_noisy, noise = self.time_scheduler.q_sample(image, t)
        predicted_noise = self(batch_noisy, t)
        loss = torch.nn.functional.mse_loss(noise, predicted_noise)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, _ = batch
        t = torch.randint(0, self.time_scheduler.timesteps, (image.shape[0],)).long().to(self.device)
        batch_noisy, noise = self.time_scheduler.q_sample(image, t)
        predicted_noise = self(batch_noisy, t)
        loss = torch.nn.functional.mse_loss(noise, predicted_noise)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        image, _ = batch
        t = torch.randint(0, self.time_scheduler.timesteps, (image.shape[0],)).long().to(self.device)
        batch_noisy, noise = self.time_scheduler.q_sample(image, t)
        predicted_noise = self(batch_noisy, t)
        loss = torch.nn.functional.mse_loss(noise, predicted_noise)
        upscaled_orig = up_scale_images(image, self.inverse_transform)
        upscaled_gen = up_scale_images(batch_noisy - predicted_noise, self.inverse_transform)
    
        self.inception.update(upscaled_gen)
        self.fid.update(upscaled_orig, real=True)
        self.fid.update(upscaled_gen, real=False)
        self.log("test_loss", loss)
        return loss
    
    def on_test_epoch_end(self):
        inception_score = self.inception.compute()
        fid_score = self.fid.compute()
        self.log("test_inception_mean", inception_score[0])
        self.log("test_inception_std", inception_score[1])
        self.log("test_fid", fid_score)
        return super().on_test_epoch_end()

    def configure_optimizers(self):
        return Config.optimizer(self.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
