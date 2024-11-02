
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from config import Config
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from util import up_scale_images

class BaseLineImageGenerationVAE(pl.LightningModule):
    def __init__(self, latent_dim, in_channels=3, inverse_transform=None):
        super().__init__()
        self.save_hyperparameters()
        self.inverse_transform = inverse_transform
        hidden_dims=[32, 64, 128, 256, 512]
        self.final_dim = hidden_dims[-1]
        modules = []

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        out = self.encoder(torch.rand(1, 3, Config.image_target_size[0], Config.image_target_size[1]))
        self.size = out.shape[2]
        self.fc_mu = nn.Linear(hidden_dims[-1] * self.size * self.size, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * self.size * self.size, latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.size * self.size)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Sigmoid())
        
        # Setup Inception and Frechet Inception Distance scoring for test
        self.inception = InceptionScore()
        self.fid = FrechetInceptionDistance(feature=64)

    def encode(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.final_dim, self.size, self.size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    @staticmethod
    def loss_function(recon_x, x, mu, log_var):
        MSE = F.mse_loss(recon_x, x.view(-1, 3 * Config.image_target_size[0] * Config.image_target_size[1]))
        KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        kld_weight = 0.00025
        loss = MSE + kld_weight * KLD
        return loss

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        z = self.decode(z)
        z = torch.flatten(z, start_dim=1)
        z = torch.nan_to_num(z)
        return z, mu, log_var

    def training_step(self, batch, batch_idx):
        image, _ = batch
        recon_batch, mu, log_var = self(image)
        log_var = torch.clamp_(log_var, -10, 10)
        loss = BaseLineImageGenerationVAE.loss_function(recon_batch, image, mu, log_var)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, _ = batch
        recon_batch, mu, log_var = self(image)
        log_var = torch.clamp_(log_var, -10, 10)
        loss = BaseLineImageGenerationVAE.loss_function(recon_batch, image, mu, log_var)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        image, _ = batch
        recon_batch, mu, log_var = self(image)
        log_var = torch.clamp_(log_var, -10, 10)
        loss = BaseLineImageGenerationVAE.loss_function(recon_batch, image, mu, log_var)
        upscaled_orig = up_scale_images(image, self.inverse_transform)
        upscaled_gen = up_scale_images(recon_batch.reshape((-1, 3) + Config.image_target_size), self.inverse_transform)
    
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
        return torch.optim.AdamW(self.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
