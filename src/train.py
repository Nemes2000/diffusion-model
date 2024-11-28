import pytorch_lightning as pl
import torch
import wandb
import argparse
from config import Config
from data_module.flowers import Flowers102DataModule
from data_module.celeba import CelebADataModule
from baseline_model.vae import BaseLineImageGenerationVAE
from model.ddpm_v2.diffusion import DiffusionModel
from model.ddpm_v2.module import DDPMModule
from data_visualization.plot_image import plot_from_noise
from model.scheduler.function import LinearScheduleFn, CosineBetaScheduleFn, QuadraticBetaSheduleFn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-log-wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='diffusion-model')
    parser.add_argument('-epoch', type=int, default=Config.num_of_epochs)
    parser.add_argument('-model-name', type=str, default='diffusion-model')
    parser.add_argument('-dataset', type=str, choices=['flowers', 'celeba'], default='flowers')
    parser.add_argument('-type', type=str, choices=['baseline', 'diffusion'], default='baseline')

    args = parser.parse_args()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", dirpath=f'./model/{args.model_name}', filename='best', save_top_k=1)
    early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", patience=30, verbose=False, mode="min")

    if args.log_wandb:
        wandb.login()
        logger = pl.loggers.WandbLogger(project=args.wandb_project, log_model="all")
    else:
        logger = pl.loggers.tensorboard.TensorBoardLogger(save_dir='./logs', name=args.model_name)

    trainer = pl.Trainer(
        max_epochs=args.epoch,
        logger=logger,
        devices=1,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=1,
    )

    if args.dataset == 'flowers':
        data_module = Flowers102DataModule()
    else:
        data_module = CelebADataModule()

    data_module.prepare_data()
    data_module.setup()  

    if args.type == 'baseline':
        model = BaseLineImageGenerationVAE(Config.latent_dims)
    else:
        diffusion_model = DiffusionModel(function=LinearScheduleFn(beta_start=0.0001, beta_end=0.02))
        model = DDPMModule(diffusion_model, inverse_transform=data_module.reverse_transform)

    trainer.fit(model, data_module)

    if args.type == 'diffusion':
        plot_from_noise(model, data_module.reverse_transform)
        
    if args.log_wandb:
        wandb.finish()

    