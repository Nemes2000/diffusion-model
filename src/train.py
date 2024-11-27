from matplotlib import pyplot as plt
import pytorch_lightning as pl
import wandb
import argparse
from config import Config
from data_module.flowers import Flowers102DataModule
from data_module.celeba import CelebADataModule
from baseline_model.vae import BaseLineImageGenerationVAE
from model.ddpm import DDPModule
from model.net import UNet
from model.net_v2 import Unet
from model.scheduler.time_scheduler import TimeScheduler
from model.scheduler.function import LinearScheduleFn

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
        time_scheduler = TimeScheduler(LinearScheduleFn(0.0001, 0.02), Config.time_steps)
        unet = Unet(
            dim=Config.image_size,
            channels=Config.channels,
            dim_mults=(1, 2, 4,)
        )
        model = DDPModule(time_scheduler=time_scheduler, model=unet, inverse_transform=data_module.reverse_transform)

    trainer.fit(model, data_module)

    samples = time_scheduler.sample(model, image_size=Config.image_size, batch_size=1, channels=Config.channels)
    random_index = 0
    sample = samples[-1]
    plt.imsave('generated_image.jpg', data_module.reverse_transform(sample[random_index]))
    
    if args.log_wandb:
        wandb.finish()

    