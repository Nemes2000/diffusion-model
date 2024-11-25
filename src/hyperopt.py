import wandb
from baseline_model.vae import BaseLineImageGenerationVAE
from config import Config
from data_module.celeba import CelebADataModule
from data_module.flowers import Flowers102DataModule
from model.ddpm import DDPModule
from model.net import UNet
from model.scheduler.function import LinearScheduleFn
from model.scheduler.time_scheduler import TimeScheduler
from train import train_node_classifier
import argparse
import pytorch_lightning as pl



def optimalization(data_module, project_name):
    sweep_config = {
        'method': 'bayes'
    }

    parameters_dict = {
        'optimizer': {
            'values': ['adam', 'sgd', 'adamW']
        },
        "weight_decay": {
            "values": [1e-4, 3e-4, 5e-4]
        },
        'time_embedding_dims': {
            'values': [64, 128, 256]
        },
    }

    parameters_dict.update({
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
        },
        'epoch': {
            'distribution': 'uniform',
            'min': 30,
            'max': 100
        },
        'batch_size': {
            'distribution': 'q_log_uniform_values',
            'q': 8,
            'min': 64,
            'max': 512
      }
    }
    )

    sweep_id = wandb.sweep(sweep_config, project=project_name)

    wandb.agent(sweep_id=sweep_id, function=wrapped_opt_train_function(data_module=data_module), count=Config.optimalization_step)
    wandb.teardown()


def wrapped_opt_train_function(data_module):
    def train_wrapper(config=None):
        optimalization_train(config=config, data_module=data_module)
    return train_wrapper

def optimalization_train(config=None, data_module=None):
    with wandb.init(config=config):
        config = wandb.config

        Config.optimizer = Config.optimizer_map[config.optimizer]
        Config.learning_rate = config.learning_rate
        Config.num_of_epochs = config.epoch
        Config.batch_size = config.batch_size
        Config.weight_decay = config.weight_decay
        Config.time_embedding_dims = config.time_embedding_dims

        
        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", dirpath=f'./model/{Config.model_name}', filename='best', save_top_k=1)
        early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", patience=5, verbose=False, mode="min")

        trainer = pl.Trainer(
            max_epochs=Config.num_of_epochs,
            logger=logger,
            devices=1,
            callbacks=[checkpoint_callback, early_stopping_callback],
            log_every_n_steps=1,
        )

        trainer.fit(model, data_module)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, choices=['flowers', 'celeba'], default='flowers')
    parser.add_argument('--wandb-project', type=str, default='diffusion-model')
    parser.add_argument('-model-name', type=str, default='diffusion-model')
    args = parser.parse_args()

    wandb.login()
    logger = pl.loggers.WandbLogger(project=args.wandb_project, log_model="all")

    Config.model_name = args.model_name

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
        unet = UNet()
        model = DDPModule(time_scheduler=time_scheduler, model=unet, inverse_transform=data_module.reverse_transform)

    optimalization(data_module=data_module, project_name=args.wandb_project)