import wandb
from baseline_model.vae import BaseLineImageGenerationVAE
from config import Config
from data_module.celeba import CelebADataModule
from data_module.flowers import Flowers102DataModule
from model.ddpm_v2.diffusion import DiffusionModel
from model.ddpm_v2.module import DDPMModule
from model.scheduler.function import LinearScheduleFn
import argparse
import pytorch_lightning as pl
import os



def optimalization(data_module, project_name, model):
    """ Called in each optimalization setup for training, and model evaluation.

        - data_module: the selected dataset on optimalization will be made
        - project_name: into this project folder will the wandb log
        - model: the selected model for optimalization 
    """
    sweep_config = {
        'method': 'bayes',
        'metric': {
        'goal': 'minimize',
        'name': 'val_loss'
      }
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
            'distribution': 'q_log_uniform_values',
            'q': 1,
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

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project=project_name)

    wandb.agent(sweep_id=sweep_id, function=wrapped_opt_train_function(data_module=data_module, model=model), count=Config.optimalization_step)
    wandb.teardown()


def wrapped_opt_train_function(data_module, model):
    """Wrapper function for passing multiple parameters for the optimalization function.
    """
    def train_wrapper(config=None):
        optimalization_train(config=config, data_module=data_module, model=model)
    return train_wrapper

def optimalization_train(config=None, data_module=None, model=None):
    """ Called in each optimalization setup for training, and model evaluation.

        - config: given config from wandb sweeper
        - data_module: the selected dataset on optimalization will be made
        - model: the selected model for optimalization 
    """
    with wandb.init(config=config):
        config = wandb.config

        Config.optimizer = Config.optimizer_map[config.optimizer]
        Config.learning_rate = config.learning_rate
        Config.num_of_epochs = config.epoch
        Config.batch_size = config.batch_size
        Config.weight_decay = config.weight_decay
        Config.time_embedding_dims = config.time_embedding_dims

        
        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", dirpath=f'./model/{Config.model_name}', filename='best', save_top_k=1)
        early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", patience=10, verbose=False, mode="min")

        logger = pl.loggers.WandbLogger(project=args.wandb_project, log_model="all")
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
    parser.add_argument('-dataset', type=str, choices=['flowers', 'celeba'], default='flowers', help="On this dataset will the train run. This can be the 'flowers' or 'celeba'.")
    parser.add_argument('--wandb-project', type=str, default='diffusion-model', help="Into this project folder will the wandb log")
    parser.add_argument('-model-name', type=str, default='diffusion-model', help="In this folder will be the logs.")
    parser.add_argument('-type', type=str, choices=['baseline', 'diffusion'], default='baseline', help="Select the model for the training. Chose from 'baseline' or 'diffusion'.")
    args = parser.parse_args()

    os.environ["WANDB_CACHE_DIR"] = "./cache/wandb"
 
    wandb.login(key=os.getenv('WANDB_API_KEY'))

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
        diffusion_model = DiffusionModel(function=LinearScheduleFn(beta_start=0.0001, beta_end=0.02))
        model = DDPMModule(diffusion_model, inverse_transform=data_module.reverse_transform)

    optimalization(data_module=data_module, project_name=args.wandb_project, model=model)