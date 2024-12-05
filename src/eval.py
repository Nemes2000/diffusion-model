import pytorch_lightning as pl
from baseline_model.vae import BaseLineImageGenerationVAE
import argparse
from data_module.flowers import Flowers102DataModule
from data_module.celeba import CelebADataModule
import json
from model.ddpm_v2.diffusion import DiffusionModel
from model.scheduler.function import LinearScheduleFn
from model.ddpm_v2.module import DDPMModule

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type=str, help="The path to the folder where the desired model for gradio app is.")
    parser.add_argument('-model', type=str, help="The model name, which will be used in the gradio service.")
    parser.add_argument('-dataset', type=str, choices=['flowers', 'celeba'], default='flowers', help="On this dataset will the train run. This can be the 'flowers' or 'celeba'.")
    parser.add_argument('-stat-file', type=str, default='stat.json', help="Statistics from the model will be saved into this file.")
    parser.add_argument('-type', type=str, choices=['baseline', 'diffusion'], default='baseline', help="Select the model for the training. Chose from 'baseline' or 'diffusion'.")
    args = parser.parse_args()

    model_path = f'{args.path}/{args.model}.ckpt'
    
    trainer = pl.Trainer(logger=False)

    if args.dataset == 'flowers':
        data_module = Flowers102DataModule()
    else:
        data_module = CelebADataModule()

    data_module.prepare_data()
    data_module.setup()  

    if args.type == 'baseline':
        model = BaseLineImageGenerationVAE.load_from_checkpoint(model_path, strict=False)

    else:
        diffusion_model = DiffusionModel(function=LinearScheduleFn(beta_start=0.0001, beta_end=0.02))
        model = DDPMModule.load_from_checkpoint(model_path, strict=False)
        model.diffusion_model = diffusion_model
    
    model.inverse_transform = data_module.reverse_transform

    test_result = trainer.test(model, datamodule=data_module)

    with open(args.stat_file, 'w') as output:
        json.dump({
            'inception_mean': test_result[0]['test_inception_mean'],
            'inception_std': test_result[0]['test_inception_std'],
            'fid': test_result[0]['test_fid'],
            'loss': test_result[0]['test_loss']
        }, output, indent=4)

