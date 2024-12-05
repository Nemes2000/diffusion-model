from data_module.flowers import Flowers102DataModule
from data_module.celeba import CelebADataModule
from baseline_model.vae import BaseLineImageGenerationVAE
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from data_visualization.plot_image import plot_image_from_latent_dim, plot_image_representations, plot_from_noise
import torch
from model.ddpm_v2.diffusion import DiffusionModel
from model.scheduler.function import LinearScheduleFn
from model.ddpm_v2.module import DDPMModule

def plot_n_representation(model: BaseLineImageGenerationVAE, dataloader: DataLoader, transform: transforms):
    """ Plots an image matrix from the VAE model latent dimension space close to the original images.

        - model: an instance of the BaseLineImageGenerationVAE class.
        - dataloader: the selected dataloader
        - transform: the reverse transformation defined to get from torch an image.
    """
    for batch, _ in dataloader:
        permuted_batch = torch.randperm(batch.size(0))
        first_image = batch[permuted_batch[0]]
        plot_image_representations(model, first_image, transform)
        break

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-path', type=str, help="The path to the folder where the desired model for gradio app is.")
    parser.add_argument('-model', type=str, help="The model name, which will be used in the gradio service.")
    parser.add_argument('-dataset', type=str, choices=['flowers', 'celeba'], default='flowers', help="On this dataset will the train run. This can be the 'flowers' or 'celeba'.")
    parser.add_argument('-mode', type=str, choices=['latent', 'random', 'noise'], default='latent', help="Images will be generated from the given space/method by this arg.")
    parser.add_argument('-type', type=str, choices=['baseline', 'diffusion'], default='baseline', help="Select the model for the training. Chose from 'baseline' or 'diffusion'.")

    args = parser.parse_args()

    if args.dataset == 'flowers':
        data_module = Flowers102DataModule()
    else:
        data_module = CelebADataModule()

    data_module.prepare_data()
    data_module.setup()  

    model_path = f'{args.path}/{args.model}.ckpt'

    if args.type == 'baseline':
        model = BaseLineImageGenerationVAE.load_from_checkpoint(model_path, strict=False)
    else:
        diffusion_model = DiffusionModel(function=LinearScheduleFn(beta_start=0.0001, beta_end=0.02))
        model = DDPMModule.load_from_checkpoint(model_path, strict=False)
        model.diffusion_model = diffusion_model

    if args.mode == 'latent':
        plot_image_from_latent_dim(model, data_module.reverse_transform)
    elif args.mode == 'noise':
        plot_from_noise(model, data_module.reverse_transform)
    else:
        plot_n_representation(model, data_module.test_dataloader(), data_module.reverse_transform)

    


    

