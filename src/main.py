from data_module.flowers import Flowers102DataModule
from data_module.celeba import CelebADataModule
from baseline_model.vae import BaseLineImageGenerationVAE
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from data_visualization.plot_image import plot_image_from_latent_dim, plot_image_representations
import torch

def plot_n_representation(model: BaseLineImageGenerationVAE, dataloader: DataLoader, transform: transforms):
    for batch, _ in dataloader:
        permuted_batch = torch.randperm(batch.size(0))
        first_image = batch[permuted_batch[0]]
        plot_image_representations(model, first_image, transform)
        break

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-path', type=str)
    parser.add_argument('-model', type=str)
    parser.add_argument('-dataset', type=str, choices=['flowers', 'celeba'], default='flowers')
    parser.add_argument('-mode', type=str, choices=['latent', 'random'], default='latent')

    args = parser.parse_args()

    if args.dataset == 'flowers':
        data_module = Flowers102DataModule()
    else:
        data_module = CelebADataModule()

    data_module.prepare_data()
    data_module.setup()  

    model_path = f'{args.path}/{args.model}.ckpt'
    model = BaseLineImageGenerationVAE.load_from_checkpoint(model_path)

    if args.mode == 'latent':
        plot_image_from_latent_dim(model, data_module.reverse_transform)
    else:
        plot_n_representation(model, data_module.test_dataloader(), data_module.reverse_transform)

    


    

