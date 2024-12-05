import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from baseline_model.vae import BaseLineImageGenerationVAE
from model.ddpm_v2.module import DDPMModule
from config import Config
import torch
from torchvision import transforms
from tqdm import tqdm

def show_images(data_loader: DataLoader, reverse_transform):
    """ Plots the first ten images from the dataloader.
    """
    _, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
    for batch, _ in data_loader:
        for ax, item in zip(axes.flat,  batch[:10]):
            image = reverse_transform(item)
            ax.imshow(image)
            ax.axis('off')
        break
    plt.tight_layout()
    plt.show()

def plot_image_from_latent_dim(model: BaseLineImageGenerationVAE, transform: transforms, n = 3, latent_dims = Config.latent_dims, random_state = 42):
    """ Plots images sampled from VAE model latent space.

        - model: an instance of the BaseLineImageGenerationVAE class.
        - transform: the reverse transformation defined to get from torch an image.
        - n: there will be n row, and each row will contain n images.
        - latent_dim: the latent space's vectore size
    """
    torch.manual_seed(random_state)
    fig, axn = plt.subplots(n, n, figsize=(8, 8), sharex=True, sharey=True)
    for i in range(n * n):
        with torch.no_grad():
            z = torch.randn(1, latent_dims, dtype=torch.float32)
            img = transform(model.decode(z)[0])
        ax = axn[i // n, i % n]
        ax.imshow(img)
        ax.axis('off')

    fig.suptitle('Generated images from latent dimension')
    plt.tight_layout()
    plt.show()

def plot_image_representations(model: BaseLineImageGenerationVAE, image: torch.Tensor, transform: transforms, n = 5, random_state = 42):
    """ Plots images which are in the VAE model latent dimension space close to the original image.

        - model: an instance of the BaseLineImageGenerationVAE class.
        - image: the image which will be passed to the VAE model
        - transform: the reverse transformation defined to get from torch an image.
        - n: there will be n row, and each row will contain n images.
    """
    torch.manual_seed(random_state)
    fig, axn = plt.subplots(1, n + 1, figsize=(2 *  (n + 1),  2), sharex=True, sharey=True)
    tensor_shape = (1, 3,) + Config.image_target_size

    pic = torch.clone(image).reshape(tensor_shape)
    ax = axn[0]
    ax.imshow(transform(pic[0]))
    ax.axis('off')

    for i in range(n):
        with torch.no_grad():
            recon, _, _ = model(pic)
            pic = recon.reshape(tensor_shape)
        ax = axn[i + 1]
        ax.imshow(transform(pic[0]))
        ax.axis('off')

    fig.suptitle('Generated images from a random image')
    plt.tight_layout()
    plt.show()

def plot_from_noise(model: DDPMModule, transform: transforms, n = 5):
    """ Saves an image matrix which are generated from the same noise with an instance of the diffusion model.

        - model: an instance of the DDPMModule class.
        - transform: the reverse transformation defined to get from torch an image.
        - n: there will be n row, and each row will contain n images.
    """
    plt.figure(figsize=(15,15))
    _, ax = plt.subplots(n, n, figsize = (32,32))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu') 
    for c in tqdm(range(n)):
        imgs = torch.randn((n, 3) + Config.image_target_size).to(device)
        for i in reversed(range(model.diffusion_model.timesteps)):
            t = torch.full((1,), i, dtype=torch.long, device=device)
            _ = torch.tensor([c] * n).resize(n, 1).float().to(device)
            diff_imgs = model.diffusion_model.backward(x=imgs, t=t, model=model.eval().to(device))
            if torch.isnan(diff_imgs).any(): break
            imgs = diff_imgs
        for idx, img in enumerate(imgs):
            ax[c][idx].imshow(transform(img))
            ax[c][idx].axis('off')
    plt.tight_layout()
    plt.savefig('diffusion-model.png')