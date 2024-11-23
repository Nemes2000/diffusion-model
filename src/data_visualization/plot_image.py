import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from baseline_model.vae import BaseLineImageGenerationVAE
from model.ddpm import DDPModule
from config import Config
import torch
from torchvision import transforms

def show_images(data_loader: DataLoader, reverse_transform):
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

def plot_from_noise(model: DDPModule, transform: transforms):
    image = model.time_scheduler.sample(model.model, Config.image_target_size[0], 1, 3)[-1]
    image = transform(image[0])
    plt.imshow(image)
    plt.tight_layout()
    plt.show()