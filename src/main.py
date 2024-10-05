from preprocess_data import DDPMImageDataset, transform, flowers_transform, reverse_transform, get_loader
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
from config import Config
import numpy as np

def get_celeba_loaders():
    train_dataset = DDPMImageDataset(root='./data/dataset/train', transform=transform)
    validation_dataset = DDPMImageDataset(root='./data/dataset/validation', transform=transform)
    test_dataset = DDPMImageDataset(root='./data/dataset/test', transform=transform)

    
    train_loader = get_loader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    test_loader = get_loader(test_dataset, batch_size=Config.batch_size, shuffle=False)
    val_loader = get_loader(validation_dataset, batch_size=Config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_flowers102_loaders():
    train_dataset = torchvision.datasets.Flowers102(root='./data/flowers102', split='train', download=True, transform=flowers_transform)
    val_dataset = torchvision.datasets.Flowers102(root='./data/flowers102', split='val', download=True, transform=flowers_transform)
    test_dataset = torchvision.datasets.Flowers102(root='./data/flowers102', split='test', download=True, transform=flowers_transform)


    train_loader = get_loader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = get_loader(val_dataset, batch_size=Config.batch_size, shuffle=False)
    test_loader = get_loader(test_dataset, batch_size=Config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def show_images(data_loader: DataLoader):
    _, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
    for batch, _ in data_loader:
        for ax, item in zip(axes.flat,  batch[:10]):
            image = reverse_transform(item)
            ax.imshow(image)
            ax.axis('off')
        break
    plt.tight_layout()
    plt.show()

def show_rgb_histogram(data_loader: DataLoader):
    plt.figure(figsize=(12, 12))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    for batch, _ in data_loader:
        for index, item in enumerate(batch[:5]):
            image = reverse_transform(item)
            image_array = np.array(image)
            R, G, B = image_array[:,:,0], image_array[:,:,1], image_array[:,:,2]

            plt.subplot(5, 4, index * 4 + 1)
            plt.imshow(image)
            plt.title(f"Image #{index + 1}")

            plt.subplot(5, 4, index * 4 + 2)
            plt.hist(R.flatten(), bins=256, color='red', alpha=0.6)
            plt.title('R')

            plt.subplot(5, 4, index * 4 + 3)
            plt.hist(G.flatten(), bins=256, color='green', alpha=0.6)
            plt.title('G')

            plt.subplot(5, 4, index * 4 + 4)
            plt.hist(B.flatten(), bins=256, color='blue', alpha=0.6)
            plt.title('B')
        break
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_flowers102_loaders()
    show_rgb_histogram(val_loader)
    

