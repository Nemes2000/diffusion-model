from preprocess_data import DDPMImageDataset, transform, flowers_transform, reverse_transform, get_loader
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
from config import Config

def get_celeba_loaders():
    train_dataset = DDPMImageDataset(root='../data/dataset/train', transform=transform)
    validation_dataset = DDPMImageDataset(root='../data/dataset/validation', transform=transform)
    test_dataset = DDPMImageDataset(root='../data/dataset/test', transform=transform)

    
    train_loader = get_loader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    test_loader = get_loader(test_dataset, batch_size=Config.batch_size, shuffle=False)
    val_loader = get_loader(validation_dataset, batch_size=Config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_flowers102_loaders():
    train_dataset = torchvision.datasets.Flowers102(root='../data/flowers102', split='train', download=True, transform=flowers_transform)
    val_dataset = torchvision.datasets.Flowers102(root='../data/flowers102', split='val', download=True, transform=flowers_transform)
    test_dataset = torchvision.datasets.Flowers102(root='./data/flowers102', split='test', download=True, transform=flowers_transform)


    train_loader = get_loader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = get_loader(val_dataset, batch_size=Config.batch_size, shuffle=False)
    test_loader = get_loader(test_dataset, batch_size=Config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_flowers102_loaders()

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
    for batch, _ in train_loader:
        for ax, item in zip(axes.flat,  batch[:10]):
            image = reverse_transform(item)
            ax.imshow(image)
            ax.axis('off')
        break
    plt.show()

