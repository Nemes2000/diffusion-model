import pytorch_lightning as pl
from torchvision import transforms
from config import Config
import torchvision
from torch.utils.data import DataLoader
import numpy as np

class Flowers102DataModule(pl.LightningDataModule):
    """ Data module for Flowers102 data. 
        We define the transformations for train, test and the reverse transformation, 
        which recreates images from tensors.
    """
    def __init__(self):
        super().__init__()
        self.train_transform = transforms.Compose([
            transforms.CenterCrop(500),
            transforms.Resize(Config.image_target_size),
            transforms.ToTensor(), 
            transforms.Lambda(lambda t: (t * 2) - 1),
        ])
        self.test_transform = transforms.Compose([
            transforms.CenterCrop(500),
            transforms.Resize(Config.image_target_size),
            transforms.ToTensor(), 
            transforms.Lambda(lambda t: (t * 2) - 1),
        ])

        self.reverse_transform = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),
            transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ])

        self.image_folder = './data/flowers102'

    def setup(self):
        """ Calling this method, will download the train, validation and test data
            from source and stored in memory.
        """
        self.train_dataset = torchvision.datasets.Flowers102(root=self.image_folder, split='test', download=True, transform=self.train_transform)
        self.val_dataset = torchvision.datasets.Flowers102(root=self.image_folder, split='val', download=True, transform=self.test_transform)
        self.test_dataset = torchvision.datasets.Flowers102(root=self.image_folder, split='train', download=True, transform=self.test_transform)

    def train_dataloader(self):
        """Returns with dataloader for training.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=Config.batch_size,
            num_workers=Config.num_of_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        """Returns with dataloader for validation.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=Config.batch_size,
            num_workers=Config.num_of_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Returns with dataloader for testing.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=Config.batch_size,
            num_workers=Config.num_of_workers,
            pin_memory=True,
        )