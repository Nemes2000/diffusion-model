import pytorch_lightning as pl
from torchvision import transforms
from config import Config
from torch.utils.data import DataLoader
import numpy as np
from data_module.celeba_dataset import CelebADataset

class CelebADataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(Config.image_target_size),
            transforms.ToTensor(), 
            transforms.Lambda(lambda t: (t * 2) - 1),
        ])
        self.test_transform = transforms.Compose([
            transforms.CenterCrop(178),
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

    def setup(self, stage=None):
        self.train_dataset = CelebADataset(root='./data/dataset/train', transform=self.train_transform)
        self.val_dataset = CelebADataset(root='./data/dataset/validation', transform=self.train_transform)
        self.test_dataset = CelebADataset(root='./data/dataset/test', transform=self.train_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=Config.batch_size,
            num_workers=Config.num_of_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=Config.batch_size,
            num_workers=Config.num_of_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=Config.batch_size,
            num_workers=Config.num_of_workers,
            pin_memory=True,
        )