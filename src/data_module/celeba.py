import pytorch_lightning as pl
from torchvision import transforms
from config import Config
from torch.utils.data import DataLoader, Subset
import numpy as np
from data_module.celeba_dataset import CelebADataset
from datasets import load_dataset

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
        self.dataset = load_dataset("nielsr/CelebA-faces", split="train", keep_in_memory=True)
        train_split = 0.6
        val_split = 0.2
        train_end_index = int(len(self.dataset) * train_split)
        val_end_index = int(len(self.dataset) * (train_split + val_split))
        self.train_dataset = CelebADataset(Subset(self.dataset, range(train_end_index)), transform=self.train_transform)
        self.val_dataset = CelebADataset(Subset(self.dataset, range(train_end_index, val_end_index)), transform=self.test_transform)
        self.test_dataset = CelebADataset(Subset(self.dataset, range(val_end_index, len(self.dataset))), transform=self.test_transform)

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