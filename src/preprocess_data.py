import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from config import Config


transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(Config.image_target_size),
    transforms.ToTensor(), 
    transforms.Lambda(lambda t: (t * 2) - 1),
])

reverse_transform = transforms.Compose([
    transforms.Lambda(lambda t: (t + 1) / 2),
    transforms.Lambda(lambda t: t.permute(1, 2, 0)),
    transforms.Lambda(lambda t: t * 255.),
    transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
    transforms.ToPILImage(),
])

flowers_transform = transforms.Compose([
    transforms.CenterCrop(500),
    transforms.Resize(Config.image_target_size),
    transforms.ToTensor(), 
    transforms.Lambda(lambda t: (t * 2) - 1),
])


class DDPMImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = [os.path.join(root, img) for img in os.listdir(root) if img.endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0
    

def get_loader(dataset: DDPMImageDataset, batch_size: int, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
