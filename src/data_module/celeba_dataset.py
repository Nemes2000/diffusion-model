from torch.utils.data import Dataset, Subset
import os
from PIL import Image

class CelebADataset(Dataset):
    """ Loads in Celeba dataset in subsets with the given transform methods.

    """
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        image = self.subset[idx]['image']
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0