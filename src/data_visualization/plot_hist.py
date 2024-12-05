import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np

def show_rgb_histogram(data_loader: DataLoader, reverse_transform):
    """ Ploting the first batch's five images' rgb histogram from the given dataloader.
    """
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