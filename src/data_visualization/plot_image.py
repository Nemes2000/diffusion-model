import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

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
