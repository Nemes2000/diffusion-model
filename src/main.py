from preprocess_data import DDPMImageDataset, transform, reverse_transform, get_loader
import matplotlib.pyplot as plt
from config import Config

if __name__ == "__main__":
    train_dataset = DDPMImageDataset(root='../data/dataset/train', transform=transform)
    validation_dataset = DDPMImageDataset(root='../data/dataset/validation', transform=transform)
    test_dataset = DDPMImageDataset(root='../data/dataset/test', transform=transform)

    
    train_loader = get_loader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    test_loader = get_loader(test_dataset, batch_size=Config.batch_size, shuffle=False)
    val_loader = get_loader(validation_dataset, batch_size=Config.batch_size, shuffle=False)

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
    for batch in train_loader:
        for ax, item in zip(axes.flat,  batch[:10]):
            image = reverse_transform(item)
            ax.imshow(image)
            ax.axis('off')
        break
    plt.show()

