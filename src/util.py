from torchvision import transforms
import torch

def up_scale_images(
        images: torch.Tensor, 
        inverse_transform: transforms,
        target_size = (299, 299), 
        interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC
    ):
    up_scale_transform = transforms.Compose([
        transforms.Resize(target_size, interpolation, antialias=True),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0)
    ])

    upscaled_images = []

    for image in images:
        upscaled_images.append(up_scale_transform(inverse_transform(image)))

    return torch.stack(upscaled_images).to(torch.uint8)
