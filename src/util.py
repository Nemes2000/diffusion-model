from torchvision import transforms
import torch

def up_scale_images(
        images: torch.Tensor, 
        inverse_transform: transforms,
        target_size = (299, 299), 
        interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC
    ):
    """Upscale images to the given size with the given transformation.
    """
    up_scale_transform = transforms.Compose([
        transforms.Resize(target_size, interpolation, antialias=True),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0)
    ])

    num_images = len(images)
    image_shape = (3, 299, 299)

    upscaled_images = torch.empty((num_images, *image_shape), dtype=torch.uint8).to(torch.device('cuda'))

    for i, image in enumerate(images):
        upscaled_images[i] = up_scale_transform(inverse_transform(image))
        
    return upscaled_images.to(torch.uint8)
