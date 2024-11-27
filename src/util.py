from torchvision import transforms
import torch
from inspect import isfunction
from torch import nn
from einops.layers.torch import Rearrange

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

    num_images = len(images)
    image_shape = (3, 299, 299)

    upscaled_images = torch.empty((num_images, *image_shape), dtype=torch.uint8).to(torch.device('cuda'))

    for i, image in enumerate(images):
        upscaled_images[i] = up_scale_transform(inverse_transform(image))
        
    return upscaled_images.to(torch.uint8)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )

def downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )
