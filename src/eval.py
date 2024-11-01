import pytorch_lightning as pl
from baseline_model.vae import BaseLineImageGenerationVAE
import argparse
from data_module.flowers import Flowers102DataModule
from data_module.celeba import CelebADataModule
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from config import Config
from tqdm import tqdm
import json

def calculate_is_and_fid(dataloader: DataLoader, feature=64):
    fid = FrechetInceptionDistance(feature=feature)
    inception = InceptionScore()

    up_scale_transform = transforms.Compose([
        transforms.Resize((299, 299), transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0)
    ])

    for batch, _ in tqdm(dataloader):
        images = []
        gen_images = []
        for image in batch:
            images.append(up_scale_transform(data_module.reverse_transform(image)))

        with torch.no_grad():
            recon_batch, _, _ = model(batch)
            recon_batch = recon_batch.reshape((-1, 3,) + Config.image_target_size)
            for image in recon_batch:
                gen_images.append(up_scale_transform(data_module.reverse_transform(image)))

        images = torch.stack(images).to(torch.uint8)
        gen_images = torch.stack(gen_images).to(torch.uint8)
        fid.update(images, real=True)
        fid.update(gen_images, real=False)
        inception.update(gen_images)
    
    return inception.compute(), fid.compute()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-path', type=str)
    parser.add_argument('-model', type=str)
    parser.add_argument('-dataset', type=str, choices=['flowers', 'celeba'], default='flowers')
    parser.add_argument('-stat-file', type=str, default='stat.json')

    args = parser.parse_args()

    model_path = f'{args.path}/{args.model}.ckpt'
    model = BaseLineImageGenerationVAE.load_from_checkpoint(model_path)

    trainer = pl.Trainer(logger=False)

    if args.dataset == 'flowers':
        data_module = Flowers102DataModule()
    else:
        data_module = CelebADataModule()

    data_module.prepare_data()
    data_module.setup()  

    test_result = trainer.test(model, datamodule=data_module)
    inception_score, fid_score = calculate_is_and_fid(data_module.test_dataloader())

    with open(args.stat_file, 'w') as output:
        json.dump({
            'inception_mean': inception_score[0].item(),
            'inception_std': inception_score[1].item(),
            'fid': fid_score.item(),
            'loss': test_result[0]['test_loss']
        }, output, indent=4)

