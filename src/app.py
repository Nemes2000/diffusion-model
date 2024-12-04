import gradio as gr
import argparse
from model.ddpm_v2.diffusion import DiffusionModel
from model.scheduler.function import LinearScheduleFn
from gradio_app.inference_module import InferenceDDPMModule
import torch
from config import Config
from torchvision import transforms
import numpy as np
from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu') 
model = None
transform = None

def generate_image():
    global model
    global device
    global transform
    imgs = torch.randn((1, 3) + Config.image_target_size).to(device)
    for _, t_i in tqdm(enumerate(reversed(range(model.diffusion_model.timesteps)))):
        t = torch.full((1,), t_i, dtype=torch.long, device=device)
        diff_imgs = model.diffusion_model.backward(x=imgs, t=t, model=model.eval().to(device))
        if torch.isnan(diff_imgs).any(): break
        imgs = diff_imgs
        yield(transform(imgs[0]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-path', type=str)
    parser.add_argument('-model', type=str)

    args = parser.parse_args()

    model_path = f'{args.path}/{args.model}.ckpt'

    diffusion_model = DiffusionModel(function=LinearScheduleFn(beta_start=0.0001, beta_end=0.02))
    model = InferenceDDPMModule.load_from_checkpoint(model_path, strict=False)
    model.diffusion_model = diffusion_model
    transform = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),
            transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ])

    iface = gr.Interface(fn=generate_image, inputs=None, outputs=gr.Image(width=64, height=64)).launch()