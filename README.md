# DeepLearning - BMEVITMMA19 2024/25/1

The repository is used to implement DDPM and document the work for the DeepLearning project homework. The aim of the project is to build a diffusion model that can generate realistic or very close images using the CelebA and Flowers102 datasets. The ultimate goal is to build a mini AI service that can generate images using our model.

## Project homework - Image generation with diffusion models

| Group     | Details                                      |
| --------- | -------------------------------------------- |
| Team name | RelAITeam                                    |
| Members   | Attila Nemes (B6RYIK), Csaba Potyok (OZNVQ4) |

## Project description

**Task description**: Implement and train unconditional diffusion models, such as DDPM (Denoising Diffusion Probabilistic Model) or DDIM (Denoising Diffusion Implicit Model) for generating realistic images. Evaluate the capabilities of the models on two different datasets, such as CelebA and Flowers102.

**Our approach**: We plan to implement a DDPM model to try to generate as realistic images as possible. We plan to measure the model on the CelebA and Flowers102 datasets. We plan to quantize the model (and measure the quality changes) and use [Gradio](https://www.gradio.app/) to build an AI service from it.

## Related works

### Related GitHub repositories:

- https://huggingface.co/blog/annotated-diffusion
- https://github.com/huggingface/diffusers

### Related papers:

- https://arxiv.org/abs/2006.11239
- https://arxiv.org/abs/2010.02502

## Files

Our .py and .ipynb files are located in the **src** folder:

- config.py: It will contains all parameter that we want to optimize at hyperopt (not all parameters currently included).
- main.py: To load data and start training.
- preprocess_data_source.py: The CelebA dataset is sometimes unavailable (restriction of Drive), so we downloaded it as ZIP and preprocess with this script (split into train/val/test based on [list_eval_partition.txt](https://drive.google.com/drive/folders/0B7EVK8r0v71pdjI3dmwtNm5jRkE?resourcekey=0-TD_RXHhlG6LPvwHReuw6IA)
- data_module:
  - celeba.py: Define datamodule to load CelebA images
  - celeba_dataset.py: custom dataset to handle CelebA images that were downloaded as ZIP from [preprocessed version](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ)
  - flowers.py: Define datamodule to load Flowers102 images, use torchvision dataset
- data_visualization:
  - plot_hist.py: Plot a histogram (RGB values) for images from dataloader (from first batch, first 5 elements)
  - plot_image.py: Plot images from dataloader (from first batch, first 10 elements)
- first_step.ipynb: we tried out the dataset loading and the visualization

Other:

- requirements.txt: Contains all Python libraries that we want to use.
- Dockerfile: Contains a basic pytorch-cuda container description.

## Run

If you don't want to use GPU when you run it in container, run this command from the root directory:

```bash
  docker compose up
```
