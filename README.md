# DeepLearning - BMEVITMMA19 2024/25/1

The repository is used to implement DDPM and document the work for the DeepLearning project homework. The aim of the project is to build a diffusion model that can generate realistic or very close images using the CelebA and Flowers102 datasets. The ultimate goal is to build a mini AI service that can generate images using our model.

## Project homework - Image generation with diffusion models

| Group     | Details                                      |
| --------- | -------------------------------------------- |
| Team name | RelAITeam                                    |
| Members   | Attila Nemes (B6RYIK), Csaba Potyok (OZNVQ4) |

## Project description

**Task description**: Implement and train unconditional diffusion models, such as DDPM (Denoising Diffusion Probabilistic Model) or DDIM (Denoising Diffusion Implicit Model) for generating realistic images. Evaluate the capabilities of the models on two different datasets, such as CelebA and Flowers102.

> [!NOTE]  
> Since the original CelebA dataset is often not accessible online, we used the CelebA dataset from the huggingface site ([nielsr/CelebA-faces](https://huggingface.co/datasets/nielsr/CelebA-faces)). There is no difference in the two datasets so it was considered more convenient for the project to have a permanently accessible dataset. But in this dataset we don't have by default the train, validation and test data labels so we created the three dataset by hand.

**Our approach**: We plan to implement a DDPM model to try to generate as realistic images as possible. We plan to measure the model on the CelebA and Flowers102 datasets. We plan to use [Gradio](https://www.gradio.app/) to build an AI service from it.

## Related works

### Related GitHub repositories:

- https://huggingface.co/blog/annotated-diffusion
- https://github.com/huggingface/diffusers

### Related papers:

- https://arxiv.org/abs/2006.11239
- https://arxiv.org/abs/2010.02502

## Files

Our .py and .ipynb files are located in the **src** folder:

- main.py: To load data and plot some generated image with a saved model
- config.py: It will contains all parameter that we want to optimize at hyperopt (not all parameters currently included).
- eval.py: Calculates a given saved model's IS and FID score on a given dataset (celeba or flowers), and save result into stat.json file
- train.py: Set up the model and wandb logging for training, and runs a training on given dataset (celeba or flowers)
- hyperopt.py: Set up the model and wandb for hyperparamter optimization.
- preprocess_data_source.py: The CelebA dataset is sometimes unavailable (restriction of Drive), so we downloaded it as ZIP and preprocess with this script (split into train/val/test based on [list_eval_partition.txt](https://drive.google.com/drive/folders/0B7EVK8r0v71pdjI3dmwtNm5jRkE?resourcekey=0-TD_RXHhlG6LPvwHReuw6IA))
- baseline_model:
  - vae.py: Our baseline model, which is a VAE modell
- data_module:
  - celeba_dataset.py: custom dataset to handle CelebA images that were downloaded as ZIP from [preprocessed version](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ)
  - celeba.py: Define datamodule to load CelebA images
  - flowers.py: Define datamodule to load Flowers102 images, use torchvision dataset
- data_visualization:
  - plot_hist.py: Plot histogram (RGB values) for images from dataloader (from first batch, first 5 elements)
  - plot_image.py: Contains the plot functions for plot images from dataloader (from first batch, first 10 elements), for plot images from latent dimension and for plot multiple generated images from a given image.
- gradio_app:
  - inference_module.py: Contains a LightningModule to load trained model for running in inference mode.
- model:
  - ddpm_v2:
    - block.py: Implement a down-/upsampling block for UNet in DDPM.
    - diffusion.py: Implement the diffusion backward and forward processes.
    - embedding.py: Implement a Sinusoidal Position Embeddings to use it for timestep embedding.
    - module.py: Contains a LigthningModule to handle train, validation and test pipeline.
  - scheduler:
    - function.py: Contains timestep function implementations for diffusion model.
- first_step.ipynb: we tried out the dataset loading and the visualization
- vae_baseline_flowers102.ipynb: We used it to train and test VAE on Colab.

Other:

- requirements.txt: Contains all Python libraries that we want to use.
- Dockerfile: Contains a basic pytorch-cuda container description.
- gradio.dockerfile: Contains the gradio application container description.
- doc:
  - [evaluation.md](https://github.com/Nemes2000/diffusion-model/blob/main/doc/evaluation.md): Contains approaches and expectations for the evaluation of models.
  - [vae-result.md](https://github.com/Nemes2000/diffusion-model/blob/main/doc/vae-result.md): Contains the results of the VAE baseline model.
  - [DL_documentation.pdf](https://github.com/Nemes2000/diffusion-model/blob/main/doc/DL_documentation.pdf): Contains the documentation of the project homework.
- results:
  - manual_fid_is_results.xlsx: Contains manual tested FID and IS results on Flowers102 and CelebA datasets.
  - Contains more hyperopt results from wandb.
- images:
  - Contains generated images (VAE + DDPM)

## Run

For our Docker container, you should mount volumes for the following folders:

- diffusion-model/data: It contains the dowloaded datasets (preprocessed celeba and flowers102).
- diffusion-model/model: It contains the best model after the training.
- diffusion-model/logs: It contains TensorBoard logs if we use default logging.
- diffusion-model/wandb: It contains Wandb logs if we use wandb logging with **-log-wandb** flag.
- diffusion-model/stat.json: It contains evaluation result in a JSON object (with **-stat-file** flag you can change it)

> [!NOTE]
> For using wandb for logging or hyperopt, you have to set WANDB_API_KEY value to your wandb api key.
> In docker run command use -e WANDB_API_KEY=my-api-key (replace my-api-key with the correct key)

First build the image, run this command from the root directory:

```bash
docker build -t [IMAGE_NAME] .
```

To train the model, you need to run the container in the following format from the root directory:

```bash
docker run --gpus all --rm -v ./data:/diffusion-model/data -v ./model:/diffusion-model/model -v ./logs:/diffusion-model/logs [IMAGE_NAME] python src/train.py
```

If you want to specify the training you can set this flag:

- _-model-name_: The model name.
- _-log-wandb_: Turn on wandb logging.
- _--wandb-project_: Set the name of the wandb project.
- _-epoch_: Set the number of the epoch (default: 50).
- _-dataset_: Set the dataset for training ('celeba' / 'flowers', default: 'flowers')

Example usage:

```bash
docker run --gpus all --rm -v ./data:/diffusion-model/data -v ./model:/diffusion-model/model -v ./logs:/diffusion-model/logs [IMAGE_NAME] python src/train.py -model-name 'vae-baseline' -log-wandb --wandb-project "vae-baseline" -dataset "celeba" -epoch 10
```

To eval the model, you need to run the container in the following format from the root directory:

```bash
docker run --gpus all --rm -v ./data:/diffusion-model/data -v ./model:/diffusion-model/model -v ./logs:/diffusion-model/logs [IMAGE_NAME] python src/eval.py -path "./model/vae-baseline" -model "best"
```

The evaluation requires that the model is located in the mounted folder pointed to by the **-path** flag.

If you want to specify the evaluation you can set this flag:

- _-stat-file_: Set the stat file's name to save the evaluation result. (default: 'stat.json')
- _-dataset_: Set the dataset for evaluation ('celeba' / 'flowers', default: 'flowers')

Example usage:

```bash
docker run --gpus all --rm -v ./data:/diffusion-model/data -v ./model:/diffusion-model/model -v ./logs:/diffusion-model/logs [IMAGE_NAME] python src/eval.py -path "./model/vae-baseline" -model "best" -dataset "flowers"
```

## Gradio

We've created a simple Gradio interface for our trained model. If you want to use it, you need a trained model, which you can place in any place you like. You will need to build and run gradio.dockerfile, which you will need to attach the folder containing your model as a volume.

> [!NOTE]
> This image uses the Dockerfile image used for training and testing. Therefore, it must first be built and named with the -t diffusion tag so that it can access it.

```
docker build -f gradio.dockerfile -t [IMAGE_NAME] .
```

Example for running Gradio in container:

```
docker run --gpus all --rm -p 7860:7860 -v ./model:/diffusion-model/model [IMAGE_NAME]
```
