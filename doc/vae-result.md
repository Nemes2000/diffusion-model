# Results

The purpose of this document is to summarise what has been achieved using the VAE baseline model.

## Measurement details

The VAE model was measured using the basic Inception V3 model to calculate the FID and Inception Score. No hyperparameter optimization was run on the model and it was run for only 10 epochs, during which the model with the best val_loss result was saved.


| Model         | Dataset    | Loss   | FID    | Inception mean | Inception std |
|---------------|------------|--------|--------|----------------|---------------|
| VAE-baseline  | Flowers102 | 0.2779 | 8.9862 | 3.2937         | 0.1748        |
| VAE-baseline  | CelebA     | 0.2353 | 4.5525 | 3.0732         | 0.0394        |


### Results on Flowers102 with VAE-baseline
```json
{
    "inception_mean": 3.29376482963562,
    "inception_std": 0.17487818002700806,
    "fid": 8.986263275146484,
    "loss": 0.27796974778175354
}
```

### Results on Celeba with VAE-baseline
```json
{
    "inception_mean": 3.073242664337158,
    "inception_std": 0.03946512192487717,
    "fid": 4.552577495574951,
    "loss": 0.23535996675491333
}
```

### Generated images with VAE-baseline on Flowers102
We generated 25 images using random vectors from the latent space, shown in the following figure. You can see that the images are quite blurred, with some flower shapes visible in a few of the images. The images are quite faint. The quality of the images may be due to not running the net long enough to learn the distribution of the images well.

![vae-flowers](https://github.com/Nemes2000/diffusion-model/blob/main/images/vae_flowers.png)

### Generated images with VAE-baseline on CelebA
Also 25 images were generated from the latent space by selecting random vectors. The result was better than for the flowers, but this can be explained by the fact that we had many more images to train on. Here again, the images are mostly blurred, but we can make out facial shapes. The result is more spectacular than with flowers, but far from perfect, partly due to the small number of epochs.

![vae-celeba](https://github.com/Nemes2000/diffusion-model/blob/main/images/vae_celeba.png)
