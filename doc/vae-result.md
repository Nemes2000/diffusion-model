# Results

The purpose of this document is to summarise what has been achieved using the VAE baseline model.

## Measurement details

The VAE model was measured using the basic Inception V3 model to calculate the FID and Inception Score. No hyperparameter optimization was run on the model and it was run for only 10 epochs, during which the model with the best val_loss result was saved.


| Model         | Dataset    | Loss   | FID    | Inception mean | Inception std |
|---------------|------------|--------|--------|----------------|---------------|
| VAE-baseline  | Flowers102 | 0.2791 | 8.8065 | 3.4239         | 0.2592        |
| VAE-baseline  | CelebA     |        |        |                |               |


### Results on Flowers102 with VAE-baseline
```json
{
    "inception_mean": 3.423909902572632,
    "inception_std": 0.25922316312789917,
    "fid": 8.806549072265625,
    "loss": 0.2791609764099121
}
```

### Results on Celeba with VAE-baseline
```json

```

### Generated images with VAE-baseline on Flowers102
We generated 25 images using random vectors from the latent space, shown in the following figure. You can see that the images are quite blurred, with some flower shapes visible in a few of the images. The images are quite faint. The quality of the images may be due to not running the net long enough to learn the distribution of the images well.

![vae-flowers](https://github.com/Nemes2000/diffusion-model/blob/main/images/vae_flowers.png)

