class Config():
  # image default crop size in celeba: 218*178
  image_target_size = (64,64)
  batch_size = 64
  num_of_workers = 0
  learning_rate = 1e-3
  weight_decay = 1e-4
  num_of_epochs = 50
  latent_dims = 256