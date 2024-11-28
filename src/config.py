import torch

class Config():
  # image default crop size in celeba: 218*178
  image_target_size = (64,64)
  batch_size = 64
  num_of_workers = 8
  learning_rate = 1e-3
  weight_decay = 1e-4
  num_of_epochs = 50
  latent_dims = 256
  optimizer = torch.optim.AdamW
  time_embedding_dims = 256
  time_steps = 300
  model_name = "diffusion-model"
  optimizer_map = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
    'adamW': torch.optim.AdamW
  }
  optimalization_step=10
  image_size= 64
  channels = 3