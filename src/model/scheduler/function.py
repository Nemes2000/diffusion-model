import torch 

class BaseScheduleFn:
    def __call__(self, timesteps: int = 1000):
        return torch.zeros((timesteps,))
    
class LinearScheduleFn(BaseScheduleFn):
    def __init__(self, beta_start: float, beta_end: float):
        self.beta_start = beta_start
        self.beta_end = beta_end

    def __call__(self, timesteps = 1000):
        return torch.linspace(self.beta_start, self.beta_end, timesteps)
    
class CosineBetaScheduleFn(BaseScheduleFn):
    def __init__(self, s: float = 0.008):
        self.s = s

    def __call__(self, timesteps = 1000):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + self.s) / (1 + self.s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
class QuadraticBetaSheduleFn(LinearScheduleFn):
    def __init__(self, beta_start: float, beta_end: float):
        super().__init__(beta_start, beta_end)

    def __call__(self, timesteps=1000):
        return torch.linspace(self.beta_start**0.5, self.beta_end**0.5, timesteps) ** 2
    
class SigmoidBetaScheduleFn(LinearScheduleFn):
    def __init__(self, beta_start: float, beta_end: float):
        super().__init__(beta_start, beta_end)

    def __call__(self, timesteps=1000):
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (self.beta_end - self.beta_start) + self.beta_start