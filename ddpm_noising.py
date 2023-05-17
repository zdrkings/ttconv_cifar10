import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
torch.manual_seed(124)

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class Append_Gaussion_noise(nn.Module):
    """
    前向加噪过程和``Diffusion.Diffusion.py``中的``GaussianDiffusionTrainer``几乎完全一样
    不同点在于模型输入，除了需要输入``x_t``, ``t``, 还要输入条件``labels``
    """
    def __init__(self, beta_1, beta_T, T):
        super().__init__()

        
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, n_upto):
        """
        Algorithm 1.
        """
        #t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        t = n_upto * torch.ones(x_0.shape[0], dtype=torch.int64).to(x_0.device)
        noise = torch.randn_like(x_0).to(x_0.device)
        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
              extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise

        return x_t
