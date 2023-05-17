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


class midstep_denoising(nn.Module):
    """
    反向扩散过程和``Diffusion.Diffusion.py``中的``GaussianDiffusionSampler``绝大部分一样，
    所以在此只说明不一样的点
    """
    def __init__(self, model, beta_1, beta_T, T, w=2.0):
        super().__init__()

        self.model = model
        self.T = T
        # In the classifier free guidence paper, w is the key to control the gudience.
        # w = 0 and with label = 0 means no guidence.
        # w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
        # 不同点1: 在初始化时需要输入一个权重系数``w``, 用来控制条件的强弱程度
        self.w = w

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t, labels):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var.to(x_t.device)
        var = extract(var, t, x_t.shape)

        # 不同点2: 模型推理时需要计算有条件和无条件(随机噪声)情况下模型的输出，
        # 将两次输出的结果用权重``self.w``进行合并得到最终输出
        eps = self.model(x_t, t, labels)
        #eps.to(x_t.device)
        nonEps = self.model(x_t, t, torch.zeros_like(labels).to(labels.device))
        #nonEps.to(x_t.device)
        # 参考原文公式(6)
        eps = (1. + self.w) * eps - self.w * nonEps
        eps.to(x_t.device)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def forward(self, x_T, labels, n_upto): #labels注意要加1和真实的对应
        """
        Algorithm 2.
        """
        #device = x_T.device
        #print('device=',device)
        x_t = x_T
        #x_t.to(x_T.device)
        for time_step in reversed(range(n_upto+1)):
            print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            #t.to(x_T.device)
            # 除了输入多一个``labels``其他都和普通Diffusion Model一样
            mean, var = self.p_mean_variance(x_t=x_t, t=t, labels=labels)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        x_0 = torch.clip(x_0, -1, 1)
        x_0 = x_0 * 0.5 + 0.5
        return x_0
