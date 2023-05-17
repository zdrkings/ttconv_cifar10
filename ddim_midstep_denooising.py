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


class midstep_denoising_ddim(nn.Module):
    """
    反向扩散过程和``Diffusion.Diffusion.py``中的``GaussianDiffusionSampler``绝大部分一样，
    所以在此只说明不一样的点
    """
    def __init__(self, model, beta_1, beta_T, T, w, stride, eta):
        super().__init__()

        self.model = model
        self.T = T
        # In the classifier free guidence paper, w is the key to control the gudience.
        # w = 0 and with label = 0 means no guidence.
        # w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
        # 不同点1: 在初始化时需要输入一个权重系数``w``, 用来控制条件的强弱程度
        self.w = w
        self.stride = stride
        self.eta = eta
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        alphas_bar_ = alphas_bar[::stride]
        alphas_bar_ = torch.cat([alphas_bar_, alphas_bar[-1:]], dim=0)
        self.T_idx = len(alphas_bar_)
        self.T_seq =  torch.arange(0, self.T, stride)
        self.T_seq = torch.cat((self.T_seq, torch.tensor([self.T-1])))


        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        alphas_bar_prev_ =  alphas_bar_prev[::stride]
        alphas_bar_prev_ = torch.cat([alphas_bar_prev_, alphas_bar_prev[-1:]], dim=0)

        #self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff1', torch.sqrt(alphas_bar_prev_ / alphas_bar_))
        #self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('coeff2', 1-alphas_bar_prev_)
        self.register_buffer('coeff3', torch.sqrt(((1. - alphas_bar_) * alphas_bar_prev_)/alphas_bar_))
        self.register_buffer('posterior_var_', (self.eta * (1-(alphas_bar_/alphas_bar_prev_)) * (1. - alphas_bar_prev_) )/ (1. - alphas_bar_))
        self.register_buffer('coeff4', torch.sqrt(self.coeff2-self.posterior_var_)-self.coeff3)
    def predict_xt_prev_mean_from_eps(self, x_t, t_, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t_, x_t.shape) * x_t +
            extract(self.coeff4, t_, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t,t_, labels):
        # below: only log_variance is used in the KL computations

        #var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        #var = extract(var, t, x_t.shape)
        var =  extract(self.posterior_var_, t_, x_t.shape)
        # 不同点2: 模型推理时需要计算有条件和无条件(随机噪声)情况下模型的输出，
        # 将两次输出的结果用权重``self.w``进行合并得到最终输出
        eps = self.model(x_t, t, labels)
        nonEps = self.model(x_t, t, torch.zeros_like(labels).to(labels.device))
        # 参考原文公式(6)
        eps = (1. + self.w) * eps - self.w * nonEps
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t=x_t, t_=t_,eps=eps)
        return xt_prev_mean, var

    def forward(self, x_T, labels,n_upto):
        """
        Algorithm 2.
        """
        #搞清楚t_和t关系
        x_t = x_T
        t_ = self.T_idx-1
        print(self.T_seq)
        #for time_step in reversed(range(self.T)):
        for idx in range(self.T_seq.shape[0] - 1, -1, -1):
            time_step = self.T_seq[t_]

            if time_step <=n_upto:
             print('time_step=', time_step)
             print('t_=', t_)
             t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
             t_ss = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * t_
             # 除了输入多一个``labels``其他都和普通Diffusion Model一样
             mean, var = self.p_mean_variance(x_t=x_t, t=t, t_=t_ss, labels=labels)
             if time_step > 0:
                noise = torch.randn_like(x_t)
             else:
                noise = 0
             x_t = mean + torch.sqrt(var) * noise
             assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
            t_ = t_-1
        x_0 = x_t
        x_0 = torch.clip(x_0, -1, 1)
        x_0 = x_0 * 0.5 + 0.5
        return x_0

