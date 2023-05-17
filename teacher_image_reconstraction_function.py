
import os
import random

import numpy as np

import torch
import torchvision.utils as tvu

from Model_teacher import teacher_UNet
from ddpm_midstep_denoising import midstep_denoising
from ddpm_noising import Append_Gaussion_noise

device = torch.device('cuda:0')
store_path_teacher = 'ckpt_69_teacher.pt'
teacher = teacher_UNet(T=500, num_labels=10, ch=128, ch_mult=[1, 2, 2, 2],
                     num_res_blocks=2, dropout=0.)
ckpt = torch.load(
        store_path_teacher, map_location=device)
teacher.load_state_dict(ckpt)
teacher.eval()
midstep_denoising_teacher = midstep_denoising(model=teacher, beta_1 =1e-4,beta_T=0.028, T=500,w=15.0).to(device)

'''
print("Model's state_dict:")
for param_tensor in midstep_denoising1.state_dict():
    print(param_tensor, "\t", midstep_denoising1.state_dict()[param_tensor].size())
print('model_loaded')
'''

def teacher_image_editing_sample(img, labels, levels):
    assert isinstance(img, torch.Tensor)
    with torch.no_grad():

        assert img.ndim == 4, img.ndim
        x0 = img
        #xs = []
        #for it in range(2): #这个参数也可以进行修改

        total_noise_levels = levels #可以修改加入噪声的步数
        Append_Gaussion_noise1 = Append_Gaussion_noise(beta_1= 1e-4, beta_T=0.028, T=500).to(device) #先实例化一个
        x_t_upto= Append_Gaussion_noise1(x0, total_noise_levels)
        x_t_upto.to(device)
        #midstep_denoising1 = midstep_denoising(nn_model=Teacher_ContextUnet(in_channels=1, n_feat=256, n_classes=10), betas=(1e-4, 0.02), n_T=500,
        #        device=device, drop_prob=0.1)
        labels = labels +1 #与之前的标签进行兼容
        labels.to(device)
        x_reconstraction = midstep_denoising_teacher(x_t_upto, labels, total_noise_levels)



        x0 = x_reconstraction
            #xs.append(x0)

        #return torch.cat(xs, dim=0)
        return x0
