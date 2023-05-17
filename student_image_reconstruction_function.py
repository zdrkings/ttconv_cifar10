
import os
import random

import numpy as np

import torch
import torchvision.utils as tvu

from Model_student import student_UNet
from ddpm_midstep_denoising import midstep_denoising
from ddpm_noising import Append_Gaussion_noise

device = torch.device('cuda:0')
store_path_student = 'ckpt_ranks_4_4_127_.pt'
student = student_UNet(T=500, num_labels=10, ch=128, ch_mult=[1, 2, 2, 2],
                     num_res_blocks=2, dropout=0.15)
ckpt = torch.load(
        store_path_student, map_location=device)
student.load_state_dict(ckpt)
student.eval()
midstep_denoising_student = midstep_denoising(model=student, beta_1 =1e-4,beta_T=0.028, T=500,w=15.0)
midstep_denoising_student.to(device)
print('test end')
'''
print("Model's state_dict:")
for param_tensor in midstep_denoising1.state_dict():
    print(param_tensor, "\t", midstep_denoising1.state_dict()[param_tensor].size())
print('model_loaded')
'''

def student_image_editing_sample(img, labels, level):
    assert isinstance(img, torch.Tensor)
    with torch.no_grad():

        assert img.ndim == 4, img.ndim
        #img.to(device)
        #print('imgdevice=',img.device)
        x0 = img
        #x0 = x0.to(device)
        #print('x0device=', x0.device)
        #xs = []
        #for it in range(2): #这个参数也可以进行修改

        total_noise_levels = level #可以修改加入噪声的步数
        Append_Gaussion_noise1 = Append_Gaussion_noise(beta_1= 1e-4, beta_T=0.028, T=500).to(device) #先实例化一个
        x_t_upto= Append_Gaussion_noise1(x0, total_noise_levels)
        #x_t_upto =x_t_upto.to(device)
        #midstep_denoising1 = midstep_denoising(nn_model=student_ContextUnet(in_channels=1, n_feat=256, n_classes=10), betas=(1e-4, 0.02), n_T=500,
        #        device=device, drop_prob=0.1)
        labels = labels +1 #与之前的标签进行兼容
        #labels = labels.to(device)
        x_reconstraction = midstep_denoising_student(x_t_upto, labels, total_noise_levels)



        x0 = x_reconstraction
            #xs.append(x0)

        #return torch.cat(xs, dim=0)
        return x0
