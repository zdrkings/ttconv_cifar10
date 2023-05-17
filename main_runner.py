import os

import torch
import torchvision
from torchvision.transforms import transforms
from autoattack import AutoAttack
from torchvision.utils import save_image
import torch.nn.functional as F

import ptytorch_ssim
from ddim_push import ddim_denoising
from gaussion_noise import add_gaussion_noise
from pepper_salt_noise import add_salt_and_pepper_noise
from student_image_reconstruction_function import student_image_editing_sample
#from student_image_reconstruction_function_ddim import student_image_editing_sample_ddim
from teacher_image_reconstraction_function import teacher_image_editing_sample
from teacher_image_reconstruction_function_ddim import teacher_image_editing_sample_ddim

#model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

'''
transform = transforms.Compose(
    [transforms.ToTensor()])
'''
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024,
                                          shuffle=False, num_workers=2)
dataiter = iter(trainloader)
images, labels = next(dataiter)

images = images.cuda()
labels = labels.cuda()

save_image(0.5*images+0.5, os.path.join(
            "./general_classifer/",  "images512.bmp"), nrow=8)
resnet_56_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
print('resnet56 loaded')
# 将模型设置为评估模式
resnet_56_model.to('cuda:0')
resnet_56_model.eval()

#device = torch.device('cpu')
#device = torch.device('cuda:0')

adversary = AutoAttack(resnet_56_model, norm='L2', eps=0.5, version='standard')
with torch.no_grad():
    #noisy_images = adversary.run_standard_evaluation(images, labels, bs=64)
    #noisy_images_gs = add_gaussion_noise(images, mean=0, std=0.2)
    #noisy_images_sp = add_salt_and_pepper_noise(images)
    noisy_images_at = adversary.run_standard_evaluation(images, labels, bs=64)
    #gs_images = adversary.run_standard_evaluation(noisy_images, labels, bs=64)
    #noisy_images_gs.cuda()
    #noisy_images_sp.cuda()
    noisy_images_at.cuda()
    save_image(0.5 * noisy_images_at + 0.5, os.path.join(
        "./teacher_adjust/", "noisyimage_at_ranks_4_4_1.bmp"), nrow=8)
    #print('device=', noisy_images_gs.device)
    #print('device=', noisy_images_sp.device)
    print('device=', noisy_images_at.device)
    #reimage_gs = teacher_image_editing_sample(noisy_images_gs, labels, 120)
    #reimage_sp = teacher_image_editing_sample(noisy_images_sp, labels,124)
    reimage_at = student_image_editing_sample(noisy_images_at, labels, 124)
    reimage_at_1 = teacher_image_editing_sample(noisy_images_at, labels, 124)
    #reimage_gs = ddim_denoising(noisy_images_gs)
    #reimage_sp = ddim_denoising(noisy_images_sp)
    #reimage_at = ddim_denoising(noisy_images_at)
    save_image(reimage_at, os.path.join(
        "./student_adjust/", "rank_4_4_127stp_at_1.bmp"), nrow=8)
    save_image(reimage_at_1, os.path.join(
        "./teacher_adjust/", "cnn_at_1.bmp"), nrow=8)
    #print('already saved')
    # accuracy = calculate_accuracy(reimage, labels)
    # print('accuracy=', accuracy)
    #print(ptytorch_ssim.ssim(reimage_gs, 0.5*images+0.5))
    
    adversary1 = AutoAttack(resnet_56_model, norm='L2', eps=0, version='apgd-ce')
    #ans_images_gs = adversary1.run_standard_evaluation(reimage_gs, labels, bs=64)
    #ans_images_sp = adversary1.run_standard_evaluation(reimage_sp, labels, bs=64)
    ans_images_at = adversary1.run_standard_evaluation(reimage_at, labels, bs=64)