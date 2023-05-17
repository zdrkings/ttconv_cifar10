import torch


def add_gaussion_noise(images, mean, std):
    noise = torch.normal(mean=mean, std=std, size=images.shape)
    noise = noise.cuda()
    noisy_image = images + noise

    return noisy_image