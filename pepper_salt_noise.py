import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from student_image_reconstruction_function import student_image_editing_sample
from teacher_image_reconstraction_function import teacher_image_editing_sample
'''
device = torch.device('cpu')
# 加载FashionMNIST数据集
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                             download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=False, num_workers=2)
'''

# 定义椒盐噪声函数
def add_salt_and_pepper_noise(images, noise_level=0.15):
    """
    给输入图像添加椒盐噪声

    参数:
        images (torch.Tensor): 输入图像数据
        noise_level (float): 噪声水平，取值范围[0, 1]，默认为0.05

    返回:
        torch.Tensor: 添加噪声后的图像数据
    """
    # 复制输入图像数据
    images_noisy = images.clone()

    # 获取图像数据形状
    n_samples, n_channels, height, width = images.shape

    # 计算要添加噪声的像素数目
    n_noisy_pixels = int(n_samples * n_channels * height * width * noise_level)

    # 在随机位置添加椒盐噪声
    for i in range(n_noisy_pixels):
        # 随机选择样本、通道、高度和宽度索引
        sample_idx = np.random.randint(0, n_samples)
        channel_idx = np.random.randint(0, n_channels)
        height_idx = np.random.randint(0, height)
        width_idx = np.random.randint(0, width)

        # 随机选择椒盐噪声类型
        #salt_or_pepper = np.random.randint(0, 2)
        salt_or_pepper = np.random.choice([-1, 1])
        # 添加椒盐噪声
        if salt_or_pepper == 1:
            images_noisy[sample_idx, channel_idx, height_idx, width_idx] = 1  # 椒噪声
        else:
            images_noisy[sample_idx, channel_idx, height_idx, width_idx] = -1  # 盐噪声

    return images_noisy

'''
# 获取一个batch的FashionMNIST数据
dataiter = iter(trainloader)
images, labels = next(dataiter)

# 添加椒盐噪声
noise_level = 0.1  # 设置噪声水平
noisy_images = add_salt_and_pepper_noise(images, noise_level).to(device)

# 将图像数据从Tensor转换为NumPy数组
'''
'''
Gaussion_noise = Gaussion_noise(betas=(1e-4, 0.02), n_T=500, device=device)
Gaussion_noise.to(device)
X_Gaussion = Gaussion_noise(noisy_images)
'''
'''
x_denoising = teacher_image_editing_sample(images, labels)
x_denoising = student_image_editing_sample(images, labels)

img_grid = torchvision.utils.make_grid(noisy_images, nrow=8)  # 将图像数据转换成图像网格
img_grid = img_grid / 2 + 0.5  # 反归一化
np_img_grid = img_grid.numpy()  # 将图像数据转换成NumPy数组
plt.imshow(np.transpose(np_img_grid, (1, 2, 0)))  # 显示图像
plt.axis('off')  # 关闭坐标轴
plt.show()  # 展示图像
'''