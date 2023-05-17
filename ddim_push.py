'''
import torch
from diffusers import DDIMPipeline

model_id = "google/ddpm-cifar10-32"

# load model and scheduler
ddim = DDIMPipeline.from_pretrained(model_id)

# run pipeline in inference (sample random noise and denoise)
image = ddim(num_inference_steps=100, batch_size=10).images[1]
print
# save image
image.save("ddim_generated_image.bmp")
'''
import os

from diffusers import DDIMScheduler, UNet2DModel
from PIL import Image
import torch
import numpy as np
import torchvision
from torchvision.utils import save_image

from ddpm_noising import Append_Gaussion_noise


def ddim_denoising(noisy_images):
  #Append_Gaussion_noise1 = Append_Gaussion_noise(beta_1=1e-4, beta_T=0.028, T=500).to("cuda")  # 先实例化一个
  #noisy_images= Append_Gaussion_noise1(noisy_images, steps)
  #save_image(noisy_images, os.path.join(
  #    "./student_adjust/", "ddim_noising.bmp"), nrow=8)

  scheduler = DDIMScheduler.from_pretrained("google/ddpm-cifar10-32")
  model = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to("cuda")
  scheduler.set_timesteps(150)

  sample_size = model.config.sample_size
  print('sample=',sample_size)
  #noise = torch.randn((1024, 3, sample_size, sample_size)).to("cuda")
  #input =noise
  input = noisy_images

  for t in scheduler.timesteps:
     print('t=',t)
     with torch.no_grad():
        noisy_residual = model(input, t).sample
        prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
        input = prev_noisy_sample

  image = (input / 2 + 0.5).clamp(0, 1)
  print(image.shape)
  return image
#save_image(image, os.path.join(
#        "./student_adjust/", "student_ddimscheler.bmp"),nrow=8)
'''
image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
image = Image.fromarray((image * 255).round().astype("uint8"))
image.save("ddim_generated_image.bmp")
'''
