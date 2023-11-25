import os

import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch as th
import pickle
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass

from diffusers import DDPMScheduler

# ========================
# @dataclass
# class config():
#     num_train_timesteps = 4000
#     beta_schedule = "squaredcos_cap_v2"
#
#
# # 设定噪声调度器
# noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps, beta_schedule=config.beta_schedule)
#
# x_start = pickle.load(open('./data/val/img2/file1000033_16.pt', 'rb'))
# # print(x_start['img'].shape)
#
# plt.figure(figsize=(3.2, 3.2))
# ax = plt.subplot()
# ax.set_xticks([])
# ax.set_yticks([])
# plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
# plt.margins(0, 0)
#
# plt.imshow(np.abs(x_start['img']), cmap='gray')
# plt.show()
#
# x_start_img = torch.tensor(x_start['img'])
# noise = th.randn_like(x_start_img)
# timestep = torch.tensor(1)
# noise_image = noise_scheduler.add_noise(x_start_img, noise, timestep)
#
# plt.figure(figsize=(3.2, 3.2))
# ax = plt.subplot()
# ax.set_xticks([])
# ax.set_yticks([])
# plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
# plt.margins(0, 0)
#
# plt.imshow(np.abs(noise_image), cmap='gray')
# plt.show()


# ========================
# import torch
# from torchvision.utils import make_grid, save_image
# from torchvision import transforms
# from PIL import Image
#
# betas = torch.linspace(0.02, 1e-4, 1000).double()
# alphas = 1. - betas
# alphas_bar = torch.cumprod(alphas, dim=0)
# sqrt_alphas_bar = torch.sqrt(alphas_bar)
# sqrt_m1_alphas_bar = torch.sqrt(1 - alphas_bar)
#
# img = Image.open('car.png')  # 读取图片
# trans = transforms.Compose([
#     transforms.Resize(224),
#     transforms.ToTensor()  # 转换为tensor
# ])
# x_0 = trans(img)
# img_list = [x_0]
# noise = torch.randn_like(x_0)
# for i in range(15):
#     x_t = sqrt_alphas_bar[i] * x_0 + sqrt_m1_alphas_bar[i] * noise
#     img_list.append(x_t)
# all_img = torch.stack(img_list, dim=0)
# all_img = make_grid(all_img)
# save_image(all_img, 'car_noise.png')


# ========================
import torch
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from PIL import Image

betas = torch.linspace(0.02, 1e-4, 4000).double()
alphas = 1. - betas
alphas_bar = torch.cumprod(alphas, dim=0)
sqrt_alphas_bar = torch.sqrt(alphas_bar)
sqrt_m1_alphas_bar = torch.sqrt(1 - alphas_bar)

# img = Image.open('a.png').convert('L')
# img = Image.open('b.png').convert('L')
img = Image.open('c.png').convert('L')
trans = transforms.Compose([
    transforms.ToTensor()
])
x_0 = trans(img)
noise = torch.randn_like(x_0)
for i in range(4000):
    # if i == 0:
    #     x_t = sqrt_alphas_bar[i] * x_0 + sqrt_m1_alphas_bar[i] * noise
    #     save_image(x_t, 'new_noise_img_a.png')
    # if i == 0:
    #     x_t = sqrt_alphas_bar[i] * x_0 + sqrt_m1_alphas_bar[i] * noise
    #     save_image(x_t, 'new_noise_img_b.png')
    if i == 0:
        x_t = sqrt_alphas_bar[i] * x_0 + sqrt_m1_alphas_bar[i] * noise
        save_image(x_t, 'new_noise_img_c.png')
