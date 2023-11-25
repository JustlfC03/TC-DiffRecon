import os

import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch as th
import pickle
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass

from diffusers import DDPMScheduler


def do_mask(img, mask):
    data = np.stack([np.real(img), np.imag(img)]).astype(np.float32)
    max_val = abs(data[:2]).max()
    data[:2] /= max_val
    # max_val = abs(data[2:4]).max()
    # data[2:4] /= max_val
    # regularizing over max value ensures this model works over different preprocessing schemes;
    # to not use the gt max value, selecting an appropriate averaged max value from training set leads to
    # similar performance, e.g.
    # data /= 7.21 (average max value); in general max_value is at DC and should be accessible.
    data1 = data[0] + data[1] * 1j
    kspace1 = np.fft.fft2(data1)

    # 用于生成图片b.png
    # out1 = mask * kspace1
    # return out1
    return kspace1


x_start = pickle.load(open('./data/val/img2/file1000033_16.pt', 'rb'))
# print(x_start['img'].shape)

mask = th.load(open('./mask_8.pt', 'rb')).reshape(320, 320)
# mask = pickle.load(open('./mask_4.pt', 'rb')).view(320, 320)
# print(mask.shape)
out_img = do_mask(x_start['img'], torch.tensor(mask))

plt.figure(figsize=(3.2, 3.2))
ax = plt.subplot()
ax.set_xticks([])
ax.set_yticks([])
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)

plt.imshow(np.log(10 * abs(out_img) + 1), cmap='gray')
# plt.savefig("b.png")
plt.savefig("c.png")
plt.show()
