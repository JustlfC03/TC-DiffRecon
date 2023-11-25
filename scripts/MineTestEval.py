import pickle

from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from PIL import Image
import numpy as np
import cv2
import torch

import matplotlib.pyplot as plt

img1 = cv2.imread('D:\\BaiduNetdiskDownload\\3\\a.png', 0)
# img2 = cv2.imread('D:\\Research_Topic\\医学超分\\DiffuseRecon-main\\scripts\\test\\file1000031_16.png')
img2 = np.array(Image.open('D:\\Research_Topic\\医学超分\\DiffuseRecon-main\\scripts\\test\\file1000033_16.png'))

if __name__ == "__main__":
    print(img1.shape)
    print(img2.shape)

    print(psnr(img1, img2))

    # If the input is a multichannel (color) image, set multichannel=True.
    print(ssim(img1, img2, multichannel=False))
