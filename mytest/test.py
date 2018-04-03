from skimage import io, util
from skimage.measure import compare_psnr
import cv2
import numpy as np
import os.path as osp
import multiprocessing
from myutils import myimutil, my_utils

PATH_BASE = 'E:/workspace/deeplearning/keras/file/nature_img_denoise'
PATH_ORIGIN = osp.join(PATH_BASE, 'origindata')

def add_noise(img, sigma):
    return util.random_noise(img, var=(sigma / 255.0) ** 2)

clean = io.imread(osp.join(PATH_ORIGIN, str(1), str(1) + '.png'), as_grey=True)
clean = myimutil.img_normalize(clean)
noise = add_noise(clean, 10)
print(clean.max(), clean.min())
print(noise.max(), noise.min())
sub = noise - clean
print(sub.max(), sub.min())