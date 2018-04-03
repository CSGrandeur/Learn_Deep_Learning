from skimage import io, util
import sklearn
import numpy as np


def img_normalize(img):
    # 把图像归一化，可以改定义
    # return util.img_as_float(img)
    return (img.astype(np.float64) - 127.5) / 127.5

def img_normalize_recover(img):
    # 从归一化的图像恢复到uint8，对应img_normalize
    # return util.img_as_ubyte(img)
    return (clip(img) * 127.5 + 127.5).astype(np.uint8)

def clip(img):
    # 处理变换后的图像越界问题
    img[np.where(img > 1)] = 1
    img[np.where(img < -1)] = -1
    return img

def imshow(img):
    io.imshow(img)