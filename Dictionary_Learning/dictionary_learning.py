import numpy as np
from ksvd import ApproximateKSVD
from skimage import io, util
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
import os
import os.path as osp
import matplotlib.pyplot as plt
PATH = 'E:/workspace/deeplearning/keras/file/gan_denoise/Sparsity_SDOCT_DATASET_2012'
IM_NUM = 1


def learn_dictionary():
    img = io.imread(osp.join(PATH, str(IM_NUM), str(IM_NUM) + '_Averaged Image.tif'), as_grey=True)
    img = util.img_as_float(img)
    patch_size = (16, 16)
    patches = extract_patches_2d(img, patch_size)
    signals = patches.reshape(patches.shape[0], -1)
    mean = np.mean(signals, axis=1)[:, np.newaxis]
    signals -= mean
    aksvd = ApproximateKSVD(n_components=64)
    dictionary = aksvd.fit(signals[:10000]).components_
    np.save('dictionary.npy', dictionary)


def save_show():
    dictionary = np.load('dictionary.npy')
    dictionary = dictionary.reshape((-1, 16, 16))
    dictionary += 0.3
    dictionary = util.img_as_ubyte(dictionary)
    save_show = np.zeros((8 * 16, 8 * 16), dtype=np.uint8)
    for i in range(8):
        for j in range(8):
            ith = i * 8 + j
            save_show[i * 16: i * 16 + 16, j * 16: j * 16 + 16] = dictionary[ith]
    io.imsave('test.png', save_show)


def clip(img):
    img[np.where(img > 1)] = 1
    img[np.where(img < 0)] = 0
    return img


def denoise():
    img = io.imread(osp.join(PATH, str(IM_NUM), str(IM_NUM) + '_Raw Image.tif'), as_grey=True)
    img = util.img_as_float(img)
    patch_size = (16, 16)
    patches = extract_patches_2d(img, patch_size)
    signals = patches.reshape(patches.shape[0], -1)
    mean = np.mean(signals, axis=1)[:, np.newaxis]
    signals -= mean
    dictionary = np.load('dictionary.npy')
    aksvd = ApproximateKSVD(n_components=64)
    aksvd.components_ = dictionary
    gamma = aksvd.transform(signals)
    reduced = gamma.dot(dictionary) + mean
    reduced_img = reconstruct_from_patches_2d(
        reduced.reshape(patches.shape), img.shape)
    io.imsave('test_output.png', clip(reduced_img))

if __name__ == '__main__':
    # learn_dictionary()
    # save_show()
    denoise()