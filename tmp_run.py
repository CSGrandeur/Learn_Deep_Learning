import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d

dictionary = np.load('E:/workspace/deeplearning/keras/file/learn/dictionary.npy')
dictionary = dictionary.reshape((-1, 16, 16))
dictionary += 0.3

c = np.zeros((128, 16, 16, 64), dtype=np.float64)
for i in range(128):
    for j in range(64):
        c[i, :, :, j] = dictionary[j]

np.save('E:/workspace/deeplearning/keras/file/learn/init_dict_ksvd.npy', c)
print(c.shape)