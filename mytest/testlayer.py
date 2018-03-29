from keras.layers import Input, Dense, BatchNormalization, \
    Conv2D, Activation, MaxPooling2D, Flatten, Add, Multiply, Reshape, Concatenate, Permute
from keras.models import Model, Sequential, load_model
from keras.datasets import mnist
from keras.optimizers import SGD
from keras import backend as K
from keras import losses
from myutils import my_utils as myutils
import math
import numpy as np
import os.path as osp

from mytest.Layers.Lslayer import Lslayer

a = Input(shape=(8, 8, 1), batch_shape=(1, 8, 8, 1))
b = Conv2D(1, (3, 3), padding='same')(a)
b = Permute((3, 1, 2))(b)
dict = [b for i in range(64)]
dict = Concatenate(axis=-3)(dict)
print(dict.shape)


aim = Reshape((1, 64,))(a)
dict = Reshape((64, 64))(dict)

print(aim.shape)
print(dict.shape)

o = Lslayer()([aim, dict])
print(o.shape)