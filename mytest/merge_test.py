from keras.layers import Input, Dense, Reshape, BatchNormalization, \
    UpSampling2D, Conv2D, Activation, MaxPooling2D, Flatten, Add
from keras.models import Model, Sequential, load_model
from keras.datasets import mnist
from keras.optimizers import SGD
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras import losses
from myutils import my_utils as myutils
import math
import numpy as np
import os.path as osp

INIT_RANDOM_SIZE = 256
PATCH_SIZE = 16

def dict_model():
    # 生成 dict 的模块
    init_random = Input(shape=(INIT_RANDOM_SIZE,))
    x = Dense(INIT_RANDOM_SIZE * 2)(init_random)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(INIT_RANDOM_SIZE * 4)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(PATCH_SIZE * PATCH_SIZE, activation='tanh')(x)
    outputs = Reshape((PATCH_SIZE, PATCH_SIZE))(x)
    G_DIC = Model(inputs=init_random, outputs=outputs)
    return G_DIC


G_DIC_LIST = []
for i in range(128):
    G_DIC_LIST.append(dict_model())

init_random = Input(shape=(INIT_RANDOM_SIZE,))
inputs = Input(shape=(PATCH_SIZE, PATCH_SIZE, 1))
G_DIC_outputs = []
for G_DIC in G_DIC_LIST:
    G_DIC_outputs.append(G_DIC(init_random))
