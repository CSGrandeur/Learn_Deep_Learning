from keras.layers import Input, Dense, Reshape, BatchNormalization, \
    UpSampling2D, Conv2D, Activation, MaxPooling2D, Flatten, Add, Multiply
from keras.models import Model, Sequential, load_model
from keras.datasets import mnist
from keras.optimizers import SGD
from keras import backend as K
from keras import losses
from myutils import my_utils as myutils
import math
import numpy as np
import os.path as osp

inputs = Input(shape=(0,))
init_noise = K.variable(np.random.random((1, 16, 16, 1)))
print(init_noise)
x = Conv2D(1, (5, 5), activation='relu', padding='same')(init_noise)

b = Input((128, ))
b = Dense(1)(b)
print(b.shape)
print(x.shape)
c = Multiply()([b, x])
print(c)
print(type(c[:,:,0]))
M = Model(inputs=inputs, outputs=c)
M.compile('sgd', 'binary_crossentropy')
