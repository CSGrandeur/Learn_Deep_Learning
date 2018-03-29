from keras.layers import Input, Dense, Reshape, BatchNormalization, \
    UpSampling2D, Conv2D, Activation, MaxPooling2D, Flatten, Add
from keras.models import Model, Sequential, load_model
from keras.datasets import mnist
from keras.optimizers import SGD
from keras import backend as K
from keras import losses
from myutils import my_utils as myutils
import math
import numpy as np
import os.path as osp
import cv2


def model_generator(inputs):
    # 生成模块
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    outputs = Conv2D(1, (1, 1), activation='tanh', padding='same')(x)
    G = Model(inputs=inputs, outputs=outputs)
    return G


def model_discriminator(inputs):
    # 判别模块
    df_dim = 1
    x = Conv2D(df_dim, (3, 3), activation='relu', padding='same')(inputs)
    x = Flatten()(x)
    x = Dense(2, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    D = Model(inputs=inputs, outputs=outputs)
    return D


def set_trainable(model, trainable=False):
    # 设置模块是否可训练
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def model_gan(G, D, inputs):
    # 生成判别连接
    generated = G(inputs)
    discriminate = D(generated)
    GAN = Model(inputs=[inputs], outputs=[generated, discriminate])
    return GAN


inputs_g = Input(shape=(8, 8, 1))
inputs_d = Input(shape=(8, 8, 1))
G = model_generator(inputs_g)
D = model_discriminator(inputs_d)

# set_trainable(D, False)
# GAN = model_gan(G, D, inputs_g)
# GAN.compile(optimizer='SGD', loss=['mse', 'binary_crossentropy'], loss_weights=[1, 1e-3])
# set_trainable(D, True)
# GAN2 = model_gan(G, D, inputs_g)
# GAN2 = model_gan(G, D, inputs_g)
# GAN2.compile(optimizer='SGD', loss=['mse', 'binary_crossentropy'], loss_weights=[1, 1e-3])
#
# GAN.summary()
# print("\n\n##########\n\n")
# GAN2.summary()
# print("\n\n##########\n\n")


GAN = model_gan(G, D, inputs_g)
GAN1 = GAN
GAN2 = GAN
set_trainable(D, False)
GAN1.compile(optimizer='SGD', loss=['mse', 'binary_crossentropy'], loss_weights=[1, 1e-3])
set_trainable(D, True)
GAN2.compile(optimizer='SGD', loss=['mse', 'binary_crossentropy'], loss_weights=[1, 1e-3])

GAN1.summary()
print("\n\n##########\n\n")
GAN2.summary()
print("\n\n##########\n\n")
