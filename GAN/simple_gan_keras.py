from keras.layers import Input, Dense, Reshape, BatchNormalization, \
    UpSampling2D, Conv2D, Activation, MaxPooling2D, Flatten
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.optimizers import SGD
from myutils import my_utils as myutils
import math
import numpy as np
import cv2

# 用 minist 数据库训练生成手写数字

PATH_BASE = '/root/userfolder/file/gan_learn'

def generator_model():
    # 生成模块
    inputs = Input(shape=(100,))  # 输入100个随机噪声float
    x = Dense(1024, activation='tanh')(inputs)  # 全连接并非线性变换
    x = Dense(128 * 7 * 7)(x)  # 全连接为特定形状，即容纳128个7*7的featuremap，不过这一步得到的还是一维向量
    x = BatchNormalization()(x)  # 归一化至：均值0、方差为1，属于梯度优化
    x = Activation('tanh')(x)
    x = Reshape((7, 7, 128))(x)  # 转换为二维featuremap
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (5, 5), activation='tanh', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)  # 通过两次上采样得到28*28的数据（minist的单张数字图像）
    outputs = Conv2D(1, (5, 5), activation='tanh', padding='same')(x)
    G = Model(inputs=inputs, outputs=outputs)
    G.compile(optimizer='SGD', loss='binary_crossentropy')
    return G


def discriminator_model():
    # 判别模块，传统的二分类网络
    inputs = Input(shape=(28, 28, 1))
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    D = Model(inputs=inputs, outputs=outputs)
    D.compile(optimizer=SGD(lr=0.0005, momentum=0.9, nesterov=True), loss='binary_crossentropy')
    return D


def gan_model(G, D):
    # 生成模型链接判别模型，用来对抗。在训练生成模型时，判别模型要固定，即 D.trainable = False
    D.trainable = False
    inputs = Input(shape=(100,))
    x = G(inputs)
    outputs = D(x)
    GAN = Model(inputs, outputs)
    GAN.compile(optimizer=SGD(lr=0.0005, momentum=0.9, nesterov=True), loss='binary_crossentropy')
    return GAN


def combine_images(generated_images):
    # 把一个 batch 的数字拼成一张图保存方便查看
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img[:, :, 0]
    return image


def train(BATCH_SIZE):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train[:, :, :, None]

    G = generator_model()
    D = discriminator_model()
    GAN = gan_model(G, D)
    BATCH_NUM = int(X_train.shape[0]/BATCH_SIZE)
    for epoch in range(100):
        print("Epoch is", epoch)
        print(("Number of batches: %d" % BATCH_NUM))
        aver_d_loss = 0
        aver_g_loss = 0
        for index in range(BATCH_NUM):
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))  # 生成噪声，作为 G 的输入
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]  # Ground Truth(GT) 的一个batch的数据
            generated_images = G.predict(noise, verbose=0)  # G 由噪声生成一批伪数字
            if index % 50 == 0:
                # 输出一次 G 的结果用于查看
                image = combine_images(generated_images)
                image = image * 127.5 + 127.5
                image[np.where(image > 255)] = 255
                image[np.where(image < 0)] = 0
                image.astype(np.uint8)
                cv2.imwrite(PATH_BASE + '/' + ("%03d" % epoch) + "_" + ("%03d" % index) + ".png", image)
            X = np.concatenate((image_batch, generated_images))  # 拼接伪数字和 GT 的数据
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE  # 给伪数字和 GT 分别赋予 0 和 1 的标签用于训练 D
            D.trainable = True  # 训练 D 时 trainable 要为 True
            d_loss = D.train_on_batch(X, y)
            aver_d_loss += d_loss
            info = ('d_loss=%.3f' % d_loss)
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            D.trainable = False
            g_loss = GAN.train_on_batch(noise, [1] * BATCH_SIZE)
            aver_g_loss += g_loss
            info += (' g_loss=%.3f' % g_loss)
            myutils.ratio_bar(index, BATCH_NUM, info, displayType='total')
            if index % 10 == 9:
                G.save_weights(PATH_BASE + '/generator.weight', True)
                D.save_weights(PATH_BASE + '/discriminator.weight', True)
        info = ('d_loss=%.3f g_loss=%.3f' % (aver_d_loss / BATCH_NUM, aver_g_loss / BATCH_NUM))
        myutils.ratio_bar(BATCH_NUM, BATCH_NUM, info, displayType='total')
        print('')


def generate(BATCH_SIZE, nice=False):
    G = generator_model()
    G.compile(loss='binary_crossentropy', optimizer="SGD")
    G.load_weights(PATH_BASE + '/generator.weight')
    if nice:
        # 输出较好的结果，即随机生成多组伪数字，通过 D 得到分数，排序后输出最好的结果
        D = discriminator_model()
        D.compile(loss='binary_crossentropy', optimizer="SGD")
        D.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))
        generated_images = G.predict(noise, verbose=1)
        d_pret = D.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        generated_images = G.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    image[np.where(image > 255)] = 255
    image[np.where(image < 0)] = 0
    cv2.imwrite(PATH_BASE + '/generated_image.png', image.astype(np.uint8))



if __name__ == '__main__':
    myutils.make_dir(PATH_BASE)
    train(128)
    generate(128, True)