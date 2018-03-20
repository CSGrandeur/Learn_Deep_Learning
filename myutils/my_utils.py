# coding=utf-8
import os, sys
import cv2
import pickle
import numpy as np
import shutil

def make_dir(dir):
    try:
        os.makedirs(dir)
    except:
        pass


def load_file_str_list(filename):
    return open(filename).readlines()


def sqr(num):
    return num * num


def save_pickle(var, file):
    output = open(file, 'wb')
    pickle.dump(var, output)
    output.close()


def load_pickle(file):
    pkl_file = open(file, 'rb')
    var = pickle.load(pkl_file)
    pkl_file.close()
    return var

def ratio_bar(num, total, info='', displayType='percentage'):
    # 绘制百分比进度条
    rate = float(num) / total
    rate_num = int(rate * 100)
    if rate_num < 100:
        r = '\r[%s>%s]' % ("=" * rate_num, "." * (100 - rate_num - 1))
    else:
        r = '\r[%s%s]' % ("=" * rate_num, "." * (100 - rate_num - 1))
    if displayType == 'percentage':
        r += '%d%%' % (rate_num)
    elif displayType == 'total':
        r += '%d/%d' % (num, total)
    r += ' ' + info
    sys.stdout.write(r)
    sys.stdout.flush()


def trim_lines(list):
    for i in range(len(list)):
        list[i] = list[i].strip()
    return list


def rm_folder(folder):
    shutil.rmtree(folder)

def rm(path):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
    except:
        print('remove ' + path + ' failed')


def copy(path, aim):
    try:
        if os.path.isdir(path):
            shutil.copytree(path, aim)
        elif os.path.isfile(path):
            shutil.copyfile(path, aim)
    except:
        print('Copy ' + path + ' failed')
