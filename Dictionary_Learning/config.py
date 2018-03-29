import os.path as osp
from myutils import my_utils

PATH_BASE = 'E:/workspace/deeplearning/keras/file/learn'

PATH_MODEL = osp.join(PATH_BASE, 'models')
PATH_WATCHING = osp.join(PATH_BASE, 'watching')

######################################################################
# 网络参数
DICT_LENTH = 64
PATCH_SIZE = 16
EPOCH_NUM = 50
BATCH_SIZE = 128
SHOW_INTERVAL = 500
MODEL_NAME = 'dic_learn'


my_utils.make_dir(PATH_WATCHING)
my_utils.make_dir(PATH_MODEL)
