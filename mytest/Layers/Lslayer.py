from keras import backend as K
from keras.engine.topology import Layer
from keras.layers.merge import _Merge
import numpy as np
import tensorflow as tf


class Lslayer(_Merge):
    # 最小二乘Layer，输入的第一个为目标矩阵，剩下的为待拟合矩阵（字典）
    # 这个pinv函数似乎不够给力，必须是方阵
    def __init__(self, axis=-1, **kwargs):
        super(Lslayer, self).__init__(**kwargs)
        self.axis = axis
        self.supports_masking = True
        self._reshape_required = False

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Concatenate` layer should be called '
                             'on a list of 2 inputs')
        if all([shape is None for shape in input_shape]):
            return
        if input_shape[0][-1] != input_shape[1][-1]:
            raise ValueError('The last axis of inputs should be the same.')

    def pinv(self, A, b, reltol=1e-6):
        s, u, v = tf.svd(A)
        atol = tf.reduce_max(s) * reltol
        s = tf.boolean_mask(s, s > atol)
        s_inv = tf.diag(tf.concat([1. / s, tf.zeros([tf.size(b) - tf.size(s)])], 0))
        return tf.matmul(v, tf.matmul(s_inv, tf.matmul(u, tf.reshape(b, [-1, 1]), transpose_a=True)))

    def LS(self, inputs):
        aim = inputs[0]
        dct = inputs[1]
        ret = []
        for i in range(aim.shape[0]):
            weights = tf.transpose(self.pinv(tf.transpose(dct[i]), tf.transpose(aim[i])))
            ret.append(dct[i] * tf.transpose(weights))
        # 字典每行是一个字典项（比如256的向量，表示16*16的一个字典项）
        # dct * tf.transpose(weights) 得到加权后的每行
        return tf.stack(ret)

    def _merge_function(self, inputs):
        return self.LS(inputs)

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A `Concatenate` layer should be called '
                             'on a list of inputs.')
        input_shapes = input_shape
        output_shape = list(input_shapes[0])
        output_shape[self.axis] = 0  # 最小二乘层输出不包含“目标"矩阵
        for shape in input_shapes[1:]:
            if output_shape[self.axis] is None or shape[self.axis] is None:
                output_shape[self.axis] = None
                break
            output_shape[self.axis] += shape[self.axis]
        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        if not isinstance(mask, list):
            raise ValueError('`mask` should be a list.')
        if not isinstance(inputs, list):
            raise ValueError('`inputs` should be a list.')
        if len(mask) != len(inputs):
            raise ValueError('The lists `inputs` and `mask` '
                             'should have the same length.')
        if all([m is None for m in mask]):
            return None
        # Make a list of masks while making sure
        # the dimensionality of each mask
        # is the same as the corresponding input.
        masks = []
        for input_i, mask_i in zip(inputs, mask):
            if mask_i is None:
                # Input is unmasked. Append all 1s to masks,
                masks.append(K.ones_like(input_i, dtype='bool'))
            elif K.ndim(mask_i) < K.ndim(input_i):
                # Mask is smaller than the input, expand it
                masks.append(K.expand_dims(mask_i))
            else:
                masks.append(mask_i)
        concatenated = K.concatenate(masks, axis=self.axis)
        return K.all(concatenated, axis=-1, keepdims=False)

    def get_config(self):
        config = {
            'axis': self.axis,
        }
        base_config = super(Lslayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))