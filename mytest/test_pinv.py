import numpy as np
from scipy.optimize import leastsq
import tensorflow as tf

# a = np.random.random((64, 256))
# aim = np.ones((1, 256))
# pa = np.linalg.pinv(a)
#
# print(aim.shape)
# print(pa.shape)
# weights = np.dot(aim, pa)
#
# print(weights.shape)
# print(a.shape)
# verify = np.dot(weights, a)
# print(verify.shape)
# print(verify)


def pinv(A, b, reltol=1e-6):
    s, u, v = tf.svd(A)
    atol = tf.reduce_max(s) * reltol
    s = tf.boolean_mask(s, s > atol)
    s_inv = tf.diag(tf.concat([1. / s, tf.zeros([tf.size(b) - tf.size(s)])], 0))
    return tf.matmul(v, tf.matmul(s_inv, tf.matmul(u, tf.reshape(b, [-1, 1]), transpose_a=True)))

A = tf.random_normal((64, 64))
aim = tf.ones((1, 64))
weights = tf.transpose(pinv(tf.transpose(A), tf.transpose(aim)))
print(weights.shape)
AA = tf.matmul(weights, A)
BB = A * tf.transpose(weights)
CC = tf.reduce_sum(BB, axis=0)
BB2 = tf.multiply(A, tf.transpose(weights))
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    AA, BB, BB2, CC = sess.run([AA, BB, BB2, CC])
    print(AA.shape)
    print(BB.shape)

# A = tf.Variable([[2, 2, 2], [3, 4, 5]])
# B = tf.Variable([[5, 6]])
# B = tf.transpose(B)
# AA = A * B
#
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print(sess.run(A))
#     print(sess.run(B))
#     print(sess.run(AA))