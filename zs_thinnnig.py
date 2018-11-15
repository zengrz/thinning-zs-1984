# File: zs_thinning.py
# Author: Zeng Ruizi (Rey)

import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as pp
from operator import itemgetter
import timeit

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

img_path = ""

m = 512
n = 512
thresh = 60

im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# crop and thresh
im = im[:m, :n]
im_mask = im < thresh
im[im_mask] = 0
im_mask = np.logical_not(im_mask)
im[im_mask] = 1

x_val = np.expand_dims(np.expand_dims(im, 0), -1)

# x_img = tf.constant(x_val, tf.float32, (1, m, n, 1), 'x')
x_img = tf.get_variable('x', shape=(1, m, n, 1), dtype=tf.float32)

def zs_thinning(x):
    x_shape = x.shape.as_list()

    indices_shape = np.array(x_shape)
    indices_shape = np.append(indices_shape, 8) # 8 = num neighbours
    indices_shape = np.append(indices_shape, len(x.shape.as_list()))

    indices_1 = np.zeros(indices_shape, np.int)
    indices_2 = np.zeros(indices_shape, np.int)

    list_1 = [[0, -1, 1, 0], [0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 1, -1, 0], [0, 0, -1, 0], [0, -1, -1, 0], [0, -1, 0, 0]] # 3, 4, 5, 6, 7, 8, 9, 2
    list_2 = [[0, -1, 0, 0], [0, -1, 1, 0], [0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 1, -1, 0], [0, 0, -1, 0], [0, -1, -1, 0]] # 2, 3, 4, 5, 6, 7, 8, 9
    """
    step 1
    In a batch, for every channel, scatter and process the neighbours across channels in orders described by list_1 and list_2.
    """

    # (1)
    nbhd_filt_val = [1, 1, 1, 1, 0, 1, 1, 1, 1]
    nbhd_filt = tf.constant(nbhd_filt_val, dtype=tf.float32, shape=(3, 3, 1, 1), name="nbhd_filt")
    x_nbhd_count = tf.nn.conv2d(x, nbhd_filt, (1, 1, 1, 1), "VALID")
    x_nbhd_count_betw_2_and_6 = tf.logical_and(x_nbhd_count > 1, x_nbhd_count < 7)
    x_nbhd_count_betw_2_and_6 = tf.pad(x_nbhd_count_betw_2_and_6, [[0, 0], [1, 1], [1, 1], [0, 0]], constant_values=tf.constant(False))

    # (2)
    # NHWC
    for i in range(x.shape[0].value):
        for j in range(1, x.shape[1].value - 1): # (0)
            for k in range(1, x.shape[2].value - 1): # (0)
                for l in range(x.shape[3].value):
                    indices_1[i, j, k, l] = np.add([i, j, k, l], list_1)
                    indices_2[i, j, k, l] = np.add([i, j, k, l], list_2)

    x_nbhd1 = tf.gather_nd(x, indices_1, name='x_nbhd1')
    x_nbhd2 = tf.gather_nd(x, indices_2, name='x_nbhd2')
    x_nbhd_01transition = x_nbhd1 - x_nbhd2
    x_nbhd_01transition = x_nbhd_01transition > 0
    # x_transition[0, 0, 0, 0] = 0 # removing aggregated values at the end
    x_nbhd_01transition = tf.cast(x_nbhd_01transition, tf.float32)
    x_nbhd_01transition_count = tf.reduce_sum(x_nbhd_01transition, axis=-1)
    x_nbhd_01transition_count_is_1 = tf.logical_and(x_nbhd_01transition_count > 0, x_nbhd_01transition_count < 2)

    # (3)
    x_negative = -(x - 1)

    nbhd_filt_val_1_1 = [0, 1, 0, 0, 0, 1, 0, 1, 0]
    nbhd_0count_1_1 = tf.constant(nbhd_filt_val_1_1, dtype=tf.float32, shape=(3, 3, 1, 1), name="nbhd_filt_1_1")
    x_0count_1_1 = tf.nn.conv2d(x_negative, nbhd_0count_1_1, (1, 1, 1, 1), "VALID")
    x_0count_1_1 = x_0count_1_1 > 0
    x_0count_1_1 = tf.pad(x_0count_1_1, [[0, 0], [1, 1], [1, 1], [0, 0]], constant_values=tf.constant(False))

    # (4)
    nbhd_filt_val_1_2 = [0, 0, 0, 1, 0, 1, 0, 1, 0]
    nbhd_0count_1_2 = tf.constant(nbhd_filt_val_1_2, dtype=tf.float32, shape=(3, 3, 1, 1), name="nbhd_filt_1_2")
    x_0count_1_2 = tf.nn.conv2d(x_negative, nbhd_0count_1_2, (1, 1, 1, 1), "VALID")
    x_0count_1_2 = x_0count_1_2 > 0
    x_0count_1_2 = tf.pad(x_0count_1_2, [[0, 0], [1, 1], [1, 1], [0, 0]], constant_values=tf.constant(False))

    x_mask_1 = tf.logical_and(x_nbhd_count_betw_2_and_6, x_nbhd_01transition_count_is_1)
    x_mask_1 = tf.logical_and(x_mask_1, x_0count_1_1)
    x_mask_1 = tf.logical_and(x_mask_1, x_0count_1_2)
    x_mask_1 = tf.cast(tf.logical_not(x_mask_1), tf.float32)
    x_1 = tf.multiply(x, x_mask_1)

    # grad_x1 = tf.gradients(x_1, [x])

    x_changed_1 = tf.reduce_sum(tf.cast(x_mask_1, tf.float32)) > 0

    """
    step 2
    similar to step 1
    """

    # (1)
    nbhd_filt_val = [1, 1, 1, 1, 0, 1, 1, 1, 1]
    nbhd_filt = tf.constant(nbhd_filt_val, dtype=tf.float32, shape=(3, 3, 1, 1), name="nbhd_filt")
    x_nbhd_count = tf.nn.conv2d(x, nbhd_filt, (1, 1, 1, 1), "VALID")
    x_nbhd_count_betw_2_and_6 = tf.logical_and(x_nbhd_count > 1, x_nbhd_count < 7)
    x_nbhd_count_betw_2_and_6 = tf.pad(x_nbhd_count_betw_2_and_6, [[0, 0], [1, 1], [1, 1], [0, 0]], constant_values=tf.constant(False))

    # (2)
    # NHWC
    for i in range(x.shape[0].value):
        for j in range(1, x.shape[1].value - 1): # (0)
            for k in range(1, x.shape[2].value - 1): # (0)
                for l in range(x.shape[3].value):
                    indices_1[i, j, k, l] = np.add([i, j, k, l], list_1)
                    indices_2[i, j, k, l] = np.add([i, j, k, l], list_2)

    x_nbhd1 = tf.gather_nd(x, indices_1, name='x_nbhd1')
    x_nbhd2 = tf.gather_nd(x, indices_2, name='x_nbhd2')
    x_nbhd_01transition = x_nbhd1 - x_nbhd2
    x_nbhd_01transition = x_nbhd_01transition > 0
    # x_transition[0, 0, 0, 0] = 0 # removing aggregated values at the end
    x_nbhd_01transition = tf.cast(x_nbhd_01transition, tf.float32)
    x_nbhd_01transition_count = tf.reduce_sum(x_nbhd_01transition, axis=-1)
    x_nbhd_01transition_count_is_1 = tf.logical_and(x_nbhd_01transition_count > 0, x_nbhd_01transition_count < 2)

    # (3)
    x_negative = -(x - 1)

    nbhd_filt_val_2_1 = [0, 0, 0, 0, 0, 1, 0, 1, 0]
    nbhd_0count_2_1 = tf.constant(nbhd_filt_val_2_1, dtype=tf.float32, shape=(3, 3, 1, 1), name="nbhd_filt_2_1")
    x_0count_2_1 = tf.nn.conv2d(x, nbhd_0count_2_1, (1, 1, 1, 1), "VALID")
    x_0count_2_1 = x_0count_2_1 > 0
    x_0count_2_1 = tf.pad(x_0count_2_1, [[0, 0], [1, 1], [1, 1], [0, 0]], constant_values=tf.constant(False))

    # (4)
    nbhd_filt_val_2_2 = [0, 0, 0, 1, 0, 1, 0, 1, 0]
    nbhd_0count_2_2 = tf.constant(nbhd_filt_val_2_2, dtype=tf.float32, shape=(3, 3, 1, 1), name="nbhd_filt_2_2")
    x_0count_2_2 = tf.nn.conv2d(x, nbhd_0count_2_2, (1, 1, 1, 1), "VALID")
    x_0count_2_2 = x_0count_2_2 > 0
    x_0count_2_2 = tf.pad(x_0count_2_2, [[0, 0], [1, 1], [1, 1], [0, 0]], constant_values=tf.constant(False))

    x_mask_2 = tf.logical_and(x_nbhd_count_betw_2_and_6, x_nbhd_01transition_count_is_1)
    x_mask_2 = tf.logical_and(x_mask_2, x_0count_2_1)
    x_mask_2 = tf.logical_and(x_mask_2, x_0count_2_2)
    x_mask_2 = tf.cast(tf.logical_not(x_mask_2), tf.float32)
    x_2 = tf.multiply(x_1, x_mask_2)

    x_changed_2 = tf.reduce_sum(tf.cast(x_mask_2, tf.float32)) > 0

    # x_changed = tf.logical_and(x_changed_1, x_changed_2)

    # grad_x2 = tf.gradients(x_2, [x])

    return x_2

num_loops = 5

def cond(i, x_in):
    return tf.less(i, num_loops)

def body(i, x_in):
    return tf.add(i, 1), zs_thinning(x_in)

c = tf.constant(0)

thinning_loop = tf.while_loop(cond, body, [c, x_img])

with tf.Session() as sess:
    c, x_2_out = sess.run(thinning_loop, feed_dict={x_img: x_val})

cv2.imwrite("x_2_out_" + str(num_loops) + ".png", np.squeeze(x_2_out*255))
