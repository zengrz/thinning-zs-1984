# File: zs_thinning.py
# Author: Zeng Ruizi (Rey)

import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as pp
from operator import itemgetter
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

img_path = "/media/rey/Data/edge_detect/archive/train20180119-111351/results_model-110000/png/test0.png"

b = 1
m = 512
n = 512
num_channels = 1
thresh = 60
num_neighbours = 8

im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# crop and thresh
im = im[:m, :n]
im_mask = im < thresh
im[im_mask] = 0
im_mask = np.logical_not(im_mask)
im[im_mask] = 1

x_val = np.expand_dims(np.expand_dims(im, 0), -1)
# x_val = np.expand_dims(np.array([im, im]), -1)

def zs_thinning(x):
    # indices_shape = tf.constant([b, m, n, num_channels, num_neighbours, x.shape.ndims], dtype=tf.int32)

    idx = tf.meshgrid(tf.range(x.shape[0]), tf.range(x.shape[1]), tf.range(x.shape[2]), tf.range(x.shape[3]))
    idx = tf.stack(idx, axis=0)
    idx = tf.transpose(idx, [2, 1, 3, 4, 0]) # each element corresponds to the index of the current element
    idx = tf.expand_dims(idx, -2)
    idx = tf.tile(idx, [1, 1, 1, 1, num_neighbours, 1])
    idx = tf.to_int32(idx)

    list_1 = tf.constant([[0, -1, 1, 0], [0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 1, -1, 0], [0, 0, -1, 0], [0, -1, -1, 0], [0, -1, 0, 0]], dtype=tf.int32) # 3, 4, 5, 6, 7, 8, 9, 2
    list_2 = tf.constant([[0, -1, 0, 0], [0, -1, 1, 0], [0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 1, -1, 0], [0, 0, -1, 0], [0, -1, -1, 0]], dtype=tf.int32) # 2, 3, 4, 5, 6, 7, 8, 9
    
    list_1_full = tf.expand_dims(tf.broadcast_to(tf.broadcast_to(tf.expand_dims(list_1, 0), [n, 1, num_neighbours, x.shape.ndims]), [m, n, num_channels, num_neighbours, x.shape.ndims]), 0)
    list_2_full = tf.expand_dims(tf.broadcast_to(tf.broadcast_to(tf.expand_dims(list_2, 0), [n, 1, num_neighbours, x.shape.ndims]), [m, n, num_channels, num_neighbours, x.shape.ndims]), 0)
    """
    step 1
    In a batch, for every channel, scatter and process the neighbours across channels in orders described by list_1 and list_2.
    """

    # (1)
    nbhd_filt = tf.constant([1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=tf.float32, shape=(3, 3, 1, 1), name="nbhd_filt")
    x_nbhd_count = tf.nn.conv2d(x, nbhd_filt, (1, 1, 1, 1), "VALID")
    x_nbhd_count_betw_2_and_6 = tf.logical_and(x_nbhd_count > 1, x_nbhd_count < 7)
    x_nbhd_count_betw_2_and_6 = tf.pad(x_nbhd_count_betw_2_and_6, [[0, 0], [1, 1], [1, 1], [0, 0]], constant_values=tf.constant(False))

    # (2)
    # NHWC
    # indices_1 = tf.zeros(indices_shape, dtype=tf.int32)
    # indices_2 = tf.zeros(indices_shape, dtype=tf.int32)
    # for i in range(x.shape[0].value):
    #     for j in range(1, x.shape[1].value - 1): # (0)
    #         for k in range(1, x.shape[2].value - 1): # (0)
    #             for l in range(x.shape[3].value):
    #                 indices_1[i, j, k, l] = [i, j, k, l] + list_1
    #                 indices_2[i, j, k, l] = [i, j, k, l] + list_2
    indices_1 = idx + list_1_full
    indices_2 = idx + list_2_full

    zero_border_mask = tf.expand_dims(tf.expand_dims(tf.pad(tf.ones([1, m-2, n-2, 1], dtype=tf.int32), [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT'), -1), -1)

    indices_1 = tf.multiply(indices_1, zero_border_mask)
    indices_2 = tf.multiply(indices_2, zero_border_mask)

    # indices_1 = tf.multiply(indices_1, tf.cast(tf.greater(indices_1, 0), indices_1.dtype))
    # indices_2 = tf.multiply(indices_2, tf.cast(tf.greater(indices_2, 0), indices_2.dtype))
    # indices_1 = tf.multiply(indices_1, tf.cast(tf.less(indices_1, m), indices_1.dtype))
    # indices_2 = tf.multiply(indices_2, tf.cast(tf.less(indices_2, n), indices_2.dtype))

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

    nbhd_0count_1_1 = tf.constant([0, 1, 0, 0, 0, 1, 0, 1, 0], dtype=tf.float32, shape=(3, 3, 1, 1), name="nbhd_filt_1_1")
    x_0count_1_1 = tf.nn.conv2d(x_negative, nbhd_0count_1_1, (1, 1, 1, 1), "VALID")
    x_0count_1_1 = x_0count_1_1 > 0
    x_0count_1_1 = tf.pad(x_0count_1_1, [[0, 0], [1, 1], [1, 1], [0, 0]], constant_values=tf.constant(False))

    # (4)
    nbhd_0count_1_2 = tf.constant([0, 0, 0, 1, 0, 1, 0, 1, 0], dtype=tf.float32, shape=(3, 3, 1, 1), name="nbhd_filt_1_2")
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
    nbhd_filt = tf.constant([1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=tf.float32, shape=(3, 3, 1, 1), name="nbhd_filt")
    x_nbhd_count = tf.nn.conv2d(x, nbhd_filt, (1, 1, 1, 1), "VALID")
    x_nbhd_count_betw_2_and_6 = tf.logical_and(x_nbhd_count > 1, x_nbhd_count < 7)
    x_nbhd_count_betw_2_and_6 = tf.pad(x_nbhd_count_betw_2_and_6, [[0, 0], [1, 1], [1, 1], [0, 0]], constant_values=tf.constant(False))

    # (2)
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

    nbhd_0count_2_1 = tf.constant([0, 0, 0, 0, 0, 1, 0, 1, 0], dtype=tf.float32, shape=(3, 3, 1, 1), name="nbhd_filt_2_1")
    x_0count_2_1 = tf.nn.conv2d(x, nbhd_0count_2_1, (1, 1, 1, 1), "VALID")
    x_0count_2_1 = x_0count_2_1 > 0
    x_0count_2_1 = tf.pad(x_0count_2_1, [[0, 0], [1, 1], [1, 1], [0, 0]], constant_values=tf.constant(False))

    # (4)
    nbhd_0count_2_2 = tf.constant([0, 0, 0, 1, 0, 1, 0, 1, 0], dtype=tf.float32, shape=(3, 3, 1, 1), name="nbhd_filt_2_2")
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

num_loops = 15

def cond(i, x_in):
    return tf.less(i, num_loops)

def body(i, x_in):
    return tf.add(i, 1), zs_thinning(x_in)

c = tf.constant(0)
x_img = tf.get_variable('x', shape=(b, m, n, 1), dtype=tf.float32)

zs_results = tf.while_loop(cond, body, [c, x_img])
print(zs_results)

init_op = tf.global_variables_initializer()

saver = tf.train.Saver(tf.all_variables())

with tf.Session() as sess:
    sess.run(init_op)
    start = time.time()
    c, x_2_out = sess.run(zs_results, feed_dict={x_img: x_val})
    end = time.time()
    print(end - start)
    save_path = saver.save(sess, "./model_tf.ckpt")

cv2.imwrite("x_2_out_" + str(num_loops) + ".png", np.squeeze(x_2_out*255))
