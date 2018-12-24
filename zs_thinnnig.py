# File: zs_thinning.py
# Author: Zeng Ruizi (Rey)

import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as pp
from operator import itemgetter
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

BATCH_SIZE = 1
m = 512
n = 512
NUM_CHANNELS = 1
thresh = 180
NUM_NEIGHBOURS = 8

img_path = ""
im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
im = im[m:2*m, :n]
im = im > thresh

print(im.shape)

cv2.imwrite("im_th.png", np.squeeze(im*255))

img_path2 = ""
im2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
im2 = im2[:m, :n]
im2 = im2 > thresh

x_img_val = np.expand_dims(np.expand_dims(im, 0), -1)
# x_img_val = np.expand_dims(np.array([im, im2]), -1)

# x_img = tf.constant(x_val, tf.float32, (1, m, n, 1), 'x')
x_img_var = tf.get_variable('x', shape=(BATCH_SIZE, m, n, NUM_CHANNELS), dtype=tf.float32)

def zs_thinning(x, indices_1, indices_2):
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
    x_nbhd1 = tf.gather_nd(x, indices_1, name='x_nbhd1')
    x_nbhd2 = tf.gather_nd(x, indices_2, name='x_nbhd2')
    x_nbhd_01transition = x_nbhd1 - x_nbhd2
    x_nbhd_01transition = x_nbhd_01transition > 0
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
    x1 = tf.multiply(x, x_mask_1)

    # grad_x1 = tf.gradients(x_1, [x])

    x_changed_1 = tf.reduce_sum(tf.cast(x_mask_1, tf.float32)) > 0

    """
    step 2
    similar to step 1
    """

    # (1)
    x1_nbhd_count = tf.nn.conv2d(x1, nbhd_filt, (1, 1, 1, 1), "VALID")
    x1_nbhd_count_betw_2_and_6 = tf.logical_and(x1_nbhd_count > 1, x1_nbhd_count < 7)
    x1_nbhd_count_betw_2_and_6 = tf.pad(x1_nbhd_count_betw_2_and_6, [[0, 0], [1, 1], [1, 1], [0, 0]], constant_values=tf.constant(False))

    # (2)
    x1_nbhd1 = tf.gather_nd(x1, indices_1, name='x1_nbhd1')
    x1_nbhd2 = tf.gather_nd(x1, indices_2, name='x1_nbhd2')
    x1_nbhd_01transition = x1_nbhd1 - x1_nbhd2
    x1_nbhd_01transition = x1_nbhd_01transition > 0
    x1_nbhd_01transition = tf.cast(x1_nbhd_01transition, dtype=tf.float32)
    x1_nbhd_01transition_count = tf.reduce_sum(x1_nbhd_01transition, axis=-1)
    x1_nbhd_01transition_count_is_1 = tf.logical_and(x1_nbhd_01transition_count > 0, x1_nbhd_01transition_count < 2)

    # (3)
    x1_negative = -(x1 - 1)

    nbhd_filt_val_2_1 = [0, 0, 0, 0, 0, 1, 0, 1, 0]
    nbhd_0count_2_1 = tf.constant(nbhd_filt_val_2_1, dtype=tf.float32, shape=(3, 3, 1, 1), name="nbhd_filt_2_1")
    x1_0count_2_1 = tf.nn.conv2d(x1_negative, nbhd_0count_2_1, (1, 1, 1, 1), "VALID")
    x1_0count_2_1 = x1_0count_2_1 > 0
    x1_0count_2_1 = tf.pad(x1_0count_2_1, [[0, 0], [1, 1], [1, 1], [0, 0]], constant_values=tf.constant(False))

    # (4)
    nbhd_filt_val_2_2 = [0, 0, 0, 1, 0, 1, 0, 1, 0]
    nbhd_0count_2_2 = tf.constant(nbhd_filt_val_2_2, dtype=tf.float32, shape=(3, 3, 1, 1), name="nbhd_filt_2_2")
    x1_0count_2_2 = tf.nn.conv2d(x1_negative, nbhd_0count_2_2, (1, 1, 1, 1), "VALID")
    x1_0count_2_2 = x1_0count_2_2 > 0
    x1_0count_2_2 = tf.pad(x1_0count_2_2, [[0, 0], [1, 1], [1, 1], [0, 0]], constant_values=tf.constant(False))

    x1_mask_2 = tf.logical_and(x1_nbhd_count_betw_2_and_6, x1_nbhd_01transition_count_is_1)
    x1_mask_2 = tf.logical_and(x1_mask_2, x1_0count_2_1)
    x1_mask_2 = tf.logical_and(x1_mask_2, x1_0count_2_2)
    x1_mask_2 = tf.cast(tf.logical_not(x1_mask_2), tf.float32)
    x2 = tf.multiply(x1, x1_mask_2)

    x2_changed = tf.reduce_sum(tf.cast(x1_mask_2, tf.float32)) > 0

    # x_changed = tf.logical_and(x_changed_1, x_changed_2)

    # grad_x2 = tf.gradients(x_2, [x])

    return x2

NUM_LOOPS = 3

x_shape = np.array([BATCH_SIZE, m, n, NUM_CHANNELS])
indices_shape = np.append(x_shape, NUM_NEIGHBOURS)
indices_shape = np.append(indices_shape, x_shape.size)

indices_1_val = np.zeros(indices_shape, np.int)
indices_2_val = np.zeros(indices_shape, np.int)

list_1 = [[0, -1, 1, 0], [0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 1, -1, 0], [0, 0, -1, 0], [0, -1, -1, 0], [0, -1, 0, 0]] # 3, 4, 5, 6, 7, 8, 9, 2
list_2 = [[0, -1, 0, 0], [0, -1, 1, 0], [0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 1, -1, 0], [0, 0, -1, 0], [0, -1, -1, 0]] # 2, 3, 4, 5, 6, 7, 8, 9

# indices = np.indices(x_shape)
# indices = np.expand_dims(indices, -1)
# indices = np.transpose(indices, [1, 2, 3, 4, 5, 0]) # each element corresponds to the index of the current element
# indices = np.broadcast_to(indices, indices_shape)

# list_1_full = np.broadcast_to(list_1, indices_shape)
# list_2_full = np.broadcast_to(list_2, indices_shape)
# indices_1_val = np.add(indices, list_1_full)
# indices_2_val = np.add(indices, list_2_full)

# (2)
# NHWC
for i in range(BATCH_SIZE): # replying on broadcasting
    for j in range(1, m - 1): # (0)
        for k in range(1, n - 1): # (0)
            for l in range(NUM_CHANNELS):
                indices_1_val[i, j, k, l] = np.add([i, j, k, l], list_1)
                indices_2_val[i, j, k, l] = np.add([i, j, k, l], list_2)

indices_1 = tf.constant(indices_1_val, dtype=tf.int32, shape=[BATCH_SIZE, m, n, NUM_CHANNELS, NUM_NEIGHBOURS, 4])
indices_2 = tf.constant(indices_2_val, dtype=tf.int32, shape=[BATCH_SIZE, m, n, NUM_CHANNELS, NUM_NEIGHBOURS, 4])

def cond(i, x_in):
    return tf.less(i, NUM_LOOPS)

def body(i, x_in):
    return tf.add(i, 1), zs_thinning(x_in, indices_1, indices_2)

c = tf.constant(0)
zs_results = tf.while_loop(cond, body, [c, x_img_var])

init_op = tf.global_variables_initializer()

saver = tf.train.Saver(tf.all_variables())

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init_op)
    start = time.time()
    c, x_2_out = sess.run(zs_results, feed_dict={x_img_var: x_img_val})
    end = time.time()
    print(end - start)
    save_path = saver.save(sess, "./model.ckpt")

cv2.imwrite("x_2_out_" + str(NUM_LOOPS) + "_0" + ".png", np.squeeze(x_2_out[0]*255))
# cv2.imwrite("x_2_out_" + str(NUM_LOOPS) + "_1" + ".png", np.squeeze(x_2_out[1]*255))
