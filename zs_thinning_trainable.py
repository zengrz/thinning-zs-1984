# File: zs_thinning_trainable.py
# Author: Zeng Ruizi (Rey)
#
# Differentiable Zhang-Suen-style thinning for use inside a training graph.
# This keeps the multiplicative thinning effect but replaces hard boolean
# deletion tests with smooth gates so gradients can flow backward.

import argparse
import os
import time

import numpy as np


tf = None
cv2 = None

NUM_CHANNELS = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply trainable Zhang-Suen-style thinning to a grayscale image."
    )
    parser.add_argument("input", help="Path to the input image.")
    parser.add_argument(
        "-o",
        "--output",
        default="zs_thinned_trainable.png",
        help="Path for the thinned output image.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=180,
        help="Foreground threshold. Pixels greater than this value are treated as 1.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Fixed thinning iterations to unroll.",
    )
    parser.add_argument(
        "--sharpness",
        type=float,
        default=20.0,
        help="Soft gate sharpness. Higher values behave more like hard Zhang-Suen rules.",
    )
    parser.add_argument(
        "--straight-through",
        action="store_true",
        help="Use hard binary output in the forward pass with soft gradients backward.",
    )
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("Y", "X", "HEIGHT", "WIDTH"),
        help="Optional crop rectangle before thresholding.",
    )
    parser.add_argument(
        "--threshold-output",
        help="Optional path for writing the thresholded input image.",
    )
    parser.add_argument(
        "--gpu",
        help="Optional CUDA_VISIBLE_DEVICES value, for example '0' or '2'.",
    )
    return parser.parse_args()


def ensure_tensorflow():
    global tf
    if tf is None:
        import tensorflow as tensorflow

        tf = tensorflow
        tf.compat.v1.disable_eager_execution()
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)


def ensure_opencv():
    global cv2
    if cv2 is None:
        import cv2 as opencv

        cv2 = opencv


def load_binary_image(path, threshold, crop=None):
    ensure_opencv()

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not read input image: {}".format(path))

    if crop is not None:
        y, x, height, width = crop
        image = image[y : y + height, x : x + width]
        if image.size == 0:
            raise ValueError("Crop produced an empty image.")

    return (image > threshold).astype(np.float32)


def neighbour_planes(x):
    filters = np.zeros((3, 3, 1, 8), dtype=np.float32)
    for channel, row_col in enumerate(
        [
            (0, 1),  # P2
            (0, 2),  # P3
            (1, 2),  # P4
            (2, 2),  # P5
            (2, 1),  # P6
            (2, 0),  # P7
            (1, 0),  # P8
            (0, 0),  # P9
        ]
    ):
        row, col = row_col
        filters[row, col, 0, channel] = 1

    return tf.nn.conv2d(
        x,
        tf.constant(filters, dtype=tf.float32, name="neighbour_plane_filters"),
        (1, 1, 1, 1),
        "SAME",
    )


def interior_pixels(x):
    interior = tf.ones_like(x[:, 1:-1, 1:-1, :], dtype=tf.bool)
    return tf.pad(
        interior,
        [[0, 0], [1, 1], [1, 1], [0, 0]],
        constant_values=tf.constant(False),
    )


def soft_between(x, low, high, sharpness):
    low_gate = tf.sigmoid(sharpness * (x - low))
    high_gate = tf.sigmoid(sharpness * (high - x))
    return low_gate * high_gate


def soft_equal(x, target, sharpness):
    return tf.exp(-sharpness * tf.square(x - target))


def straight_through_binary(x, threshold=0.5):
    hard = tf.cast(x > threshold, tf.float32)
    return tf.stop_gradient(hard - x) + x


def soft_deletion_candidates(x, step, sharpness=20.0):
    neighbours = neighbour_planes(x)
    p2 = neighbours[:, :, :, 0:1]
    p3 = neighbours[:, :, :, 1:2]
    p4 = neighbours[:, :, :, 2:3]
    p5 = neighbours[:, :, :, 3:4]
    p6 = neighbours[:, :, :, 4:5]
    p7 = neighbours[:, :, :, 5:6]
    p8 = neighbours[:, :, :, 6:7]
    p9 = neighbours[:, :, :, 7:8]

    neighbour_count = tf.reduce_sum(neighbours, axis=3, keepdims=True)
    count_gate = soft_between(neighbour_count, 2.0, 6.0, sharpness)

    ordered = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
    transitions = [(1.0 - ordered[i]) * ordered[i + 1] for i in range(8)]
    transition_count = tf.add_n(transitions)
    transition_gate = soft_equal(transition_count, 1.0, sharpness)

    if step == 1:
        condition_3 = soft_equal(p2 * p4 * p6, 0.0, sharpness)
        condition_4 = soft_equal(p4 * p6 * p8, 0.0, sharpness)
    else:
        condition_3 = soft_equal(p2 * p4 * p8, 0.0, sharpness)
        condition_4 = soft_equal(p2 * p6 * p8, 0.0, sharpness)

    foreground_gate = x
    interior_gate = tf.cast(interior_pixels(x), tf.float32)
    delete_prob = (
        foreground_gate
        * interior_gate
        * count_gate
        * transition_gate
        * condition_3
        * condition_4
    )
    return tf.clip_by_value(delete_prob, 0.0, 1.0)


def soft_zs_thinning_iteration(x, sharpness=20.0, straight_through=False):
    step_1_delete = soft_deletion_candidates(x, step=1, sharpness=sharpness)
    x_after_step_1 = x * (1.0 - step_1_delete)
    x_after_step_1 = tf.clip_by_value(x_after_step_1, 0.0, 1.0)
    if straight_through:
        x_after_step_1 = straight_through_binary(x_after_step_1)

    step_2_delete = soft_deletion_candidates(x_after_step_1, step=2, sharpness=sharpness)
    x_after_step_2 = x_after_step_1 * (1.0 - step_2_delete)
    x_after_step_2 = tf.clip_by_value(x_after_step_2, 0.0, 1.0)
    if straight_through:
        x_after_step_2 = straight_through_binary(x_after_step_2)

    return x_after_step_2


def soft_zs_thinning(x, iterations=5, sharpness=20.0, straight_through=False):
    for _ in range(iterations):
        x = soft_zs_thinning_iteration(
            x, sharpness=sharpness, straight_through=straight_through
        )
    return x


def run_thinning(binary_image, iterations, sharpness, straight_through=False):
    ensure_tensorflow()

    height, width = binary_image.shape
    if height < 3 or width < 3:
        raise ValueError("Image must be at least 3x3.")

    x_img_val = np.expand_dims(np.expand_dims(binary_image, 0), -1)

    x_img = tf.compat.v1.placeholder(
        tf.float32, shape=(1, height, width, NUM_CHANNELS), name="x"
    )
    result = soft_zs_thinning(
        x_img,
        iterations=iterations,
        sharpness=sharpness,
        straight_through=straight_through,
    )

    with tf.compat.v1.Session() as sess:
        start = time.time()
        output = sess.run(result, feed_dict={x_img: x_img_val})
        elapsed = time.time() - start

    return np.squeeze(output[0]), elapsed


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    binary_image = load_binary_image(args.input, args.threshold, args.crop)
    if args.threshold_output:
        ensure_opencv()
        cv2.imwrite(args.threshold_output, (binary_image * 255).astype(np.uint8))

    thinned, elapsed = run_thinning(
        binary_image,
        iterations=args.iterations,
        sharpness=args.sharpness,
        straight_through=args.straight_through,
    )
    ensure_opencv()
    cv2.imwrite(args.output, (thinned * 255).astype(np.uint8))

    print("Input shape: {}".format(binary_image.shape))
    print("Iterations: {}".format(args.iterations))
    print("Sharpness: {:.6f}".format(args.sharpness))
    print("Straight-through: {}".format(args.straight_through))
    print("Elapsed seconds: {:.6f}".format(elapsed))
    print("Wrote: {}".format(args.output))


if __name__ == "__main__":
    main()
