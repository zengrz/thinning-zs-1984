# File: zs_thinning.py
# Author: Zeng Ruizi (Rey)

import argparse
import os
import time

import numpy as np


tf = None
cv2 = None

NUM_CHANNELS = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply Zhang-Suen thinning to a thresholded grayscale image."
    )
    parser.add_argument("input", help="Path to the input image.")
    parser.add_argument(
        "-o",
        "--output",
        default="zs_thinned.png",
        help="Path for the thinned output image.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=180,
        help="Foreground threshold. Pixels greater than this value are treated as 1.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum Zhang-Suen iterations. The script stops earlier at convergence.",
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


def deletion_candidates(x, step):
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
    neighbour_count_between_2_and_6 = tf.logical_and(
        neighbour_count >= 2, neighbour_count <= 6
    )

    ordered = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
    transitions = [
        tf.logical_and(ordered[i] == 0, ordered[i + 1] == 1) for i in range(8)
    ]
    transition_count = tf.add_n([tf.cast(t, tf.float32) for t in transitions])
    transition_count_is_1 = tf.equal(transition_count, 1)

    if step == 1:
        # P2 * P4 * P6 == 0 and P4 * P6 * P8 == 0.
        condition_3 = tf.equal(p2 * p4 * p6, 0)
        condition_4 = tf.equal(p4 * p6 * p8, 0)
    else:
        # P2 * P4 * P8 == 0 and P2 * P6 * P8 == 0.
        condition_3 = tf.equal(p2 * p4 * p8, 0)
        condition_4 = tf.equal(p2 * p6 * p8, 0)

    return tf.logical_and(
        tf.logical_and(x > 0, interior_pixels(x)),
        tf.logical_and(
            neighbour_count_between_2_and_6,
            tf.logical_and(transition_count_is_1, tf.logical_and(condition_3, condition_4)),
        ),
    )


def zs_thinning_iteration(x):
    step_1_delete = deletion_candidates(x, step=1)
    x_after_step_1 = x * tf.cast(tf.logical_not(step_1_delete), tf.float32)

    step_2_delete = deletion_candidates(x_after_step_1, step=2)
    x_after_step_2 = x_after_step_1 * tf.cast(tf.logical_not(step_2_delete), tf.float32)

    changed = tf.logical_or(tf.reduce_any(step_1_delete), tf.reduce_any(step_2_delete))
    return x_after_step_2, changed


def run_thinning(binary_image, max_iterations):
    ensure_tensorflow()

    height, width = binary_image.shape
    if height < 3 or width < 3:
        raise ValueError("Image must be at least 3x3.")

    x_img_val = np.expand_dims(np.expand_dims(binary_image, 0), -1)

    x_img = tf.compat.v1.placeholder(
        tf.float32, shape=(1, height, width, NUM_CHANNELS), name="x"
    )

    def cond(i, x_in, changed):
        del x_in
        return tf.logical_and(tf.less(i, max_iterations), changed)

    def body(i, x_in, changed):
        del changed
        x_out, next_changed = zs_thinning_iteration(x_in)
        return tf.add(i, 1), x_out, next_changed

    iteration_count, result, _ = tf.while_loop(
        cond,
        body,
        [tf.constant(0), x_img, tf.constant(True)],
        shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([1, height, width, NUM_CHANNELS]),
            tf.TensorShape([]),
        ],
    )

    with tf.compat.v1.Session() as sess:
        start = time.time()
        iterations, output = sess.run(
            [iteration_count, result], feed_dict={x_img: x_img_val}
        )
        elapsed = time.time() - start

    return np.squeeze(output[0]), iterations, elapsed


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    binary_image = load_binary_image(args.input, args.threshold, args.crop)
    if args.threshold_output:
        ensure_opencv()
        cv2.imwrite(args.threshold_output, (binary_image * 255).astype(np.uint8))

    thinned, iterations, elapsed = run_thinning(binary_image, args.max_iterations)
    ensure_opencv()
    cv2.imwrite(args.output, (thinned * 255).astype(np.uint8))

    print("Input shape: {}".format(binary_image.shape))
    print("Iterations: {}".format(iterations))
    print("Elapsed seconds: {:.6f}".format(elapsed))
    print("Wrote: {}".format(args.output))


if __name__ == "__main__":
    main()
