# File: zs_thinning_neural.py
# Author: Zeng Ruizi (Rey)
#
# Convolutional edge thinning network. The CNN predicts an edge probability map,
# then the differentiable Zhang-Suen block from zs_thinning_trainable.py thins it.

import argparse
import os
import time

import numpy as np

import zs_thinning_trainable as zst


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or run a CNN followed by differentiable Zhang-Suen thinning."
    )
    parser.add_argument("input", nargs="?", help="Path to the input image for inference.")
    parser.add_argument(
        "-o",
        "--output",
        default="zs_thinned_neural.png",
        help="Path for the neural thinned output image.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=180,
        help="Optional foreground threshold applied before inference.",
    )
    parser.add_argument(
        "--no-threshold",
        action="store_true",
        help="Feed normalized grayscale pixels instead of thresholding input.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Fixed differentiable thinning iterations to unroll.",
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
        help="Use hard binary thinning in the forward pass with soft gradients backward.",
    )
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("Y", "X", "HEIGHT", "WIDTH"),
        help="Optional crop rectangle before inference.",
    )
    parser.add_argument(
        "--gpu",
        help="Optional CUDA_VISIBLE_DEVICES value, for example '0' or '2'.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train on generated thick-edge/thin-edge pairs.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/edge_thinner",
        help="Directory for neural network checkpoints.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=1000,
        help="Number of synthetic training batches.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Square synthetic training image size.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print training loss every N steps.",
    )
    return parser.parse_args()


def load_grayscale_image(path, threshold=None, crop=None):
    zst.ensure_opencv()

    image = zst.cv2.imread(path, zst.cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not read input image: {}".format(path))

    if crop is not None:
        y, x, height, width = crop
        image = image[y : y + height, x : x + width]
        if image.size == 0:
            raise ValueError("Crop produced an empty image.")

    if threshold is not None:
        return (image > threshold).astype(np.float32)
    return image.astype(np.float32) / 255.0


def conv2d(x, filters, kernel_size, activation=None, name=None):
    layer = zst.tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        padding="same",
        activation=activation,
        kernel_initializer=zst.tf.keras.initializers.GlorotUniform(),
        name=name,
    )
    return layer(x)


def residual_block(x, filters, name):
    with zst.tf.compat.v1.variable_scope(name):
        skip = x
        if x.shape[-1] != filters:
            skip = conv2d(x, filters, 1, activation=None, name="skip")

        h = conv2d(x, filters, 3, activation="relu", name="conv1")
        h = conv2d(h, filters, 3, activation=None, name="conv2")
        return zst.tf.nn.relu(h + skip, name="relu")


def build_edge_thinning_network(x, iterations, sharpness, straight_through=False):
    """Predict a clean edge map, then thin it with the differentiable ZS block."""
    with zst.tf.compat.v1.variable_scope("edge_thinner", reuse=zst.tf.compat.v1.AUTO_REUSE):
        h = conv2d(x, 16, 3, activation="relu", name="stem")
        h = residual_block(h, 32, name="residual1")
        h = residual_block(h, 32, name="residual2")
        h = conv2d(h, 16, 3, activation="relu", name="head")
        logits = conv2d(h, 1, 1, activation=None, name="edge_logits")
        edge_prob = zst.tf.sigmoid(logits, name="edge_probability")
        thinned = zst.soft_zs_thinning(
            edge_prob,
            iterations=iterations,
            sharpness=sharpness,
            straight_through=straight_through,
        )
    return thinned, edge_prob, logits


def binary_cross_entropy(y_true, y_pred, epsilon=1e-6):
    y_pred = zst.tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    return -zst.tf.reduce_mean(
        y_true * zst.tf.math.log(y_pred)
        + (1.0 - y_true) * zst.tf.math.log(1.0 - y_pred)
    )


def dice_loss(y_true, y_pred, epsilon=1e-6):
    axes = [1, 2, 3]
    intersection = zst.tf.reduce_sum(y_true * y_pred, axis=axes)
    denominator = zst.tf.reduce_sum(y_true + y_pred, axis=axes)
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - zst.tf.reduce_mean(dice)


def draw_random_edge_pair(image_size):
    zst.ensure_opencv()

    target = np.zeros((image_size, image_size), dtype=np.uint8)
    shape_count = np.random.randint(3, 9)
    margin = max(4, image_size // 16)

    for _ in range(shape_count):
        shape_type = np.random.randint(0, 4)
        color = 255
        if shape_type == 0:
            pt1 = (
                np.random.randint(margin, image_size - margin),
                np.random.randint(margin, image_size - margin),
            )
            pt2 = (
                np.random.randint(margin, image_size - margin),
                np.random.randint(margin, image_size - margin),
            )
            zst.cv2.line(target, pt1, pt2, color, 1, lineType=zst.cv2.LINE_AA)
        elif shape_type == 1:
            pt1 = (
                np.random.randint(margin, image_size // 2),
                np.random.randint(margin, image_size // 2),
            )
            pt2 = (
                np.random.randint(image_size // 2, image_size - margin),
                np.random.randint(image_size // 2, image_size - margin),
            )
            zst.cv2.rectangle(target, pt1, pt2, color, 1, lineType=zst.cv2.LINE_AA)
        elif shape_type == 2:
            center = (
                np.random.randint(margin, image_size - margin),
                np.random.randint(margin, image_size - margin),
            )
            radius = np.random.randint(max(3, image_size // 16), max(4, image_size // 4))
            zst.cv2.circle(target, center, radius, color, 1, lineType=zst.cv2.LINE_AA)
        else:
            points = np.random.randint(
                margin, image_size - margin, size=(np.random.randint(3, 7), 2)
            )
            zst.cv2.polylines(target, [points], False, color, 1, lineType=zst.cv2.LINE_AA)

    target = (target > 64).astype(np.float32)
    kernel_size = np.random.choice([3, 5, 7])
    kernel = zst.cv2.getStructuringElement(
        zst.cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    thick = zst.cv2.dilate(target, kernel, iterations=np.random.randint(1, 3))

    if np.random.rand() < 0.5:
        thick = zst.cv2.GaussianBlur(thick, (3, 3), 0)
    noise = np.random.normal(0.0, 0.04, thick.shape).astype(np.float32)
    thick = np.clip(thick + noise, 0.0, 1.0)
    return thick[:, :, None].astype(np.float32), target[:, :, None].astype(np.float32)


def synthetic_edge_batch(batch_size, image_size):
    inputs = []
    targets = []
    for _ in range(batch_size):
        x, y = draw_random_edge_pair(image_size)
        inputs.append(x)
        targets.append(y)
    return np.stack(inputs, axis=0), np.stack(targets, axis=0)


def train_neural_network(
    checkpoint_dir,
    steps,
    batch_size,
    image_size,
    iterations,
    sharpness,
    straight_through,
    learning_rate,
    log_every,
):
    zst.ensure_tensorflow()
    zst.ensure_opencv()
    os.makedirs(checkpoint_dir, exist_ok=True)

    x_img = zst.tf.compat.v1.placeholder(
        zst.tf.float32, shape=(None, image_size, image_size, zst.NUM_CHANNELS), name="x"
    )
    y_img = zst.tf.compat.v1.placeholder(
        zst.tf.float32, shape=(None, image_size, image_size, zst.NUM_CHANNELS), name="y"
    )
    thinned, edge_prob, _ = build_edge_thinning_network(
        x_img,
        iterations=iterations,
        sharpness=sharpness,
        straight_through=straight_through,
    )

    thinning_loss = binary_cross_entropy(y_img, thinned) + dice_loss(y_img, thinned)
    edge_loss = binary_cross_entropy(y_img, edge_prob)
    loss = thinning_loss + 0.25 * edge_loss
    train_op = zst.tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
    saver = zst.tf.compat.v1.train.Saver(max_to_keep=3)

    with zst.tf.compat.v1.Session() as sess:
        sess.run(zst.tf.compat.v1.global_variables_initializer())
        start = time.time()
        for step in range(1, steps + 1):
            batch_x, batch_y = synthetic_edge_batch(batch_size, image_size)
            _, loss_value = sess.run(
                [train_op, loss], feed_dict={x_img: batch_x, y_img: batch_y}
            )
            if step == 1 or step % log_every == 0 or step == steps:
                print("Step {}/{} loss {:.6f}".format(step, steps, loss_value))

        checkpoint_path = saver.save(sess, os.path.join(checkpoint_dir, "model.ckpt"))
        elapsed = time.time() - start

    return checkpoint_path, elapsed


def run_neural_thinning(image, checkpoint_dir, iterations, sharpness, straight_through):
    zst.ensure_tensorflow()

    height, width = image.shape
    if height < 3 or width < 3:
        raise ValueError("Image must be at least 3x3.")

    x_img_val = np.expand_dims(np.expand_dims(image, 0), -1)
    x_img = zst.tf.compat.v1.placeholder(
        zst.tf.float32, shape=(1, height, width, zst.NUM_CHANNELS), name="x"
    )
    thinned, edge_prob, _ = build_edge_thinning_network(
        x_img,
        iterations=iterations,
        sharpness=sharpness,
        straight_through=straight_through,
    )
    saver = zst.tf.compat.v1.train.Saver()
    checkpoint = zst.tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint is None:
        raise ValueError("No checkpoint found in {}".format(checkpoint_dir))

    with zst.tf.compat.v1.Session() as sess:
        saver.restore(sess, checkpoint)
        start = time.time()
        thinned_out, edge_out = sess.run(
            [thinned, edge_prob], feed_dict={x_img: x_img_val}
        )
        elapsed = time.time() - start

    return np.squeeze(thinned_out[0]), np.squeeze(edge_out[0]), elapsed, checkpoint


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.train:
        checkpoint_path, elapsed = train_neural_network(
            checkpoint_dir=args.checkpoint_dir,
            steps=args.train_steps,
            batch_size=args.batch_size,
            image_size=args.image_size,
            iterations=args.iterations,
            sharpness=args.sharpness,
            straight_through=args.straight_through,
            learning_rate=args.learning_rate,
            log_every=args.log_every,
        )
        print("Elapsed seconds: {:.6f}".format(elapsed))
        print("Wrote checkpoint: {}".format(checkpoint_path))
        return

    if args.input is None:
        raise ValueError("An input image is required unless --train is used.")

    threshold = None if args.no_threshold else args.threshold
    image = load_grayscale_image(args.input, threshold, args.crop)
    thinned, edge_prob, elapsed, checkpoint = run_neural_thinning(
        image,
        checkpoint_dir=args.checkpoint_dir,
        iterations=args.iterations,
        sharpness=args.sharpness,
        straight_through=args.straight_through,
    )
    zst.ensure_opencv()
    zst.cv2.imwrite(args.output, (thinned * 255).astype(np.uint8))
    edge_output = os.path.splitext(args.output)[0] + "_edge_probability.png"
    zst.cv2.imwrite(edge_output, (edge_prob * 255).astype(np.uint8))

    print("Input shape: {}".format(image.shape))
    print("Iterations: {}".format(args.iterations))
    print("Sharpness: {:.6f}".format(args.sharpness))
    print("Straight-through: {}".format(args.straight_through))
    print("Checkpoint: {}".format(checkpoint))
    print("Elapsed seconds: {:.6f}".format(elapsed))
    print("Wrote: {}".format(args.output))
    print("Wrote: {}".format(edge_output))


if __name__ == "__main__":
    main()
