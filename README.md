# Zhang-Suen Edge Thinning Algorithm in TensorFlow

A TensorFlow implementation of "A fast parallel algorithm for thinning digital patterns" by T. Y. Zhang and C. Y. Suen, 1984.

## Usage

```bash
./run-zs-thinning.sh input.png -o thinned.png --threshold 180
```

Optional arguments:

```bash
./run-zs-thinning.sh input.png \
  --crop 512 0 512 512 \
  --threshold-output thresholded.png \
  --max-iterations 100
```

The script repeats Zhang-Suen iterations until no more pixels are removed, or until `--max-iterations` is reached. The default cap is `3`, matching the original script's limited-compute behavior.

Use `run-zs-thinning.sh` when running from this repository. It activates the CUDA library paths installed by `tensorflow[and-cuda]` before starting Python.

## Trainable neural edge thinning

`zs_thinning_neural.py` contains a small residual convolutional edge predictor followed by the differentiable Zhang-Suen thinning block from `zs_thinning_trainable.py`. Train it on generated thick-edge/thin-edge pairs:

```bash
.venv/bin/python zs_thinning_neural.py --train \
  --checkpoint-dir checkpoints/edge_thinner \
  --train-steps 1000 \
  --batch-size 8 \
  --image-size 128 \
  --iterations 5
```

Run the trained network on a thresholded edge image:

```bash
.venv/bin/python zs_thinning_neural.py input.png \
  --checkpoint-dir checkpoints/edge_thinner \
  -o neural_thinned.png \
  --threshold 180 \
  --iterations 5
```

The neural path writes both the thinned result and an `_edge_probability.png` image showing the CNN output before differentiable thinning.

## Dependencies

```bash
pip install numpy opencv-python "tensorflow[and-cuda]"
```
