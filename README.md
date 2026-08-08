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

## Dependencies

```bash
pip install numpy opencv-python "tensorflow[and-cuda]"
```
