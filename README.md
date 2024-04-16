# IEKS for PDEs

## Installation
1. Install JAX for the available hardware following [https://github.com/google/jax#installation](https://github.com/google/jax#installation).
2. Install [sqrt-parallel-smoothers](https://github.com/EEA-sensors/sqrt-parallel-smoothers) in editable mode (some function signatures have changed in recent JAX versions and may need to be edited).
3. Install `pytest` for testing and `matplotlib` for plots, if needed.

The first working example is in `examples/burgers.py`.

## Issues
- The square root version of the smoother is very slow. Could be the QR decomposition that's to blame.
- Using `parallel=True` in `iterated_smoothing` returns a different, and worse, solution. The tests for the parallel implementation in `sqrt-parallel-smoothers` all pass, so not sure why this is happening.
