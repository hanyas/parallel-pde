# IEKS for PDEs

## Installation
1. Install JAX for the available hardware following [https://github.com/google/jax#installation](https://github.com/google/jax#installation).
2. Install [sqrt-parallel-smoothers](https://github.com/EEA-sensors/sqrt-parallel-smoothers) in editable mode (some function signatures have changed in recent JAX versions and may need to be edited).
3. Install `pytest` for testing and `matplotlib` for plots, if needed.

The first working example is in `examples/burgers.py`.

## TODO
- [ ] Try the square-root Kalman filter for Burger's equation. Currently dense grids produce NaNs.
