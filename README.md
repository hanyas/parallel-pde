# Parallel-in-Time Probabilistic Solutions for PDEs
Companion code for the paper "Parallel-in-Time Probabilistic Solutions for Time-Dependent Nonlinear Partial Differential Equations".

This code was written by [Hany Abdulsamad](https://github.com/hanyas), [Sahel Iqbal](https://github.com/Sahel13) and [Tripp Cator](https://github.com/DurwardCator)

## Installation
 
Create a conda environment
    
    conda create -n NAME python=3.11

then install a GPU-supported version of JAX
 
    pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Finally, install parallel-in-time Kalman-smoothers toolbox

    pip install git+https://github.com/hanyas/sqrt-parallel-smoothers

The Newton-smoother package is optional for second-order and regularized algorithms

    pip install git+https://github.com/hanyas/second-order-smoothers

Install `pytest` for testing, `matplotlib` for plots, and `py-pde` for rereference solutions.

 ## Examples
 
    python examples/burgers_parallel.py
