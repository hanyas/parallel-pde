# Parallel-in-Time Probabilistic Solutions for PDEs

## Installation
 
Create a conda environment
    
    conda create -n NAME python=3.11

then install a GPU-supported version of JAX
 
    pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
     
To avoide numerical problems downgrade to an earlier JAX-lib version
 
    pip install jaxlib==0.4.20+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Finally, install parallel-in-time Kalman-smoothers toolbox

    pip install git+https://github.com/hanyas/sqrt-parallel-smoothers

Install `pytest` for testing and `matplotlib` for plots, if needed.

 ## Examples
 
    python examples/burgers.py
