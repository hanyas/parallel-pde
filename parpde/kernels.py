from typing import Callable

import jax
import jax.numpy as jnp

from parpde.objects import SEParams, IWParams
from parpde.utils import get_gram_matrix


@jax.jit
def squared_exponential(
    x: float, y: float,
    length_scale: float,
    signal_stddev: float
):
    return signal_stddev**2 * jnp.exp(-0.5 * (x - y)**2 / length_scale**2)


def once_integrated_wiener(
    time_step: float,
    noise_stddev: float
):
    dt = time_step
    q = noise_stddev

    A_dt = jnp.array(
        [
            [1, dt],
            [0, 1]
        ]
    )
    Q_dt = q * jnp.array(
        [
            [dt**3 / 3, dt**2 / 2],
            [dt**2 / 2, dt]
        ]
    )
    return A_dt, Q_dt


def twice_integrated_wiener(
    time_step: float,
    noise_stddev: float
):
    dt = time_step
    q = noise_stddev

    A_dt = jnp.array(
        [
            [1, dt, dt**2/2],
            [0, 1,  dt],
            [0, 0,  1]
        ]
    )
    Q_dt = q * jnp.array(
        [
            [dt**5/20, dt**4/8, dt**3/6],
            [dt**4/8,  dt**3/3, dt**2/2],
            [dt**3/6,  dt**2/2, dt]
        ]
    )
    return A_dt, Q_dt


def spatio_temporal(
    spatial_params: SEParams,
    temporal_params: IWParams,
    spatial_kernel: Callable,
    temporal_kernel: Callable,
    spatial_grid: jnp.ndarray,
    spatial_diff: float,
    time_step: float
):
    A_dt, Q_dt = temporal_kernel(
        time_step,
        temporal_params.noise_stddev
    )

    I_Jm1 = jnp.eye(len(spatial_grid) - 2)
    C_Jm1 = get_gram_matrix(
        x=spatial_grid[1:-1],
        kernel_fn=spatial_kernel,
        kernel_params=SEParams(
            spatial_params.length_scale * spatial_diff,
            spatial_params.signal_stddev
        )
    )

    A = jnp.kron(A_dt, I_Jm1)
    Q = jnp.kron(Q_dt, C_Jm1)
    return A, Q