from typing import Callable, NamedTuple

import jax
from jax import numpy as jnp


class PDE(NamedTuple):
    a: float
    b: float
    u_a: float
    u_b: float
    u_0: Callable
    t: float
    dx: float
    dt: float


class Grid(NamedTuple):
    dx: float
    dt: float
    xs: jnp.ndarray
    ts: jnp.ndarray


# Squared Exponential
class SEParams(NamedTuple):
    length_scale: float
    signal_variance: float


# Stochastic Diff. Eq.
class SDEParams(NamedTuple):
    noise_variance: float
