from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from bayes_pde.objects import PDE, SEParams


@partial(jax.jit, static_argnums=1)
def get_gram_matrix(
    x: jax.Array,
    kernel_fn: Callable,
    kernel_params: SEParams
):
    return jax.vmap(
        lambda a: jax.vmap(
            lambda b: kernel_fn(b, a, **kernel_params._asdict())
        )(x)
    )(x)


def get_grid(pde: PDE):
    xs_size = int((pde.b - pde.a) / pde.dx)
    ts_size = int(pde.t / pde.dt)
    xs = jnp.linspace(pde.a, pde.b, xs_size + 1)
    ts = jnp.linspace(0.0, pde.t, ts_size + 1)
    return xs, ts
