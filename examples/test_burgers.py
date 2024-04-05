import jax
import jax.numpy as jnp
from burgers import (
    PDE,
    construct_gram_matrix,
    construct_grid,
    pseudo_observation_fn,
    squared_exponential_kernel,
)


def test_construct_gram_matrix():
    x = jnp.array([0.0, 1.0, 2.0, 3.0])
    kernel_fn = squared_exponential_kernel
    sigma = 1.0
    ell = 1.0
    C = construct_gram_matrix(x, kernel_fn, sigma, ell)
    assert C.shape == (4, 4)
    assert C[0, 0] == 1.0
    assert C[3, 2] == kernel_fn(3.0, 2.0, sigma, ell)


def test_pseudo_observation_fn_derivative():
    """Test the Jacobian of the pseudo-observation function."""
    flux = lambda u: u**2 / 2
    u_0 = lambda u: -jnp.sin(jnp.pi * u)
    burgers = PDE(-1.0, 1.0, 0.0, 0.0, u_0, 1.5, 0.1, 0.05, flux)
    x, _ = construct_grid(burgers)
    J = len(x) - 1
    us = burgers.u_0(x[1:-1])
    dus = jnp.zeros(J - 1)
    m0 = jnp.concat((us, dus))
    jac = jax.jacfwd(pseudo_observation_fn, argnums=1)(burgers, m0)
    jac = jac[:, : J - 1]
    assert jnp.allclose(jnp.diagonal(jac, offset=-1), -m0[: J - 2] / (2.0 * burgers.dx))
    assert jnp.allclose(jnp.diagonal(jac, offset=1), m0[1 : J - 1] / (2.0 * burgers.dx))
