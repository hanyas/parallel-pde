from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from parsmooth._base import FunctionalModel, MVNStandard
from parsmooth.linearization import extended
from parsmooth.methods import filter_smoother, iterated_smoothing

import time


class PDE(NamedTuple):
    a: float
    b: float
    u_a: float
    u_b: float
    u_0: Callable
    T: float
    dx: float
    dt: float
    flux: Callable


def construct_grid(pde: PDE):
    J = int((pde.b - pde.a) / pde.dx)
    N = int(pde.T / pde.dt)
    x_grid = jnp.linspace(pde.a, pde.b, J + 1)
    t_grid = jnp.linspace(0.0, pde.T, N + 1)
    return x_grid, t_grid


@jax.jit
def squared_exponential_kernel(x1: float, x2: float, sigma: float, ell: float):
    return sigma**2 * jnp.exp(-0.5 * (x1 - x2) ** 2 / ell**2)


@partial(jax.jit, static_argnums=1)
def construct_gram_matrix(x: jax.Array, kernel_fn: Callable, *args):
    return jax.vmap(lambda x1: jax.vmap(lambda x2: kernel_fn(x1, x2, *args))(x))(x)


def get_transition_model(
    pde: PDE, x_grid: jax.Array, q: float = 0.5, sigma: float = 25, nell: float = 2.0
):
    dt = pde.dt
    dx = pde.dx

    # IWP-1 prior.
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

    ell = nell * dx
    C_Jm1 = construct_gram_matrix(x_grid[1:-1], squared_exponential_kernel, sigma, ell)

    J = len(x_grid) - 1
    I_Jm1 = jnp.eye(J - 1)

    A = jnp.kron(A_dt, I_Jm1)
    Q = jnp.kron(Q_dt, C_Jm1)
    return FunctionalModel(
        lambda u: A @ u,
        MVNStandard(jnp.zeros(Q.shape[0]), Q)
    )


def burgers_equation_fn(pde: PDE, s: jax.Array):
    J = int(len(s) / 2) + 1
    u = s[:J - 1]
    u_jp1 = jnp.concat((u[1:], jnp.array([pde.u_b])))
    u_jm1 = jnp.concat((jnp.array([pde.u_a]), u[:-1]))
    f = (pde.flux(u_jp1) - pde.flux(u_jm1)) / (2.0 * pde.dx)
    du = s[J - 1:]
    return du + f


def flux(u):
    return 0.5 * u**2


def u_0(u):
    return -jnp.sin(jnp.pi * u)


def measure_runtime(dt_list, nb_iter=10, nb_runs=25, parallel=True):
    runtime_mean = []
    runtime_median = []

    for k, dt in enumerate(dt_list):
        print(f"Task {k + 1} out of {len(dt_list)}", end="\n")

        max_t = 1.0
        min_x = -1.0
        max_x = 1.0

        dx = 0.01

        pde = PDE(min_x, max_x, 0.0, 0.0, u_0, max_t, dx, dt, flux)

        x_grid, t_grid = construct_grid(pde)
        x_size = len(x_grid) - 1
        t_size = len(t_grid) - 1

        us = pde.u_0(x_grid[1:-1])
        dus = jnp.zeros(x_size - 1)

        m0 = jnp.concat((us, dus))
        P0 = 1e-2 * jax.scipy.linalg.block_diag(
            jnp.eye(x_size - 1) * 0.0,
            jnp.eye(x_size - 1),
        )
        prior = MVNStandard(m0, P0)

        transition_model = get_transition_model(pde, x_grid)

        observation_model = FunctionalModel(
            lambda s: burgers_equation_fn(pde, s),
            MVNStandard(jnp.zeros(x_size - 1), jnp.eye(x_size - 1) * 1e-8)
        )

        observations = jnp.zeros((t_size, x_size - 1))

        init_nominal_trajectory = filter_smoother(
            observations, prior, transition_model, observation_model, extended, None, False
        )

        @jax.jit
        def pde_solver(init_trajectory, nb_iter):
            result = iterated_smoothing(
                observations,
                prior,
                transition_model,
                observation_model,
                extended,
                init_trajectory,
                parallel=parallel,
                criterion=lambda i, *_: i < nb_iter,
                return_loglikelihood=False,
            )
            return result

        smoothed_trajectory = pde_solver(init_nominal_trajectory, nb_iter)
        jax.block_until_ready(smoothed_trajectory)

        runtimes = []
        for i in range(nb_runs):
            start = time.time()
            smoothed_trajectory = pde_solver(init_nominal_trajectory, nb_iter)
            jax.block_until_ready(smoothed_trajectory)
            end = time.time()
            runtimes.append(end - start)

            print(f"run {i + 1} out of {nb_runs}", end="\n")

        runtime_mean.append(jnp.mean(jnp.array(runtimes)))
        runtime_median.append(jnp.median(jnp.array(runtimes)))
    return jnp.array(runtime_mean), jnp.array(runtime_median)


if __name__ == "__main__":

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cuda")

    import argparse

    parser = argparse.ArgumentParser(description='Configer PDE solvers')
    parser.add_argument('--parallel', action=argparse.BooleanOptionalAction, help='use parallel or sequential solver')
    parser.add_argument('--nb-iter', metavar='nb_iter', type=int, default=10, help='number of solver iterations')
    parser.add_argument('--nb-runs', metavar='nb_runs', type=int, default=25, help='number of overall evaluations')

    args = parser.parse_args()

    dt_list = [0.08, 0.04, 0.02, 0.01, 0.008, 0.004, 0.002, 0.001, 0.0008, 0.0004]
    runtime_mean, runtime_median = \
        measure_runtime(dt_list, nb_iter=args.nb_iter, nb_runs=args.nb_runs, parallel=args.parallel)

    import pandas as pd

    dt_list_arr = jnp.array(dt_list)
    res_arr = jnp.vstack((dt_list_arr, runtime_median))
    df = pd.DataFrame(res_arr)

    if args.parallel:
        df.to_csv("burgers_runtime_parallel.csv")
    else:
        df.to_csv("burgers_runtime_sequential.csv")
