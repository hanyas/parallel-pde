import jax
import jax.numpy as jnp

from parpde.objects import PDE, SEParams, IWParams
from parpde.kernels import squared_exponential
from parpde.kernels import once_integrated_wiener
from parpde.kernels import spatio_temporal

from parpde.utils import get_grid
from parpde.solvers import sequential_solver, parallel_solver

from parsmooth._base import FunctionalModel, MVNStandard
from parsmooth.methods import filter_smoother
from parsmooth.linearization import extended

import time


def burgers_equation(pde: PDE, s: jax.Array):
    flux = lambda u: 0.5 * u ** 2
    J = int(len(s) / 2) + 1
    u = s[:J - 1]
    u_jp1 = jnp.hstack((u[1:], pde.u_b))
    u_jm1 = jnp.hstack((pde.u_a, u[:-1]))
    f = (flux(u_jp1) - flux(u_jm1)) / (2.0 * pde.dx)
    du = s[J - 1:]
    return du + f


def u_0(u):
    return - jnp.sin(jnp.pi * u)


def measure_runtime(dt_list, nb_iter=10, nb_runs=25, parallel=True):
    runtime_mean = []
    runtime_median = []

    for k, dt in enumerate(dt_list):
        print(f"Task {k + 1} out of {len(dt_list)}", end="\n")

        dx = 0.1
        t_max = 1.5

        pde = PDE(
            a=-1.0, b=1.0,
            u_a=0.0, u_b=0.0, u_0=u_0,
            t=t_max, dx=dx, dt=dt
        )

        xs, ts = get_grid(pde)
        xs_size = len(xs)
        ts_size = len(ts)

        us = pde.u_0(xs[1:-1])
        dus = jnp.zeros(xs_size - 2)

        m0 = jnp.concat((us, dus))
        P0 = jax.scipy.linalg.block_diag(
            jnp.eye(xs_size - 2) * 1e-8,
            jnp.eye(xs_size - 2),
        )
        prior = MVNStandard(m0, P0)

        # Specify transition model
        spatial_params = SEParams(length_scale=2.0, signal_stddev=25.0)
        temporal_params = IWParams(noise_stddev=0.5)

        A, Q = spatio_temporal(
            spatial_params,
            temporal_params,
            squared_exponential,
            once_integrated_wiener,
            xs, dx, dt
        )
        transition_model = FunctionalModel(
            lambda u: A @ u,
            MVNStandard(jnp.zeros(Q.shape[0]), Q)
        )

        # Specify observation model
        observation_model = FunctionalModel(
            lambda s: burgers_equation(pde, s),
            MVNStandard(
                jnp.zeros(xs_size - 2),
                1e-8 * jnp.eye(xs_size - 2)
            )
        )

        observations = jnp.zeros((ts_size - 1, xs_size - 2))

        init_trajectory = filter_smoother(
            observations, prior, transition_model, observation_model, extended, None, False
        )

        @jax.jit
        def _solver(init_trajectory):
            if parallel:
                return parallel_solver(
                    observations,
                    prior,
                    transition_model,
                    observation_model,
                    init_trajectory,
                    nb_iter=nb_iter,
                )
            else:
                return sequential_solver(
                    observations,
                    prior,
                    transition_model,
                    observation_model,
                    init_trajectory,
                    nb_iter=nb_iter,
                )

        smoothed_trajectory = _solver(init_trajectory)
        jax.block_until_ready(smoothed_trajectory)

        runtimes = []
        for i in range(nb_runs):
            start = time.time()
            smoothed_trajectory = _solver(init_trajectory)
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
    parser.add_argument('--nb-runs', metavar='nb_runs', type=int, default=10, help='number of overall evaluations')

    args = parser.parse_args()

    dt_list = [0.08, 0.04, 0.02, 0.01, 0.008, 0.004, 0.002, 0.001, 0.0008, 0.0004]
    runtime_mean, runtime_median = \
        measure_runtime(dt_list, nb_iter=args.nb_iter, nb_runs=args.nb_runs, parallel=args.parallel)

    import pandas as pd

    dt_list_arr = jnp.array(dt_list)
    res_arr = jnp.vstack((dt_list_arr, runtime_median))
    df = pd.DataFrame(res_arr)

    if args.parallel:
        df.to_csv("./results/burgers_runtime_parallel.csv")
    else:
        df.to_csv("./results/burgers_runtime_sequential.csv")
