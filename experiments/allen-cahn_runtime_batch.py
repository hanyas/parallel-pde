import jax
import jax.numpy as jnp

from bayes_pde.objects import PDE, SEParams, IWParams
from bayes_pde.kernels import squared_exponential
from bayes_pde.kernels import once_integrated_wiener
from bayes_pde.kernels import spatio_temporal

from bayes_pde.utils import get_grid
from bayes_pde.solvers import batch_solver

from newton_smoothers.base import FunctionalModel, MVNStandard
from newton_smoothers.utils import none_or_concat

import time


def allen_cahn_equation(pde: PDE, s: jax.Array):
    gamma_1 = 0.0001
    gamma_2 = 5.0
    J = int(len(s) / 2) + 1
    u = s[:J - 1]
    u_jp1 = jnp.hstack((u[1:], pde.u_b))
    u_jm1 = jnp.hstack((pde.u_a, u[:-1]))
    f = (
        - gamma_1 * (u_jp1 - 2. * u + u_jm1) / (pde.dx * pde.dx)
        + gamma_2 * (jnp.power(u, 3) - u)
    )
    du = s[J - 1:]
    return du + f


def u_0(u):
    return jnp.power(u, 2) * jnp.cos(jnp.pi * u)


def measure_runtime(dt_list, nb_iter=10, nb_runs=25):
    runtime_mean = []
    runtime_median = []

    for k, dt in enumerate(dt_list):
        print(f"Task {k + 1} out of {len(dt_list)}", end="\n")

        dx = 0.1
        t_max = 1.0

        pde = PDE(
            a=-1.0, b=1.0,
            u_a=-1.0, u_b=-1.0, u_0=u_0,
            t=t_max, dx=dx, dt=dt
        )

        xs, ts = get_grid(pde)
        xs_size = len(xs)
        ts_size = len(ts)

        # Specify prior
        us = pde.u_0(xs[1:-1])
        dus = jnp.zeros(xs_size - 2)

        m0 = jnp.hstack((us, dus))
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
            lambda s: allen_cahn_equation(pde, s),
            MVNStandard(
                jnp.zeros(xs_size - 2),
                1e-8 * jnp.eye(xs_size - 2)
            )
        )

        observations = jnp.zeros((ts_size - 1, xs_size - 2))

        init_trajectory = MVNStandard(
            mean=jnp.zeros((ts_size - 1, 2 * (xs_size - 2))),
            cov=jnp.repeat(
                jnp.eye(2 * (xs_size - 2))[jnp.newaxis, ...],
                ts_size - 1, axis=0
            ),
        )
        init_trajectory = none_or_concat(init_trajectory, prior, 1)

        @jax.jit
        def _solver(init_trajectory):
            return batch_solver(
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
    parser.add_argument('--nb-iter', metavar='nb_iter', type=int, default=10, help='number of solver iterations')
    parser.add_argument('--nb-runs', metavar='nb_runs', type=int, default=10, help='number of overall evaluations')

    args = parser.parse_args()

    dt_list = [0.08, 0.04, 0.02, 0.01, 0.008, 0.004, 0.002]
    runtime_mean, runtime_median = \
        measure_runtime(dt_list, nb_iter=args.nb_iter, nb_runs=args.nb_runs)

    import pandas as pd

    dt_list_arr = jnp.array(dt_list)
    res_arr = jnp.vstack((dt_list_arr, runtime_median))
    df = pd.DataFrame(res_arr)
    df.to_csv("./results/allen-cahn_runtime_batch.csv")