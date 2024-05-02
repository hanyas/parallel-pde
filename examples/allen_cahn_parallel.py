import jax
import jax.numpy as jnp

from bayes_pde.objects import PDE, SEParams, IWParams
from bayes_pde.kernels import squared_exponential
from bayes_pde.kernels import once_integrated_wiener
from bayes_pde.kernels import spatio_temporal

from bayes_pde.utils import get_grid
from bayes_pde.solvers import parallel_solver
from allen_cahn_high_fidelity import run_high_fidelity

from parsmooth._base import FunctionalModel, MVNStandard
from parsmooth.methods import filter_smoother
from parsmooth.linearization import extended

import time
import matplotlib.pyplot as plt


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


if __name__ == "__main__":

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cuda")

    dx = 0.01
    dt = 0.01
    t_max = 1.0

    pde = PDE(
        a=-1.0, b=1.0,
        u_a=-1.0, u_b=-1.0, u_0=u_0,
        t=t_max, dx=dx, dt=dt
    )

    xs, ts = get_grid(pde)
    xs_size = len(xs)
    ts_size = len(ts)

    # Specify the initial mean and covariance.
    us = pde.u_0(xs[1:-1])
    dus = jnp.zeros(xs_size - 2)

    m0 = jnp.hstack((us, dus))
    P0 = jax.scipy.linalg.block_diag(
        jnp.eye(xs_size - 2) * 1e-8,
        jnp.eye(xs_size - 2),
    )
    prior = MVNStandard(m0, P0)

    spatial_params = SEParams(length_scale=1.0, signal_stddev=1.0)
    temporal_params = IWParams(noise_stddev=1.0)

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

    init_trajectory = filter_smoother(
        observations,
        prior,
        transition_model,
        observation_model,
        extended,
        None,
        False
    )

    @jax.jit
    def _solver(init_trajectory):
        return parallel_solver(
            observations,
            prior,
            transition_model,
            observation_model,
            init_trajectory,
            extended,
            nb_iter=50,
        )

    start = time.time()
    smoothed_trajectory = _solver(init_trajectory)
    jax.block_until_ready(smoothed_trajectory)
    end = time.time()
    print("time: ", end - start)

    # Get the posterior mean and variance of the solution
    mask = jnp.kron(jnp.array([1.0, 0.0]), jnp.eye(xs_size - 2))
    us_par = jnp.einsum("ij, tj -> ti", mask, smoothed_trajectory.mean)
    Ps_par = jnp.einsum("ij, tjk, kl -> til", mask, smoothed_trajectory.cov, mask.T)
    Ps_par = jnp.diagonal(Ps_par, axis1=1, axis2=2)

    # Frame have to be adjusted according to the discretization
    xs_hf, ts_hf, us_hf = run_high_fidelity(dt, dx, t_max, -1.0, 1.0)
    hf_frames = [0, 50, 75, 100]
    ieks_frames = [0, 50, 75, 100]

    fig = plt.figure(figsize=(12, 9))
    for idx, frame_idx in enumerate(ieks_frames):
        ax = fig.add_subplot(1, 4, idx + 1)
        ax.fill_between(
            xs[1:-1],
            us_par[frame_idx, :] - 2.0 * jnp.sqrt(Ps_par[frame_idx, :]),
            us_par[frame_idx, :] + 2.0 * jnp.sqrt(Ps_par[frame_idx, :]),
            label="Confidence",
        )
        ax.plot(xs[1:-1], us_par[frame_idx, :], "r-", linewidth=2.0, label="IEKS")
        ax.plot(xs_hf, us_hf[hf_frames[idx]], 'k--', linewidth=2.5, label='HF')
        if idx == 0:
            ax.set_ylabel("$u$", fontsize="large")
        ax.set_title(r"time = {:.2f}".format(ts[frame_idx]), fontsize="large")
        ax.legend(fontsize="large", loc="lower right")
    plt.show()


    # import pandas as pd
    #
    # cwd = "/tmp/pycharm_project_689/experiments/plots"
    #
    # for idx, frame_idx in enumerate(hf_frames):
    #     out = pd.DataFrame({
    #         "x": xs_hf,
    #         "y": us_hf[frame_idx, :],
    #     })
    #     file_name = f"{cwd}/allen-cahn_ref_time_{ts[frame_idx]}.csv"
    #     out.to_csv(file_name, index=False)
    #
    #
    # for idx, frame_idx in enumerate(ieks_frames):
    #     out = pd.DataFrame({
    #         "x": xs[1:-1],
    #         "y": us_par[frame_idx, :],
    #         "err": 2.0 * jnp.sqrt(Ps_par[frame_idx, :]),
    #     })
    #     file_name = f"{cwd}/allen-cahn_sol_time_{ts[frame_idx]}.csv"
    #     out.to_csv(file_name, index=False)