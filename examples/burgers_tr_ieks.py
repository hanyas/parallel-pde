import jax
import jax.numpy as jnp

from parpde.objects import PDE, SEParams, IWParams
from parpde.kernels import squared_exponential
from parpde.kernels import once_integrated_wiener
from parpde.kernels import spatio_temporal

from burgers_cole_hopf import run_cole_hopf

from parpde.utils import get_grid
from parpde.solvers import sequential_solver_with_trust_region

from newton_smoothers.recursive.kalman import init_filtering, init_smoothing
from newton_smoothers.base import FunctionalModel, MVNStandard
from newton_smoothers.approximation import extended
from newton_smoothers.utils import mvn_logpdf

import time
import matplotlib.pyplot as plt


def burgers_equation(pde: PDE, s: jax.Array):
    nu = 1e-2  # 0.005 / jnp.pi
    flux = lambda u: 0.5 * u ** 2
    J = int(len(s) / 2) + 1
    u = s[:J - 1]
    u_jp1 = jnp.hstack((u[1:], pde.u_b))
    u_jm1 = jnp.hstack((pde.u_a, u[:-1]))
    f = (
        (flux(u_jp1) - flux(u_jm1)) / (2.0 * pde.dx)
        - nu * (u_jp1 - 2. * u + u_jm1) / (pde.dx * pde.dx)
    )
    du = s[J - 1:]
    return du + f


def u_0(u):
    return - jnp.sin(jnp.pi * u)


if __name__ == "__main__":

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")

    dx = 0.01
    dt = 0.01
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

    m0 = jnp.hstack((us, dus))
    P0 = jax.scipy.linalg.block_diag(
        jnp.eye(xs_size - 2) * 1e-8,
        jnp.eye(xs_size - 2),
    )
    prior = MVNStandard(m0, P0)

    # spatial_params = SEParams(length_scale=0.95, signal_stddev=3.65)
    # temporal_params = IWParams(noise_stddev=0.065)

    spatial_params = SEParams(length_scale=1.0, signal_stddev=25.0)
    temporal_params = IWParams(noise_stddev=25.0)

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

    init_trajectory = init_filtering(
        observations,
        prior,
        transition_model,
        observation_model,
        extended
    )
    init_trajectory = init_smoothing(
        transition_model,
        init_trajectory,
        extended
    )

    @jax.jit
    def gauss_solver(init_trajectory):
        return sequential_solver_with_trust_region(
            observations,
            prior,
            transition_model,
            observation_model,
            init_trajectory,
            extended,
            nb_iter=25,
        )

    start = time.time()
    gauss_trajectory = gauss_solver(init_trajectory)
    jax.block_until_ready(gauss_trajectory)
    end = time.time()
    print("time: ", end - start)

    filtered_trajectory = init_filtering(
        observations,
        prior,
        transition_model,
        observation_model,
        extended,
    )

    mask = jnp.kron(jnp.array([1.0, 0.0]), jnp.eye(xs_size - 2))

    # Get the posterior mean and variance of the solution
    us_filt = jnp.einsum("ij, tj -> ti", mask, filtered_trajectory.mean)
    Ps_filt = jnp.einsum("ij, tjk, kl -> til", mask, filtered_trajectory.cov, mask.T)
    var_filt = jnp.diagonal(Ps_filt, axis1=1, axis2=2)

    us_gauss = jnp.einsum("ij, tj -> ti", mask, gauss_trajectory.mean)
    Ps_gauss = jnp.einsum("ij, tjk, kl -> til", mask, gauss_trajectory.cov, mask.T)
    var_gauss = jnp.diagonal(Ps_gauss, axis1=1, axis2=2)

    # Frame have to be adjusted according to the discretization
    xs_ch, ts_ch, us_ch = run_cole_hopf(dt, dx, t_max, -1.0, 1.0)
    ch_frames = [0, int(t_max / dt / 4), int(2 * t_max / dt / 4), int(3 * t_max / dt / 4)]
    ieks_frames = [0, int(t_max / dt / 4), int(2 * t_max / dt / 4), int(3 * t_max / dt / 4)]

    fig = plt.figure(figsize=(12, 9))
    for idx, frame_idx in enumerate(ieks_frames):
        ax = fig.add_subplot(1, 4, idx + 1)
        # ax.fill_between(
        #     xs[1:-1],
        #     us_filt[frame_idx, :] - 2.0 * jnp.sqrt(var_filt[frame_idx, :]),
        #     us_filt[frame_idx, :] + 2.0 * jnp.sqrt(var_filt[frame_idx, :]),
        #     label="Confidence",
        #     color='b',
        #     alpha=0.25
        # )
        ax.fill_between(
            xs[1:-1],
            us_gauss[frame_idx, :] - 2.0 * jnp.sqrt(var_gauss[frame_idx, :]),
            us_gauss[frame_idx, :] + 2.0 * jnp.sqrt(var_gauss[frame_idx, :]),
            label="Confidence",
            color='r',
            alpha=0.25
        )
        # ax.plot(xs[1:-1], us_filt[frame_idx, :], "b-", linewidth=2.0, label="EKF")
        ax.plot(xs[1:-1], us_gauss[frame_idx, :], "r-", linewidth=2.0, label="IEKS")
        ax.plot(xs_ch, us_ch[ch_frames[idx], :], 'k--', linewidth=2.5, label='CH')
        if idx == 0:
            ax.set_ylabel("s$u$", fontsize="large")
        ax.set_title(r"time = {:.2f}".format(ts[frame_idx]), fontsize="large")
        ax.legend(fontsize="large", loc="upper right")
        ax.set_ylim([-1.55, 1.55])
    plt.show()

    # print("EKF Error: ", jnp.mean(jnp.linalg.norm(us_filt - us_ch[:, 1:-1], axis=0)))
    # print("IEKS Error: ", jnp.mean(jnp.linalg.norm(us_gauss - us_ch[:, 1:-1], axis=0)))
    #
    # print("EKF ll: ", jax.vmap(mvn_logpdf)(us_ch[:, 1:-1], us_filt, Ps_filt).sum())
    # print("IEKS ll: ", jax.vmap(mvn_logpdf)(us_ch[:, 1:-1], us_gauss, Ps_gauss).sum())

    # import pandas as pd
    #
    # # cwd = "/tmp/pycharm_project_689/experiments/plots"
    # cwd = "../experiments/plots"
    #
    # for idx, frame_idx in enumerate(ch_frames):
    #     out = pd.DataFrame({
    #         "x": xs,
    #         "y": us_ch[frame_idx, :],
    #     })
    #     file_name = f"{cwd}/burgers_ref_time_{ts[frame_idx]}_dx_0025.csv"
    #     out.to_csv(file_name, index=False)
    #
    # for idx, frame_idx in enumerate(ieks_frames):
    #     out = pd.DataFrame({
    #         "x": xs[1:-1],
    #         "y": us_gauss[frame_idx, :],
    #         "err": 2.0 * jnp.sqrt(var_gauss[frame_idx, :]),
    #     })
    #     file_name = f"{cwd}/burgers_sol_time_{ts[frame_idx]}_dx_0025.csv"
    #     out.to_csv(file_name, index=False)
