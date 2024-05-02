from parsmooth.linearization import extended, cubature
from parsmooth.methods import iterated_smoothing
from newton_smoothers import iterated_batch_gauss_newton_smoother


def batch_solver(
    observations,
    prior,
    transition_model,
    observation_model,
    init_trajectory,
    nb_iter=10,
):
    return iterated_batch_gauss_newton_smoother(
        init_trajectory.mean,
        observations,
        prior,
        transition_model,
        observation_model,
        nb_iter=10,
    )[0]


def sequential_solver(
    observations,
    prior,
    transition_model,
    observation_model,
    init_trajectory,
    nb_iter=10,
    return_loglikelihood=False
):
    return iterated_smoothing(
        observations,
        prior,
        transition_model,
        observation_model,
        cubature,
        init_trajectory,
        parallel=False,
        criterion=lambda i, *_: i < nb_iter,
        return_loglikelihood=return_loglikelihood,
    )


def parallel_solver(
    observations,
    prior,
    transition_model,
    observation_model,
    init_trajectory,
    nb_iter=10,
    return_loglikelihood=False
):
    return iterated_smoothing(
        observations,
        prior,
        transition_model,
        observation_model,
        cubature,
        init_trajectory,
        parallel=True,
        criterion=lambda i, *_: i < nb_iter,
        return_loglikelihood=return_loglikelihood,
    )