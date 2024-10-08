from parsmooth.linearization import extended
from parsmooth.methods import iterated_smoothing

from newton_smoothers import trust_region_iterated_batch_gauss_newton_smoother
from newton_smoothers import trust_region_iterated_recursive_gauss_newton_smoother
from newton_smoothers import trust_region_iterated_recursive_newton_smoother


def batch_solver(
    observations,
    prior,
    transition_model,
    observation_model,
    init_trajectory,
    nb_iter=10,
):
    return trust_region_iterated_batch_gauss_newton_smoother(
        init_trajectory.mean,
        observations,
        prior,
        transition_model,
        observation_model,
        nb_iter=nb_iter,
    )[0]


def sequential_solver(
    observations,
    prior,
    transition_model,
    observation_model,
    init_trajectory,
    linearization_method=extended,
    nb_iter=10,
    return_loglikelihood=False
):
    return iterated_smoothing(
        observations,
        prior,
        transition_model,
        observation_model,
        linearization_method,
        init_trajectory,
        parallel=False,
        criterion=lambda i, *_: i < nb_iter,
        return_loglikelihood=return_loglikelihood,
    )


def sequential_solver_with_trust_region(
    observations,
    prior,
    transition_model,
    observation_model,
    init_trajectory,
    linearization_method,
    nb_iter=10,
):
    return trust_region_iterated_recursive_gauss_newton_smoother(
        init_trajectory,
        observations,
        prior,
        transition_model,
        observation_model,
        linearization_method,
        nb_iter=nb_iter,
    )[0]


def second_order_sequential_solver_with_trust_region(
    observations,
    prior,
    transition_model,
    observation_model,
    init_trajectory,
    linearization_method,
    nb_iter=10,
):
    return trust_region_iterated_recursive_newton_smoother(
        init_trajectory,
        observations,
        prior,
        transition_model,
        observation_model,
        linearization_method,
        nb_iter=nb_iter,
    )[0]


def parallel_solver(
    observations,
    prior,
    transition_model,
    observation_model,
    init_trajectory,
    linearization_method=extended,
    nb_iter=10,
    return_loglikelihood=False
):
    return iterated_smoothing(
        observations,
        prior,
        transition_model,
        observation_model,
        linearization_method,
        init_trajectory,
        parallel=True,
        criterion=lambda i, *_: i < nb_iter,
        return_loglikelihood=return_loglikelihood,
    )