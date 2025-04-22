from enum import Enum

import numpy as np

from methods.derivation_methods import DerivationMethod, nabla
from methods.interval_methods import sven
from custom_types import *


# =======================================================================================
class TerminationCriterion(Enum):
    X_AND_F_CHANGE = 1
    S_NORM = 2
    NABLA_NORM = 3


def get_func_of_lambda(func: Callable, x_prev: np.array, s: np.array):
    return lambda lamb: func(*(x_prev + lamb * s))


# =======================================================================================
def fletcher_reeves(
        func: Callable,
        derivation_h: float, derivation_method: DerivationMethod,
        x0: np.ndarray,
        lambda_method: Callable, delta_lambda: float, lambda_accuracy: float,
        accuracy: float, termination_criterion: TerminationCriterion,
        output_receiver: Callable = None
):
    if x0.dtype != np.float64:
        raise Warning('Method might not work as expected if the x0 vector consists of non-floats')

    f0 = func(*x0)
    nabla0 = nabla(func, x0, derivation_h, derivation_method, f0=f0)
    s0 = -nabla0

    return _fletcher_reeves(
        func,
        derivation_h, derivation_method,
        x0, f0, s0, nabla0,
        lambda_method, delta_lambda, lambda_accuracy,
        accuracy, termination_criterion,
        output_receiver
    )


def _fletcher_reeves(
        func,
        derivation_h, derivation_method,
        x0, f0, s0, nabla0,
        lambda_method, delta_lambda, lambda_accuracy,
        accuracy, termination_criterion,
        output_receiver
):
    func_lamb = get_func_of_lambda(func, x0, s0)
    lambda_opt_x_interval, lambda_opt_f_interval = sven(func_lamb, 0, delta_lambda)
    lambda_opt, f1 = lambda_method(func_lamb, lambda_opt_x_interval, lambda_opt_f_interval, lambda_accuracy)

    if output_receiver:
        output_receiver(
            x = x0,
            f = f0,
            nabla = nabla0,
            s = s0,
            lambda_interval = lambda_opt_x_interval,
            lambda_opt = lambda_opt
        )

    x1 = x0 + lambda_opt * s0
    nabla1 = nabla(func, x1, derivation_h, derivation_method, f0=f1)

    if termination_criterion == TerminationCriterion.X_AND_F_CHANGE:
        if np.linalg.norm(x0) == 0:
            terminate = False
        else:
            terminate = np.linalg.norm(x1 - x0) / np.linalg.norm(x0) <= accuracy and abs(f1 - f0) <= accuracy
    elif termination_criterion == TerminationCriterion.S_NORM:
        terminate = np.linalg.norm(s0) <= accuracy
    elif termination_criterion == TerminationCriterion.NABLA_NORM:
        terminate = np.linalg.norm(nabla1) <= accuracy
    else:
        raise ValueError('Unknown termination criterion')

    if terminate:
        return x1, f1

    s1 = -nabla1 + np.inner(nabla1, nabla1) / np.inner(nabla0, nabla0) * s0

    # TODO: restart

    return _fletcher_reeves(
        func, derivation_h, derivation_method,
        x1, f1, s1, nabla1,
        lambda_method, delta_lambda, lambda_accuracy,
        accuracy, termination_criterion,
        output_receiver
    )
