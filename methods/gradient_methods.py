from enum import Enum

import numpy as np

from methods.derivation_methods import DerivationMethod, nabla
from methods.interval_methods import sven
from custom_types import *


# =======================================================================================
class TerminationCriterion(Enum):
    X_AND_F_CHANGE = 1
    S_NORM = 2


class Modification(Enum):
    FLETCHER_REEVES = 1
    POLAK_RIBIERE = 2


def get_func_of_lambda(func: Callable, x_prev: np.array, s: np.array):
    return lambda lamb: func(*(x_prev + lamb * s))


# =======================================================================================
def fletcher_reeves(
        func: Callable,
        x0: np.ndarray,
        derivation_method: DerivationMethod, derivation_h: float,
        lambda_method: Callable, delta_lambda: float, lambda_accuracy: float,
        modification: Modification, termination_criterion: TerminationCriterion, accuracy: float,
        restart_lambda_threshold: float = -1,
        max_iter: int = -1,
        output_receiver: Callable = None
):
    if x0.dtype != np.float64:
        raise Warning('Method might not work as expected if the x0 vector consists of non-floats')

    f0 = func(*x0)
    nabla0 = nabla(func, x0, derivation_h, derivation_method, f0=f0)
    s0 = -nabla0

    iter_n = 0

    while True:
        func_lamb = get_func_of_lambda(func, x0, s0)
        lambda_opt_x_interval, lambda_opt_f_interval = sven(func_lamb, 0, delta_lambda)
        lambda_opt, f1 = lambda_method(func_lamb, lambda_opt_x_interval, lambda_opt_f_interval, lambda_accuracy)

        # print('=' * 100)
        # print('iter_n', iter_n)

        if output_receiver:
            output_receiver(
                iter_n=iter_n,
                x=x0,
                f=f0,
                nabla=nabla0,
                s=s0,
                lambda_interval=lambda_opt_x_interval,
                lambda_opt=lambda_opt
            )

        if restart_lambda_threshold > 0 and lambda_opt < restart_lambda_threshold:
            s0 = -nabla0  # restart

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

        if iter_n == max_iter:
            terminate = True

        if terminate:
            return x1, f1

        s1 = -nabla1 + _fletcher_reeves_w(nabla0, nabla1, modification) * s0

        x0 = x1
        f0 = f1
        s0 = s1
        nabla0 = nabla1

        iter_n += 1


def _fletcher_reeves_w(nabla0, nabla1, modification):
    if modification == Modification.FLETCHER_REEVES:
        return max(0, np.inner(nabla1, nabla1) / np.inner(nabla0, nabla0))
    elif modification == Modification.POLAK_RIBIERE:
        return max(0, np.inner(nabla1, nabla1 - nabla0) / np.inner(nabla0, nabla0))
    else:
        raise ValueError('Unknown modification')
