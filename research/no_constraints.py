from copy import deepcopy
from typing import Callable

import numpy as np

import output.images as img
from methods.derivation_methods import DerivationMethod
from methods.gradient_methods import fletcher_reeves, Modification, TerminationCriterion
from methods.interval_methods import dsk_powell, golden_section
from research.utils import image_search_path, image_call_and_deviation, table_call_and_deviation


SEARCH_METHOD = fletcher_reeves
SUB_DIR = 'no_constraints'


def research(func: Callable, base_params: dict, real_target: np.array):
    params = deepcopy(base_params)


    # === SURFACE ===
    img.surface(
        func, base_params['x0'], real_target, (-3, 3), (-3, 3),
        None, SUB_DIR, 'target_func'
    )


    # === FIRST RUN ===
    image_search_path(
        func, SEARCH_METHOD, params, real_target,
        'First run', SUB_DIR, 'first_run'
    )


    # === SVENN ===
    image_call_and_deviation(
        func, SEARCH_METHOD, params,
        'delta_lambda', np.linspace(1e-5, 0.5, 128),
        params['delta_lambda'], 0.05,
        real_target,
        None, 'Δλ',
        SUB_DIR, 'svenn'
    )

    params['delta_lambda'] = 0.05


    # === LAMBDA ===
    table_call_and_deviation(
        func, SEARCH_METHOD, params,
        'lambda_method', (golden_section, dsk_powell),
        'lambda_accuracy', (1e-1, 1e-2, 1e-3, 1e-4, 1e-5),
        real_target,
        SUB_DIR, 'lambda'
    )

    params['lambda_method'] = golden_section
    params['lambda_accuracy'] = 1e-2

    image_search_path(
        func, SEARCH_METHOD, params, real_target,
        'Golden section, ε=1E-2', SUB_DIR, 'golden'
    )

    params['lambda_method'] = dsk_powell
    params['lambda_accuracy'] = 1e-1

    image_search_path(
        func, SEARCH_METHOD, params, real_target,
        'DSK-Powell, ε=1E-1', SUB_DIR, 'dsk'
    )


    # === RESTART ===
    image_call_and_deviation(
        func, SEARCH_METHOD, params,
        'restart_lambda_threshold', np.linspace(1e-5, 0.5, 128),
        None, 0.24,
        real_target,
        None, 'Restart λ threshold',
        SUB_DIR, 'restart'
    )

    params['restart_lambda_threshold'] = 0.24

    image_search_path(
        func, SEARCH_METHOD, params, real_target,
        'Restart threshold: λ=0.24', SUB_DIR, 'restart'
    )


    # === DERIVATION ===
    table_call_and_deviation(
        func, SEARCH_METHOD, params,
        'derivation_method', (DerivationMethod.LEFT_DIFF, DerivationMethod.RIGHT_DIFF, DerivationMethod.SYM_DIFF),
        'derivation_h', (1e+3, 1e+2, 1e+1, 1, 1e-1, 1e-2, 1e-3),
        real_target,
        SUB_DIR, 'derivation'
    )

    params['derivation_method'] = DerivationMethod.SYM_DIFF
    params['derivation_h'] = 1

    image_search_path(
        func, SEARCH_METHOD, params, real_target,
        'Symmetric difference, h=1', SUB_DIR, 'derivation'
    )


    # === TERMINATION ===
    table_call_and_deviation(
        func, SEARCH_METHOD, params,
        'termination_criterion', (TerminationCriterion.X_AND_F_CHANGE, TerminationCriterion.S_NORM),
        'accuracy', (1e-7, 1e-8, 1e-9, 1e-10, 1e-11),
        real_target,
        SUB_DIR, 'termination'
    )

    params['termination_criterion'] = TerminationCriterion.X_AND_F_CHANGE
    params['accuracy'] = 1e-9


    # === MODIFICATION ===
    params['modification'] = Modification.POLAK_RIBIERE

    image_search_path(
        func, SEARCH_METHOD, params, real_target,
        'Polak-Ribiere', SUB_DIR, 'polak_ribiere'
    )


    return params
