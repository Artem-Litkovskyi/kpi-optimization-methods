from copy import deepcopy
from typing import Callable

import numpy as np

from methods.derivation_methods import DerivationMethod
from methods.gradient_methods import fletcher_reeves, Modification, TerminationCriterion
from methods.interval_methods import dsk_powell, golden_section
from research.utils import image_search_path, image_call_and_deviation, table_call_and_deviation
import output.images as img


SEARCH_METHOD = fletcher_reeves
SUB_DIR = 'no_constraints'


def research(func: Callable, x0: np.array, base_params: dict, real_target_x: np.array, real_target_f: float):
    params = deepcopy(base_params)


    # === SURFACE ===
    img.surface(
        func, x0, real_target_x, (-3, 3), (-3, 3),
        None, SUB_DIR, 'target_func'
    )


    # === FIRST RUN ===
    image_search_path(
        func, x0, SEARCH_METHOD, params,
        real_target_x, real_target_f,
        'First run', SUB_DIR, 'first_run'
    )


    # === SVENN ===
    image_call_and_deviation(
        func, x0, SEARCH_METHOD, params,
        'delta_lambda', np.linspace(1e-5, 0.5, 128),
        params['delta_lambda'], 0.05,
        real_target_x, real_target_f,
        None, 'Δλ',
        SUB_DIR, 'svenn'
    )

    params['delta_lambda'] = 0.05


    # === LAMBDA ===
    table_call_and_deviation(
        func, x0, SEARCH_METHOD, params,
        'lambda_method', (golden_section, dsk_powell),
        'lambda_accuracy', (1e-1, 1e-2, 1e-3, 1e-4, 1e-5),
        real_target_x, real_target_f,
        SUB_DIR, 'lambda'
    )

    params['lambda_method'] = golden_section
    params['lambda_accuracy'] = 1e-2

    image_search_path(
        func, x0, SEARCH_METHOD, params,
        real_target_x, real_target_f,
        'Golden section, ε=0.01', SUB_DIR, 'golden'
    )

    params['lambda_method'] = dsk_powell
    params['lambda_accuracy'] = 1e-1

    image_search_path(
        func, x0, SEARCH_METHOD, params,
        real_target_x, real_target_f,
        'DSK-Powell, ε=0.1', SUB_DIR, 'dsk'
    )


    # === DERIVATION ===
    table_call_and_deviation(
        func, x0, SEARCH_METHOD, params,
        'derivation_method', (DerivationMethod.LEFT_DIFF, DerivationMethod.RIGHT_DIFF, DerivationMethod.SYM_DIFF),
        'derivation_h', (1e+3, 1e+2, 1e+1, 1, 1e-1, 1e-2, 1e-3),
        real_target_x, real_target_f,
        SUB_DIR, 'derivation'
    )

    params['derivation_method'] = DerivationMethod.SYM_DIFF
    params['derivation_h'] = 1

    image_search_path(
        func, x0, SEARCH_METHOD, params,
        real_target_x, real_target_f,
        'Symmetric difference, h=1', SUB_DIR, 'derivation'
    )


    # === TERMINATION ===
    table_call_and_deviation(
        func, x0, SEARCH_METHOD, params,
        'termination_criterion', (TerminationCriterion.X_AND_F_CHANGE, TerminationCriterion.S_NORM),
        'accuracy', (1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15),
        real_target_x, real_target_f,
        SUB_DIR, 'termination'
    )

    params['termination_criterion'] = TerminationCriterion.X_AND_F_CHANGE
    params['accuracy'] = 1e-8


    # === MODIFICATION ===
    params['modification'] = Modification.POLAK_RIBIERE

    image_search_path(
        func, x0, SEARCH_METHOD, params,
        real_target_x, real_target_f,
        'Polak-Ribiere', SUB_DIR, 'polak_ribiere'
    )


    # === SVENN 2 ===
    image_call_and_deviation(
        func, x0, SEARCH_METHOD, params,
        'delta_lambda', np.linspace(1e-5, 0.4, 128),
        params['delta_lambda'], 0.31,
        real_target_x, real_target_f,
        None, 'Δλ',
        SUB_DIR, 'repeat_svenn'
    )

    params['delta_lambda'] = 0.31

    image_search_path(
        func, x0, SEARCH_METHOD, params,
        real_target_x, real_target_f,
        'Svenn, Δλ=0.31', SUB_DIR, 'repeat_svenn'
    )


    # === LAMBDA 2 ===
    table_call_and_deviation(
        func, x0, SEARCH_METHOD, params,
        'lambda_method', (golden_section, dsk_powell),
        'lambda_accuracy', (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
        real_target_x, real_target_f,
        SUB_DIR, 'repeat_lambda'
    )

    params['lambda_accuracy'] = 1e-6


    # === DERIVATION 2 ===
    table_call_and_deviation(
        func, x0, SEARCH_METHOD, params,
        'derivation_method', (DerivationMethod.LEFT_DIFF, DerivationMethod.RIGHT_DIFF, DerivationMethod.SYM_DIFF),
        'derivation_h', (1e+3, 1e+2, 1e+1, 1, 1e-1, 1e-2, 1e-3),
        real_target_x, real_target_f,
        SUB_DIR, 'repeat_derivation'
    )

    params['derivation_h'] = 1e-1


    # === RESTART ===
    image_call_and_deviation(
        func, x0, SEARCH_METHOD, params,
        'restart_lambda_threshold', np.linspace(1e-5, 0.5, 128),
        None, 1e-2,
        real_target_x, real_target_f,
        None, 'Restart λ threshold',
        SUB_DIR, 'restart'
    )

    params['restart_lambda_threshold'] = 1e-2

    image_search_path(
        func, x0, SEARCH_METHOD, params,
        real_target_x, real_target_f,
        'Restart λ threshold: 0.01', SUB_DIR, 'restart'
    )

    params.pop('restart_lambda_threshold')


    # === START POINT ===
    another_x0 = np.array((100, 0), dtype=np.float64)

    image_search_path(
        func, another_x0, SEARCH_METHOD, params,
        real_target_x, real_target_f,
        'Another start point', SUB_DIR, 'start_point',
        pixels_per_unit=32
    )


    return params
