from copy import deepcopy
from typing import Callable

import numpy as np

from methods.derivation_methods import DerivationMethod
from methods.gradient_methods import fletcher_reeves, Modification, TerminationCriterion
from methods.interval_methods import golden_section, dsk_powell
from methods.penalty_methods import barrier_search, barrier_circle, barrier_line, barrier_ellipse
from research.utils import image_search_path, image_call_and_deviation, table_call_and_deviation


SEARCH_METHOD = barrier_search
SUB_DIR = 'with_constraints'


def research(func: Callable, base_params: dict, real_target: np.array):
    params = {
        'search_method': fletcher_reeves,
        'search_params': deepcopy(base_params),
        'r0': 1,
        'r_mult': 2,
        'barrier_accuracy': 1e-5,
        'max_iter': 10
    }

    params = part1_target_inside(func, params, real_target)

    params = part2_target_outside(func, params, real_target)

    params = part3_target_outside_concave(func, params, real_target)

    return params


def part1_target_inside(func, base_params, real_target):
    params = deepcopy(base_params)
    params['constraints'] = [barrier_circle(0.5, 0.75, 0.7, False)]


    # === FIRST RUN ===
    image_search_path(
        func, SEARCH_METHOD, params,
        real_target,
        'First run (target inside)',
        SUB_DIR, 'first_run_inside',
        constrained_target=None
    )


    # === DERIVATION ===
    table_call_and_deviation(
        func, SEARCH_METHOD, params,
        ['search_params', 'derivation_method'],
        (DerivationMethod.LEFT_DIFF, DerivationMethod.RIGHT_DIFF, DerivationMethod.SYM_DIFF),
        ['search_params', 'derivation_h'],
        (1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5),
        real_target,
        SUB_DIR, 'derivation_inside'
    )

    params['search_params']['derivation_method'] = DerivationMethod.SYM_DIFF
    params['search_params']['derivation_h'] = 1e-1

    image_search_path(
        func, SEARCH_METHOD, params,
        real_target,
        'Symmetric difference, h=0.1 (target inside)',
        SUB_DIR, 'derivation_inside',
        constrained_target=None
    )


    # === LAMBDA ===
    table_call_and_deviation(
        func, SEARCH_METHOD, params,
        ['search_params', 'lambda_method'], (golden_section, dsk_powell),
        ['search_params', 'lambda_accuracy'], (1, 1e-1, 1e-2, 1e-3),
        real_target,
        SUB_DIR, 'lambda_inside'
    )

    params['search_params']['lambda_method'] = dsk_powell
    params['search_params']['lambda_accuracy'] = 1e-2

    image_search_path(
        func, SEARCH_METHOD, params,
        real_target,
        'DSK-Powell, ε=0.01 (target inside)',
        SUB_DIR, 'lambda_inside',
        constrained_target=None
    )


    # === SVENN ===
    image_call_and_deviation(
        func, SEARCH_METHOD, params,
        ['search_params', 'delta_lambda'], np.linspace(1e-5, 0.5, 128),
        params['search_params']['delta_lambda'], 0.01,
        real_target,
        None, 'Δλ',
        SUB_DIR, 'svenn_inside',
        min_calls=0, max_calls=2000
    )

    params['search_params']['delta_lambda'] = 0.01

    image_search_path(
        func, SEARCH_METHOD, params,
        real_target,
        'Δλ=0.01 (target inside)',
        SUB_DIR, 'svenn_inside',
        constrained_target=None
    )


    return params


def part2_target_outside(func, base_params, real_target):
    params = deepcopy(base_params)
    params['constraints'] = [barrier_circle(0.25, 0.4, 0.7, False)]

    target_wolfram_2d = np.array((0.822493, 0.815033))

    t = 0.625463
    target_wolfram_1d = np.array((0.25 + 0.7 * np.cos(t), 0.4 + 0.7 * np.sin(t)))


    # === FIRST RUN ===
    image_search_path(
        func, SEARCH_METHOD, params,
        real_target,
        'First run (target outside)',
        SUB_DIR, 'first_run_outside',
        constrained_target=target_wolfram_2d
    )


    # === DERIVATION ===
    table_call_and_deviation(
        func, SEARCH_METHOD, params,
        ['search_params', 'derivation_method'],
        (DerivationMethod.LEFT_DIFF, DerivationMethod.RIGHT_DIFF, DerivationMethod.SYM_DIFF),
        ['search_params', 'derivation_h'],
        (1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5),
        target_wolfram_1d,
        SUB_DIR, 'derivation_outside'
    )

    params['search_params']['derivation_method'] = DerivationMethod.SYM_DIFF
    params['search_params']['derivation_h'] = 1e-4

    image_search_path(
        func, SEARCH_METHOD, params,
        real_target,
        'Symmetric difference, h=0.0001 (target outside)',
        SUB_DIR, 'derivation_outside',
        constrained_target=target_wolfram_1d
    )


    # === RESTART ===
    image_call_and_deviation(
        func, SEARCH_METHOD, params,
        ['search_params', 'restart_lambda_threshold'], np.linspace(1e-5, 0.5, 128),
        None, None,
        target_wolfram_1d,
        None, 'Restart λ threshold',
        SUB_DIR, 'restart_outside'
    )

    image_call_and_deviation(
        func, SEARCH_METHOD, params,
        ['search_params', 'restart_lambda_threshold'], np.linspace(1e-5, 1e-2, 128),
        None, 1e-3,
        target_wolfram_1d,
        None, 'Restart λ threshold',
        SUB_DIR, 'restart_outside_zoom'
    )

    params['search_params']['restart_lambda_threshold'] = 1e-3

    image_search_path(
        func, SEARCH_METHOD, params,
        real_target,
        'Restart λ threshold: λ=0.001 (target outside)',
        SUB_DIR, 'restart_outside',
        constrained_target=target_wolfram_1d
    )


    # === LAMBDA ===
    table_call_and_deviation(
        func, SEARCH_METHOD, params,
        ['search_params', 'lambda_method'], (golden_section, dsk_powell),
        ['search_params', 'lambda_accuracy'], (1, 1e-1, 1e-2, 1e-3, 1e-4),
        target_wolfram_1d,
        SUB_DIR, 'lambda_outside'
    )

    params['search_params']['lambda_method'] = dsk_powell
    params['search_params']['lambda_accuracy'] = 1e-2

    image_search_path(
        func, SEARCH_METHOD, params,
        real_target,
        'DSK-Powell, ε=0.01 (target outside)',
        SUB_DIR, 'lambda_outside',
        constrained_target=target_wolfram_1d
    )


    return params


def part3_target_outside_concave(func, base_params, real_target):
    params = deepcopy(base_params)
    params['constraints'] = [
        barrier_circle(0.25, 0.4, 0.7, False),
        barrier_ellipse(0.25, 0, 0.4, 0.7, 0, True)
    ]

    t = 0.625463
    target_wolfram_1d = np.array((0.25 + 0.7 * np.cos(t), 0.4 + 0.7 * np.sin(t)))


    # === FIRST RUN ===
    image_search_path(
        func, SEARCH_METHOD, params,
        real_target,
        'First run (target outside, concave region)',
        SUB_DIR, 'first_run_concave',
        constrained_target=target_wolfram_1d
    )


    # === CHANGE REGION ===
    params['constraints'][1] = barrier_ellipse(0.6, 0.2, 0.3, 0.7, 30, True)

    image_search_path(
        func, SEARCH_METHOD, params,
        real_target,
        'Another region',
        SUB_DIR, 'change_region_concave',
        constrained_target=target_wolfram_1d
    )


    return params
