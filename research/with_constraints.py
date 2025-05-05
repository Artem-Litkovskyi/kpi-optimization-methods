from copy import deepcopy
from typing import Callable

import numpy as np

from methods.derivation_methods import DerivationMethod
from methods.gradient_methods import fletcher_reeves, Modification, TerminationCriterion
from methods.interval_methods import golden_section, dsk_powell
from methods.penalty_methods import barrier_search, barrier_circle, barrier_line, barrier_ellipse
from research.utils import image_search_path, image_call_and_deviation, table_call_and_deviation
from output import images as img


SEARCH_METHOD = barrier_search
SUB_DIR = 'with_constraints'


def research(func: Callable, x0: np.array, base_params: dict, real_target_x: np.array, real_target_f: float):
    params = {
        'search_method': fletcher_reeves,
        'search_params': deepcopy(base_params),
        'r0': 1,
        'r_mult': 2,
        'accuracy': 1e-5,
        'max_iter': 10
    }

    params = part1_target_inside(func, x0, params, real_target_x, real_target_f)

    params = part2_target_outside(func, x0, params, real_target_x, real_target_f)
    
    params = part3_target_outside_concave(func, x0, params, real_target_x, real_target_f)

    return params


def part1_target_inside(func, x0, base_params, real_target_x, real_target_f):
    params = deepcopy(base_params)
    params['constraints'] = [barrier_circle(0.5, 0.75, 0.7, False)]

    image_search_path(
        func, x0, SEARCH_METHOD, params,
        real_target_x, real_target_f,
        'Target inside #1',
        SUB_DIR, 'inside1'
    )

    params['search_params']['lambda_accuracy'] = 1e-4
    image_search_path(
        func, np.array((75, 25), dtype=np.float64), SEARCH_METHOD, params,
        real_target_x, real_target_f,
        'Target inside #2',
        SUB_DIR, 'inside2',
        levels_n=21, pixels_per_unit=32,
        pad_big_x=5, pad_big_y=10, pad_small_x=1, pad_small_y=1,
    )

    image_search_path(
        func, np.array((75, 0), dtype=np.float64), SEARCH_METHOD, params,
        real_target_x, real_target_f,
        'Target inside #3',
        SUB_DIR, 'inside3',
        levels_n=21, pixels_per_unit=32,
        pad_big_x=5, pad_big_y=20, pad_small_x=1, pad_small_y=1,
    )

    params['search_params']['lambda_accuracy'] = 1e-6
    
    return params


def part2_target_outside(func, x0, base_params, real_target_x, real_target_f):
    params = deepcopy(base_params)
    params['constraints'] = [barrier_circle(0.25, 0.4, 0.7, False)]

    target_wolfram_2d_x = np.array((0.822493, 0.815033))

    img.search_path(
        func, real_target_x, [target_wolfram_2d_x],
        'Wolfram 2D',
        SUB_DIR, 'wolfram_2d',
        constraints=params['constraints'], constrained_target=target_wolfram_2d_x
    )

    t = 0.625463
    target_wolfram_1d_x = np.array((0.25 + 0.7 * np.cos(t), 0.4 + 0.7 * np.sin(t)))
    target_wolfram_1d_f = func(*target_wolfram_1d_x)


    # === FIRST RUN ===
    image_search_path(
        func, x0,
        SEARCH_METHOD, params,
        real_target_x, real_target_f,
        'Target outside #1',
        SUB_DIR, 'outside1',
        constrained_target_x=target_wolfram_1d_x, constrained_target_f=target_wolfram_1d_f,
    )

    image_search_path(
        func, np.array((2, 1), dtype=np.float64),
        SEARCH_METHOD, params,
        real_target_x, real_target_f,
        'Target outside #2',
        SUB_DIR, 'outside2',
        constrained_target_x=target_wolfram_1d_x, constrained_target_f=target_wolfram_1d_f,
    )
    
    return params


def part3_target_outside_concave(func, x0, base_params, real_target_x, real_target_f):
    params = deepcopy(base_params)
    params['constraints'] = [
        barrier_circle(0.25, 0.4, 0.7, False),
        barrier_ellipse(0.25, 0, 0.4, 0.7, 0, True)
    ]

    t = 0.625463
    target_wolfram_1d_x = np.array((0.25 + 0.7 * np.cos(t), 0.4 + 0.7 * np.sin(t)))
    target_wolfram_1d_f = func(*target_wolfram_1d_x)


    # === FIRST RUN ===
    image_search_path(
        func, x0,
        SEARCH_METHOD, params,
        real_target_x, real_target_f,
        'Target outside, concave region #1',
        SUB_DIR, 'concave1',
        constrained_target_x=target_wolfram_1d_x, constrained_target_f=target_wolfram_1d_f
    )


    # === CHANGE REGION ===
    t = 0.775393
    target_wolfram_1d_x = np.array((np.cos(t), np.sin(t)))
    target_wolfram_1d_f = func(*target_wolfram_1d_x)

    params['constraints'] = [
        barrier_circle(0, 0, 1, False),
        barrier_ellipse(0.5, -0.5, 0.6, 1.4, 45, True)
    ]

    image_search_path(
        func, np.array((-0.6, -0.6), dtype=np.float64),
        SEARCH_METHOD, params,
        real_target_x, real_target_f,
        'Target outside, concave region #2',
        SUB_DIR, 'concave2',
        constrained_target_x=target_wolfram_1d_x, constrained_target_f=target_wolfram_1d_f
    )


    return params
