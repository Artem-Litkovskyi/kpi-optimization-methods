from copy import deepcopy
from typing import Callable

import numpy as np

from methods.derivation_methods import DerivationMethod
from methods.gradient_methods import fletcher_reeves, Modification, TerminationCriterion
from methods.interval_methods import golden_section, dsk_powell
from methods.penalty_methods import barrier_search, barrier_circle, barrier_line, barrier_ellipse
from research.utils import image_search_path, image_call_and_deviation, table_call_and_deviation, \
    table_penalty_method_iters
from output import images as img


SEARCH_METHOD = barrier_search
SUB_DIR = 'with_constraints'


def research(func: Callable, x0: np.array, base_params: dict, real_target_x: np.array, real_target_f: float):
    params = {
        'search_method': fletcher_reeves,
        'search_params': deepcopy(base_params),
        'r0': 1,
        'r_mult': 10,
        'accuracy': 1e-4,
        'max_iter': 12
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
    table_penalty_method_iters(func, x0, SEARCH_METHOD, params, SUB_DIR, 'inside1')

    x0_new = np.array((75, 25), dtype=np.float64)
    image_search_path(
        func, x0_new, SEARCH_METHOD, params,
        real_target_x, real_target_f,
        'Target inside #2',
        SUB_DIR, 'inside2',
        levels_n=21, pixels_per_unit=32,
        pad_big_x=5, pad_big_y=10, pad_small_x=1, pad_small_y=1,
    )
    table_penalty_method_iters(func, x0_new, SEARCH_METHOD, params, SUB_DIR, 'inside2')
    
    return params


def part2_target_outside(func, x0, base_params, real_target_x, real_target_f):
    params = deepcopy(base_params)
    params['constraints'] = [barrier_circle(0.25, 0.4, 0.7, False)]

    target_wolfram_x = np.array((0.817485, 0.809831))
    target_wolfram_f = func(*target_wolfram_x)


    # === FIRST RUN ===
    image_search_path(
        func, x0,
        SEARCH_METHOD, params,
        real_target_x, real_target_f,
        'Target outside #1',
        SUB_DIR, 'outside1',
        constrained_target_x=target_wolfram_x, constrained_target_f=target_wolfram_f,
    )
    table_penalty_method_iters(func, x0, SEARCH_METHOD, params, SUB_DIR, 'outside1')

    x0_new = np.array((2, 1), dtype=np.float64)
    image_search_path(
        func, x0_new,
        SEARCH_METHOD, params,
        real_target_x, real_target_f,
        'Target outside #2',
        SUB_DIR, 'outside2',
        constrained_target_x=target_wolfram_x, constrained_target_f=target_wolfram_f,
    )
    table_penalty_method_iters(func, x0_new, SEARCH_METHOD, params, SUB_DIR, 'outside2')
    
    return params


def part3_target_outside_concave(func, x0, base_params, real_target_x, real_target_f):
    params = deepcopy(base_params)
    params['constraints'] = [
        barrier_circle(0.25, 0.4, 0.7, False),
        barrier_ellipse(0.25, 0, 0.4, 0.7, 0, True)
    ]

    target_wolfram_x = np.array((0.817485, 0.809831))
    target_wolfram_f = func(*target_wolfram_x)


    # === FIRST RUN ===
    image_search_path(
        func, x0,
        SEARCH_METHOD, params,
        real_target_x, real_target_f,
        'Target outside, concave region',
        SUB_DIR, 'concave1',
        constrained_target_x=target_wolfram_x, constrained_target_f=target_wolfram_f
    )
    table_penalty_method_iters(func, x0, SEARCH_METHOD, params, SUB_DIR, 'concave1')

    x0_new = np.array((1.25, 0.8), dtype=np.float64)
    image_search_path(
        func, x0_new,
        SEARCH_METHOD, params,
        real_target_x, real_target_f,
        'Target outside, concave region (another start point)',
        SUB_DIR, 'concave2',
        constrained_target_x=target_wolfram_x, constrained_target_f=target_wolfram_f,
        pad_big_x=0.5, pad_big_y=0.5
    )
    table_penalty_method_iters(func, x0_new, SEARCH_METHOD, params, SUB_DIR, 'concave2')


    return params
