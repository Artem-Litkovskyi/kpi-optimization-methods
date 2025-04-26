from copy import deepcopy
from typing import Callable

import numpy as np

import output.images as img
import output.tables as tbl
from methods.derivation_methods import DerivationMethod
from methods.gradient_methods import fletcher_reeves, Modification, TerminationCriterion
from methods.interval_methods import dsk_powell, golden_section
from methods.penalty_methods import barrier_search, outer_barrier
from research.utils import image_search_path, feed_values_1d, feed_values_2d


SEARCH_METHOD = barrier_search
SUB_DIR = 'with_constraints'


def target_inside(x1, x2):
    return (x1 - 0.5) ** 2 + (x2 - 0.75) ** 2 - 0.5


def target_outside(x1, x2):
    return (x1 - 0.25) ** 2 + (x2 - 0.4) ** 2 - 0.5


def research(func: Callable, base_params: dict, real_target: np.array):
    constraint_params = {
        'search_method': fletcher_reeves,
        'search_params': deepcopy(base_params),
        'r0': 1,
        'r_mult': 10,
        'accuracy': 1e-5,
    }

    part1_target_inside(func, constraint_params, real_target)


def part1_target_inside(func, base_params, real_target):
    base_params['constraints'] = [target_inside]

    image_search_path(
        func, SEARCH_METHOD, base_params,
        real_target,
        'First run (constrained)',
        SUB_DIR, 'first_run_constrained',
        constrained_target=None
    )

    # part1_derivation(func, base_params, real_target)

# def part1_derivation(func, base_params, real_target):
#     values1 = (DerivationMethod.LEFT_DIFF, DerivationMethod.RIGHT_DIFF, DerivationMethod.SYM_DIFF)
#     values2 = (1e+3, 1e+2, 1e+1, 1, 1e-1, 1e-2, 1e-3)
#
#     results = feed_values_2d(
#         func, SEARCH_METHOD, base_params,
#         ['search_params', 'derivation_method'], values1,
#         ['search_params', 'derivation_h'], values2,
#         real_target
#     )
#
#     tbl.calls_and_deviation(
#         results,
#         map(lambda v: v.value, values1),
#         map(lambda v: '%.0E' % v, values2),
#         'derivation_constrained'
#     )


# def part2_target_outside(base_params, real_target):
#     base_params['constraints'] = [target_outside]
#
#     image_search_path(
#         base_params['search_params']['func'],
#         SEARCH_METHOD,
#         base_params,
#         real_target,
#         'test', 'test',
#         constrained_target=None
#     )
