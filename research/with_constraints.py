from copy import deepcopy
from typing import Callable

import numpy as np

import output.images as img
import output.tables as tbl
from methods.derivation_methods import DerivationMethod
from methods.gradient_methods import fletcher_reeves, Modification, TerminationCriterion
from methods.interval_methods import dsk_powell, golden_section
from methods.penalty_methods import barrier_search, outer_barrier


def target_inside(x1, x2):
    return (x1 - 0.5) ** 2 + (x2 - 0.75) ** 2 - 0.5


def target_outside(x1, x2):
    return (x1 - 0.25) ** 2 + (x2 - 0.4) ** 2 - 0.5


def research(base_params: dict, real_target: np.array):
    params = deepcopy(base_params)

    part2_target_outside(params, real_target)


def part1_target_inside(base_params, real_target):
    output = []

    x, f = barrier_search(
        fletcher_reeves, base_params,
        [target_inside],
        1, 10, 1e-5,
        output_receiver=lambda **kwargs: output.append(kwargs)
    )

    img.search_path(
        base_params['func'],
        real_target, [r['x'] for r in output],
        'test', 'test',
        constraints=[target_inside]
    )


def part2_target_outside(base_params, real_target):
    output = []

    x, f = barrier_search(
        fletcher_reeves, base_params,
        [target_outside],
        1, 10, 1e-5,
        output_receiver=lambda **kwargs: output.append(kwargs)
    )

    img.search_path(
        base_params['func'],
        real_target, [r['x'] for r in output],
        'test', 'test',
        constraints=[target_outside]
    )
