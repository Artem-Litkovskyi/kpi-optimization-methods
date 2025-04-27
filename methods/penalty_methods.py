from copy import deepcopy
from typing import Callable

import numpy as np


# === BARRIER GENERATORS ===
def get_barrier_circle(x: float, y: float, r: float, invert: bool):
    return lambda x1, x2: _inv((x1 - x) ** 2 + (x2 - y) ** 2 - r ** 2, invert)


def get_barrier_line(x: float, y: float, angle: float, invert: bool):
    if angle == -90:
        return lambda x1, x2: _inv((x1 - x), invert)
    elif angle == 90:
        return lambda x1, x2: _inv(-(x1 - x), invert)

    return lambda x1, x2: _inv(np.tan(np.deg2rad(angle)) * (x1 - x) + y - x1, invert)


def _inv(v: float, invert: bool):
    if invert:
        return -v
    return v


# === BARRIER WRAPPERS ===
def outer_barrier(func: Callable, r: float, *constraints: Callable):
    def wrapped(*args, **kwargs):
        value = func(*args, **kwargs)

        for constraint in constraints:
            c = constraint(*args, **kwargs)
            c = np.maximum(c, 0)
            value += r * c ** 2

        return value

    return wrapped


# === SEARCH ===
def barrier_search(
        func: Callable,
        search_method: Callable, search_params: dict,
        constraints: list[Callable],
        r0: float, r_mult: float,
        barrier_accuracy: float,
        max_iter: int = -1,
        output_receiver: Callable = None
):
    params = deepcopy(search_params)

    r = r0

    iter_n = 0

    while True:
        print('BARRIR ++++++++++++++++++++++++++++++++++++++++++++++++++++++++', r)
        p_func = outer_barrier(func, r, *constraints)

        output = []

        x, f = search_method(
            p_func, **params,
            output_receiver=lambda **kwargs: output.append(kwargs)
        )

        for row in output:
            row['constraint_r'] = r

        max_dev = 0
        for constraint in constraints:
            c = constraint(*x)
            if c <= max_dev:
                continue
            max_dev = c

        if max_dev <= barrier_accuracy or iter_n == max_iter - 1:
            for row in output:
                output_receiver(**row)
            return x, f

        r *= r_mult
        iter_n += 1