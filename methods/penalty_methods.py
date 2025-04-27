from copy import deepcopy
from typing import Callable

import numpy as np


# === BARRIER GENERATORS ===
def barrier_circle(x: float, y: float, r: float, invert: bool):
    return lambda x1, x2: _inv((x1 - x) ** 2 + (x2 - y) ** 2 - r ** 2, invert)


def barrier_line(x: float, y: float, angle: float, invert: bool):
    if angle == -90:
        return lambda x1, x2: _inv((x1 - x), invert)
    elif angle == 90:
        return lambda x1, x2: _inv(-(x1 - x), invert)

    return lambda x1, x2: _inv(np.tan(np.deg2rad(angle)) * (x1 - x) + y - x1, invert)


def barrier_ellipse(x: float, y: float, a: float, b: float, angle: float, invert: bool):
    theta = np.deg2rad(angle)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    k1 = (a * sin_theta) ** 2 + (b * cos_theta) ** 2
    k2 = 2 * (b ** 2 - a ** 2) * sin_theta * cos_theta
    k3 = (a * cos_theta) ** 2 + (b * sin_theta) ** 2
    k4 = -2 * k1 * x - k2 * y
    k5 = -k2 * x - 2 * k3 * y
    k6 = k1 * x ** 2 + k2 * x * y + k3 * y ** 2 - a ** 2 * b ** 2

    return lambda x1, x2: _inv(k1 * x1 ** 2 + k2 * x1 * x2 + k3 * x2 ** 2 + k4 * x1 + k5 * x2 + k6, invert)


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