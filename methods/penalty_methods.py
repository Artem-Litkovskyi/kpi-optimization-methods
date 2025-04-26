from copy import deepcopy
from typing import Callable

import numpy as np


def outer_barrier(func: Callable, r: float, *constraints: Callable):
    def wrapped(*args, **kwargs):
        value = func(*args, **kwargs)

        for constraint in constraints:
            c = constraint(*args, **kwargs)
            c = np.maximum(c, 0)
            value += r * c ** 2

        return value

    return wrapped


def barrier_search(
        func: Callable,
        search_method: Callable, search_params: dict,
        constraints: list[Callable],
        r0: float, r_mult: float,
        accuracy: float,
        output_receiver: Callable = None
):
    params = deepcopy(search_params)

    r = r0

    output = []

    while True:
        p_func = outer_barrier(func, r, *constraints)

        tmp = []

        x, f = search_method(
            p_func, **params,
            output_receiver=lambda **kwargs: tmp.append(kwargs)
        )

        for row in tmp:
            row['constraint_r'] = r

        output.extend(tmp)

        max_dev = 0
        for constraint in constraints:
            c = constraint(*x)
            if c <= max_dev:
                continue
            max_dev = c

        if max_dev <= accuracy:
            for row in output:
                output_receiver(**row)
            return x, f

        r *= r_mult