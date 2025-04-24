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
        search_method: Callable, search_params: dict,
        constraints: list[Callable],
        r0: float, r_mult: float,
        accuracy: float,
        output_receiver: Callable = None
):
    params = deepcopy(search_params)

    r = r0

    while True:
        p_func = outer_barrier(search_params['func'], r, *constraints)

        params['func'] = p_func

        output = []
        x, f = search_method(
            **params,
            output_receiver=lambda **kwargs: output.append(kwargs)
        )

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