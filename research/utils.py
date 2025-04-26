from copy import deepcopy
from enum import Enum

import numpy as np

import output.images as img
import output.tables as tbl


# === IMAGES ===
def image_search_path(
        func, search_method, search_params,
        real_target,
        title, subdir, filename,
        constrained_target=None
):
    output = []

    if func.calls is not None:
        func.calls = 0

    x, _ = search_method(func, **search_params, output_receiver=lambda **kwargs: output.append(kwargs))

    constraints = search_params.get('constraints')

    img.search_path(
        func, real_target, [r['x'] for r in output], title, subdir, filename,
        constraints=constraints, constrained_target=constrained_target
    )

    message = 'Rendered "%s" (%s)' % (title, filename)
    if 'constraint_r' in output[-1]:
        message += '. Constraint R = %.00f' % output[-1]['constraint_r']
    if func.calls:
        message += '. Target function calls: %i' % func.calls
    message += '. Deviation: %.2E' % np.linalg.norm(x - real_target)
    print(message)


def image_call_and_deviation(
        func, search_method, search_params,
        change_param, values, old_v, new_v,
        real_target,
        title, param_label, subdir, filename
):
    params = deepcopy(search_params)

    results = feed_values_1d(func, search_method, params, change_param, values, real_target)

    img.calls_and_deviation(
        values,
        [r['calls'] for r in results],
        [r['deviation'] for r in results],
        param_label, old_v, new_v,
        title, subdir, filename
    )


# === TABLES ===
def table_call_and_deviation(
        func, search_method, search_params,
        change_param1, values1,
        change_param2, values2,
        real_target,
        subdir, filename
):
    results = feed_values_2d(
        func, search_method, search_params,
        change_param1, values1,
        change_param2, values2,
        real_target
    )

    tbl.calls_and_deviation(
        results,
        map(lambda v: _smart_value_to_str(v), values1),
        map(lambda v: '%.2E' % v, values2),
        subdir, filename
    )


def _smart_value_to_str(v):
    if callable(v):
        return v.__name__
    elif isinstance(v, Enum):
        return v.value
    return str(v)



# === OTHER ===
def call_counter(f):
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return f(*args, **kwargs)

    wrapped.calls = 0
    return wrapped


def feed_values_2d(func, search_method, search_params, change_param1, values1, change_param2, values2, real_target):
    params = deepcopy(search_params)

    results = []
    for v in values1:
        if type(change_param1) != list:
            params[change_param1] = v
        else:
            param = params
            for i in change_param1:
                param = param[i]
            param = v
        results.append(feed_values_1d(func, search_method, params, change_param2, values2, real_target))

    return results


def feed_values_1d(func, search_method, search_params, change_param, values, real_target):
    results = []

    for value in values:
        output = []

        params = deepcopy(search_params)

        if type(change_param) != list:
            params[change_param] = value
        else:
            param = params
            for i in change_param:
                param = param[i]
            param = value

        func.calls = 0

        x, f = search_method(
            func, **params,
            output_receiver=lambda **kwargs: output.append(kwargs)
        )

        results.append({
            'output': output,
            'x': x, 'f': f,
            'deviation': np.linalg.norm(x - real_target),
            'calls': func.calls
        })

    return results