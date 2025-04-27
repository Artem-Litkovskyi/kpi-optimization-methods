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

    search_method(func, **search_params, output_receiver=lambda **kwargs: output.append(kwargs))

    constraints = search_params.get('constraints')

    img.search_path(
        func, real_target, [r['x'] for r in output], title, subdir, filename,
        constraints=constraints, constrained_target=constrained_target
    )

    # Additional console log
    message = 'Rendered "%s" (%s)' % (title, filename)

    if 'constraint_r' in output[-1]:
        message += '. Constraint R = %.00f' % output[-1]['constraint_r']

    message += '. Target function calls: %i' % output[-1]['calls']

    active_target = real_target
    if constrained_target is not None:
        active_target = constrained_target

    message += '. Deviation: %.2E' % np.linalg.norm(output[-1]['x'] - active_target)

    print(message)


def image_call_and_deviation(
        func, search_method, search_params,
        change_param, values, old_v, new_v,
        real_target,
        title, param_label, subdir, filename,
        min_calls=None, max_calls=None, min_dev=None, max_dev=None
):
    params = deepcopy(search_params)

    results = feed_values_1d(func, search_method, params, change_param, values, real_target)

    img.calls_and_deviation(
        values,
        [r['output'][-1]['calls'] for r in results],
        [r['deviation'] for r in results],
        param_label, old_v, new_v,
        title, subdir, filename,
        min_calls=min_calls, max_calls=max_calls, min_dev=min_dev, max_dev=max_dev
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
def feed_values_2d(func, search_method, search_params, change_param1, values1, change_param2, values2, real_target):
    params = deepcopy(search_params)

    results = []
    for v in values1:
        if type(change_param1) != list:
            params[change_param1] = v
        else:
            param = params
            for i in change_param1[:-1]:
                param = param[i]
            param[change_param1[-1]] = v
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
            for i in change_param[:-1]:
                param = param[i]
            param[change_param[-1]] = value

        search_method(func, **params, output_receiver=lambda **kwargs: output.append(kwargs))

        results.append({
            'output': output,
            'deviation': np.linalg.norm(output[-1]['x'] - real_target)
        })

    return results