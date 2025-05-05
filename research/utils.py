from copy import deepcopy
from enum import Enum

import numpy as np

import output.images as img
import output.tables as tbl


def call_counter(f):
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return f(*args, **kwargs)

    wrapped.calls = 0
    return wrapped


# === IMAGES ===
def image_search_path(
        func, x0, search_method, search_params,
        real_target_x, real_target_f,
        title, subdir, filename,
        levels_n=7, pixels_per_unit=256,
        pad_big_x=0.25, pad_big_y=0.25, pad_small_x=0.05, pad_small_y=0.05,
        constrained_target_x=None, constrained_target_f=None
):
    output = []

    func.calls = 0
    search_method(func, x0, **search_params, output_receiver=lambda **kwargs: output.append(kwargs))

    constraints = search_params.get('constraints')

    img.search_path(
        func, real_target_x, [r['x'] for r in output], title, subdir, filename,
        levels_n=levels_n, pixels_per_unit=pixels_per_unit,
        pad_big_x=pad_big_x, pad_big_y=pad_big_y, pad_small_x=pad_small_x, pad_small_y=pad_small_y,
        constraints=constraints, constrained_target=constrained_target_x
    )

    # Additional console log
    message = 'Rendered path "%s" (%s)' % (title, filename)

    if 'constraint_r' in output[-1]:
        message += '. Constraint R = %.2f' % output[-1]['constraint_r']

    message += '. Target function calls: %i' % func.calls

    active_target_x = real_target_x
    if constrained_target_x is not None:
        active_target_x = constrained_target_x

    active_target_f = real_target_f
    if constrained_target_f is not None:
        active_target_f = constrained_target_f

    message += '. X deviation: %.2E. F deviation: %.2E' % (
        np.linalg.norm(output[-1]['x'] - active_target_x),
        np.linalg.norm(output[-1]['f'] - active_target_f)
    )

    print(message)


def image_call_and_deviation(
        func, x0, search_method, search_params,
        change_param, values, old_v, new_v,
        real_target_x, real_target_f,
        title, param_label, subdir, filename,
        min_calls=None, max_calls=None, min_dev=None, max_dev=None
):
    params = deepcopy(search_params)

    results = feed_values_1d(func, x0, search_method, params, change_param, values, real_target_x, real_target_f)

    img.calls_and_deviation(
        values,
        [r['calls'] for r in results],
        [r['f_deviation'] for r in results],
        param_label, old_v, new_v,
        title, subdir, filename,
        min_calls=min_calls, max_calls=max_calls, min_dev=min_dev, max_dev=max_dev
    )

    # Additional console log
    message = 'Rendered plot "%s" (%s). Change parameter: %s' % (title, filename, change_param)
    print(message)


# === TABLES ===
def table_call_and_deviation(
        func, x0, search_method, search_params,
        change_param1, values1,
        change_param2, values2,
        real_target_x, real_target_f,
        subdir, filename
):
    results = feed_values_2d(
        func, x0, search_method, search_params,
        change_param1, values1,
        change_param2, values2,
        real_target_x, real_target_f
    )

    tbl.calls_and_deviation(
        results,
        map(lambda v: _smart_value_to_str(v), values1),
        map(lambda v: '%.2E' % v, values2),
        subdir, filename
    )

    # Additional console log
    message = 'Calculated table (%s). Change parameters: %s, %s' % (filename, change_param1, change_param2)
    print(message)


def _smart_value_to_str(v):
    if callable(v):
        return v.__name__
    elif isinstance(v, Enum):
        return v.value
    return str(v)



# === OTHER ===
def feed_values_2d(
        func, x0, search_method, search_params,
        change_param1, values1, change_param2, values2,
        real_target_x, real_target_f
):
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
        results.append(
            feed_values_1d(func, x0, search_method, params, change_param2, values2, real_target_x, real_target_f)
        )

    return results


def feed_values_1d(
        func, x0, search_method, search_params,
        change_param, values,
        real_target_x, real_target_f
):
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

        func.calls = 0
        search_method(func, x0, **params, output_receiver=lambda **kwargs: output.append(kwargs))

        results.append({
            'output': output,
            'calls': func.calls,
            'x_deviation': np.linalg.norm(output[-1]['x'] - real_target_x),
            'f_deviation': np.linalg.norm(output[-1]['f'] - real_target_f)
        })

    return results