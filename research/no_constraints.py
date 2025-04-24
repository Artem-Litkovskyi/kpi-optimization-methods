from copy import deepcopy

import numpy as np

import output.images as img
import output.tables as tbl
from methods.derivation_methods import DerivationMethod
from methods.gradient_methods import fletcher_reeves, Modification, TerminationCriterion
from methods.interval_methods import dsk_powell, golden_section


def research(base_params: dict, real_target: np.array):
    params = deepcopy(base_params)

    # ===
    img.surface(
        base_params['func'], base_params['x0'], real_target, (-3, 3), (-3, 3),
        None, 'target_func'
    )

    # ===
    render_search_path(params, real_target, 'First run', 'first_run')

    # ===
    part1_svenn(params, real_target, 0.05)

    params['delta_lambda'] = 0.05

    # ===
    part2_lambda(params, real_target)

    params['lambda_method'] = golden_section
    params['lambda_accuracy'] = 1e-2

    render_search_path(params, real_target, 'Golden section, ε=1E-2', 'golden')

    params['lambda_method'] = dsk_powell
    params['lambda_accuracy'] = 1e-1

    render_search_path(params, real_target, 'DSK-Powell, ε=1E-1', 'dsk')

    # ===
    part3_restart(params, real_target, 0.24)

    params['restart_lambda_threshold'] = 0.24

    render_search_path(params, real_target, 'Restart threshold: λ=0.24', 'restart')

    # ===
    part4_derivation(params, real_target)

    params['derivation_method'] = DerivationMethod.SYM_DIFF
    params['derivation_h'] = 1

    render_search_path(params, real_target, 'Symmetric difference, h=1', 'derivation')

    # ===
    part5_termination(params, real_target)

    params['termination_criterion'] = TerminationCriterion.X_AND_F_CHANGE
    params['accuracy'] = 1e-9

    # ===
    params['modification'] = Modification.POLAK_RIBIERE

    render_search_path(params, real_target, 'Polak-Ribiere', 'polak_ribiere')

    return params


# === RESEARCH STEPS ===
def part1_svenn(base_params, real_target, new_v):
    params = deepcopy(base_params)

    values = np.linspace(1e-5, 0.5, 128)

    results = feed_values_1d(params, 'delta_lambda', values, real_target)

    img.calls_and_deviation(
        values,
        [r['calls'] for r in results],
        [r['deviation'] for r in results],
        'Δλ', base_params['delta_lambda'], new_v,
        None,
        'svenn'
    )


def part2_lambda(base_params, real_target):
    values1 = (golden_section, dsk_powell)
    values2 = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)

    results = feed_values_2d(
        base_params,
        'lambda_method', values1,
        'lambda_accuracy', values2,
        real_target
    )

    tbl.calls_and_deviation(
        results,
        map(lambda v: v.__name__, values1),
        map(lambda v: '%.0E' % v, values2),
        'lambda'
    )


def part3_restart(base_params, real_target, new_v):
    params = deepcopy(base_params)

    values = np.linspace(1e-5, 0.5, 128)

    results = feed_values_1d(params, 'restart_lambda_threshold', values, real_target)

    img.calls_and_deviation(
        values,
        [r['calls'] for r in results],
        [r['deviation'] for r in results],
        'Restart λ threshold', None, new_v,
        None,
        'restart'
    )


def part5_termination(base_params, real_target):
    values1 = (TerminationCriterion.X_AND_F_CHANGE, TerminationCriterion.S_NORM)
    values2 = (1e-7, 1e-8, 1e-9, 1e-10, 1e-11)

    results = feed_values_2d(
        base_params,
        'termination_criterion', values1,
        'accuracy', values2,
        real_target
    )

    tbl.calls_and_deviation(
        results,
        map(lambda v: v.value, values1),
        map(lambda v: '%.0E' % v, values2),
        'termination'
    )


def part4_derivation(base_params, real_target):
    values1 = (DerivationMethod.LEFT_DIFF, DerivationMethod.RIGHT_DIFF, DerivationMethod.SYM_DIFF)
    values2 = (1e+3, 1e+2, 1e+1, 1, 1e-1, 1e-2, 1e-3)

    results = feed_values_2d(
        base_params,
        'derivation_method', values1,
        'derivation_h', values2,
        real_target
    )

    tbl.calls_and_deviation(
        results,
        map(lambda v: v.value, values1),
        map(lambda v: '%.0E' % v, values2),
        'derivation'
    )


# === HELPERS ===
def render_search_path(base_params, real_target, title, filename):
    output = []
    base_params['func'].calls = 0
    x, _ = fletcher_reeves(**base_params, output_receiver=lambda **kwargs: output.append(kwargs))
    img.search_path(base_params['func'], real_target, [r['x'] for r in output], title, filename)
    print(
        'Rendered "%s" (%s). Target function calls: %i. Deviation: %.2E' % (
            title, filename, base_params['func'].calls, np.linalg.norm(x - real_target)
        )
    )


def feed_values_2d(base_params, change_param1, values1, change_param2, values2, real_target):
    params = deepcopy(base_params)

    results = []
    for v in values1:
        params[change_param1] = v
        results.append(feed_values_1d(params, change_param2, values2, real_target))

    return results


def feed_values_1d(base_params, change_param, values, real_target):
    results = []

    for value in values:
        output = []

        params = deepcopy(base_params)
        params[change_param] = value

        params['func'].calls = 0

        x, f = fletcher_reeves(
            **params,
            output_receiver=lambda **kwargs: output.append(kwargs)
        )

        results.append({
            'output': output,
            'x': x, 'f': f,
            'deviation': np.linalg.norm(x - real_target),
            'calls': params['func'].calls
        })

    return results