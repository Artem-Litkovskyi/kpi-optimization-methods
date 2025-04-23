import numpy as np

import visualization.visualization as vis
from methods.derivation_methods import DerivationMethod
from methods.gradient_methods import fletcher_reeves, Modification, TerminationCriterion
from methods.interval_methods import dsk_powell


def call_counter(f):
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return f(*args, **kwargs)
    wrapped.calls = 0
    return wrapped


X0 = np.array((-1.2, 0), dtype=np.float64)
REAL_TARGET = np.array((1, 1), dtype=np.float64)


@call_counter
def root_func(x1, x2):
    return (10 * (x1 - x2) ** 2 + (x1 - 1) ** 2) ** (1/4)


def main():
    output = []

    fletcher_reeves(
        root_func, X0,
        DerivationMethod.SYM_DIFF, 0.1,
        dsk_powell, 1, 0.01,
        Modification.FLETCHER_REEVES, TerminationCriterion.NABLA_NORM, 0.01,
        output_receiver=lambda **kwargs: output.append(kwargs)
    )

    vis.search_path(
        root_func, REAL_TARGET, [r['x'] for r in output],
        'Fletcher-Reeves', 'test'
    )

    vis.surface(
        root_func, X0, REAL_TARGET, (-3, 3), (-3, 3),
        'Fletcher-Reeves', 'test'
    )


if __name__ == '__main__':
    main()