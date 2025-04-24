import numpy as np

from methods.derivation_methods import DerivationMethod
from methods.gradient_methods import fletcher_reeves, Modification, TerminationCriterion
from methods.interval_methods import dsk_powell, golden_section
from research import no_constraints
from research import with_constraints


# === TARGET FUNCTION ===
X0 = np.array((-1.2, 0), dtype=np.float64)
REAL_TARGET = np.array((1, 1), dtype=np.float64)


def call_counter(f):
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return f(*args, **kwargs)

    wrapped.calls = 0
    return wrapped


@call_counter
def root_func(x1, x2):
    return (10 * (x1 - x2) ** 2 + (x1 - 1) ** 2) ** (1/4)


# === RESEARCH ===
def main():
    params = {
        'func': root_func,
        'x0': X0,
        'derivation_method': DerivationMethod.SYM_DIFF, 'derivation_h': 0.1,
        'lambda_method': dsk_powell, 'delta_lambda': 0.1, 'lambda_accuracy': 0.01,
        'modification': Modification.FLETCHER_REEVES,
        'termination_criterion': TerminationCriterion.X_AND_F_CHANGE, 'accuracy': 1e-5,
        'max_iter': 10000
    }

    # params = no_constraints.research(root_func, params, REAL_TARGET)

    params = {
        'func': root_func,
        'x0': X0,
        'derivation_method': DerivationMethod.SYM_DIFF, 'derivation_h': 0.1,
        'lambda_method': dsk_powell, 'delta_lambda': 0.05, 'lambda_accuracy': 0.1,
        'modification': Modification.FLETCHER_REEVES,
        'termination_criterion': TerminationCriterion.X_AND_F_CHANGE, 'accuracy': 1e-9,
        'max_iter': 10000,
        'restart_lambda_threshold': 0.24
    }

    with_constraints.research(params, REAL_TARGET)


if __name__ == '__main__':
    main()