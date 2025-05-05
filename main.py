import numpy as np

from methods.derivation_methods import DerivationMethod
from methods.gradient_methods import fletcher_reeves, Modification, TerminationCriterion
from methods.interval_methods import dsk_powell, golden_section
from research import no_constraints
from research import with_constraints
from research.utils import call_counter

# === TARGET FUNCTION ===
X0 = np.array((-1.2, 0), dtype=np.float64)
REAL_TARGET_X = np.array((1, 1), dtype=np.float64)
REAL_TARGET_F = 0


@call_counter
def root_func(x1, x2):
    return (10 * (x1 - x2) ** 2 + (x1 - 1) ** 2) ** (1/4)


# === RESEARCH ===
def main():
    params = {
        'derivation_method': DerivationMethod.SYM_DIFF, 'derivation_h': 0.1,
        'lambda_method': dsk_powell, 'delta_lambda': 0.1, 'lambda_accuracy': 0.01,
        'modification': Modification.FLETCHER_REEVES,
        'termination_criterion': TerminationCriterion.X_AND_F_CHANGE, 'accuracy': 1e-5,
        'max_iter': 10000
    }

    params = no_constraints.research(root_func, X0, params, REAL_TARGET_X, REAL_TARGET_F)

    # params = {
    #     'derivation_method': DerivationMethod.SYM_DIFF, 'derivation_h': 0.1,
    #     'lambda_method': dsk_powell, 'delta_lambda': 0.31, 'lambda_accuracy': 1e-6,
    #     'modification': Modification.POLAK_RIBIERE,
    #     'termination_criterion': TerminationCriterion.X_AND_F_CHANGE, 'accuracy': 1e-8,
    #     'max_iter': 10000
    # }

    output = []
    root_func.calls = 0
    x, f = fletcher_reeves(root_func, X0, **params, output_receiver=lambda **kwargs: output.append(kwargs))
    print('NO CONSTRAINTS BEST RESULT:')
    print('X: %s, F: %.8f, iterations: %i, calls: %i\n' % (x, f, output[-1]['iter_n'], root_func.calls))

    with_constraints.research(root_func, X0, params, REAL_TARGET_X, REAL_TARGET_F)


if __name__ == '__main__':
    main()