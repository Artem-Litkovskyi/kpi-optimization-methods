import unittest

from methods.gradient_methods import *
from methods.interval_methods import dsk_powell


ATOL = 1e-10


class Test(unittest.TestCase):

    @staticmethod
    def output(**kwargs):
        print(kwargs)

    def test_fletcher_reeves(self):
        func = lambda x1, x2: 2 * x1 ** 2 + x1 * x2 + 2 * x2 ** 2 + 8 * x1

        x0 = np.array((0, 0), dtype=np.float64)

        result_x, result_f = fletcher_reeves(
            func, 0.01, DerivationMethod.SYM_DIFF,
            x0,
            dsk_powell, 0.1, 0.01,
            0.01, TerminationCriterion.X_AND_F_CHANGE,
            output_receiver=self.output
        )

        correct_x = (-32/15, 8/15)
        correct_f = func(*correct_x)

        self.assertTrue(np.allclose(result_x, correct_x, atol=ATOL))  # Check x
        self.assertTrue(np.isclose(result_f, correct_f, atol=ATOL))  # Check f
