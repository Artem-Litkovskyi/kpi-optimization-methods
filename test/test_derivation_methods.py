import unittest

from methods.derivation_methods import *


ATOL = 1e-10


class Test(unittest.TestCase):

    def test_left_difference(self):
        func = lambda x1, x2: x1 ** 2 + 2 * x2 ** 2
        x0 = np.array((2, 1), dtype=np.float64)
        h = 0.5

        actual = nabla(func, x0, h, method=DerivationMethod.LEFT_DIFF)

        f_0 = func(*x0)
        expected = np.array((
            (f_0 - ((x0[0] - h) ** 2 + 2 * x0[1] ** 2)) / h,
            (f_0 - (x0[0] ** 2 + 2 * (x0[1] - h) ** 2)) / h,
        ), dtype=np.float64)

        self.assertTrue(np.allclose(actual, expected, atol=ATOL))

    def test_right_difference(self):
        func = lambda x1, x2: x1 ** 2 + 2 * x2 ** 2
        x0 = np.array((2, 1), dtype=np.float64)
        h = 0.5

        actual = nabla(func, x0, h, method=DerivationMethod.RIGHT_DIFF)

        f_0 = func(*x0)
        expected = np.array((
            (((x0[0] + h) ** 2 + 2 * x0[1] ** 2) - f_0) / h,
            ((x0[0] ** 2 + 2 * (x0[1] + h) ** 2) - f_0) / h,
        ), dtype=np.float64)

        self.assertTrue(np.allclose(actual, expected, atol=ATOL))

    def test_symmetric_difference(self):
        func = lambda x1, x2: x1 ** 2 + 2 * x2 ** 2
        x0 = np.array((2, 1), dtype=np.float64)
        h = 0.5

        actual = nabla(func, x0, h, method=DerivationMethod.SYM_DIFF)

        expected = np.array((
            (((x0[0] - h) ** 2 + 2 * x0[1] ** 2) - ((x0[0] + h) ** 2 + 2 * x0[1] ** 2)) / 2 / h,
            ((x0[0] ** 2 + 2 * (x0[1] - h) ** 2) - (x0[0] ** 2 + 2 * (x0[1] + h) ** 2)) / 2 / h,
        ), dtype=np.float64)

        self.assertTrue(np.allclose(actual, expected, atol=ATOL))
