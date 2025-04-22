import unittest

from numpy import isclose

from methods.interval_methods import *


ATOL = 1e-3


class Test(unittest.TestCase):

    def test_sven_move_right(self):
        func = lambda x: (100 - x) ** 2
        x0 = 30
        step = 5

        result = sven(func, x0, step)

        correct_xs = (65, 105, 145)
        correct_fs = tuple(func(x) for x in correct_xs)

        for i in range(3):
            self.assertTrue(isclose(result[0][i], correct_xs[i], atol=ATOL))  # Check x
            self.assertTrue(isclose(result[1][i], correct_fs[i], atol=ATOL))  # Check f

    def test_sven_move_left(self):
        func = lambda x: x * (2 * x - 3)
        x0 = 3.5
        step = 0.1

        result = sven(func, x0, step)

        correct_xs = (-1.2, 0.4, 2)
        correct_fs = tuple(func(x) for x in correct_xs)

        for i in range(3):
            self.assertTrue(isclose(result[0][i], correct_xs[i], atol=ATOL))  # Check x
            self.assertTrue(isclose(result[1][i], correct_fs[i], atol=ATOL))  # Check f

    def test_golden_section(self):
        func = lambda x: x * (2 * x - 3)
        interval_xs = (-1.20, 0.40, 2.00)
        interval_fs = tuple(func(x) for x in interval_xs)
        accuracy = 0.01

        actual_x, actual_f = golden_section(func, interval_xs, interval_fs, accuracy)

        expected_x = 0.75
        expected_f = func(expected_x)

        self.assertTrue(isclose(expected_x, actual_x, atol=ATOL))  # Check x
        self.assertTrue(isclose(expected_f, actual_f, atol=ATOL))  # Check f

    def test_dsk_powell(self):
        func = lambda x: x * (2 * x - 3)
        interval_xs = (-1.20, 0.40, 2.00)
        interval_fs = tuple(func(x) for x in interval_xs)
        accuracy = 0.01

        actual_x, actual_f = dsk_powell(func, interval_xs, interval_fs, accuracy)

        expected_x = 0.75
        expected_f = func(expected_x)

        self.assertTrue(isclose(expected_x, actual_x, atol=ATOL))  # Check x
        self.assertTrue(isclose(expected_f, actual_f, atol=ATOL))  # Check f


if __name__ == '__main__':
    unittest.main()