import numpy as np

from custom_types import *


GOLDEN_SECTION_A = 2 / (3 + np.sqrt(5))
NOT_UNIMODAL_ERROR = ValueError('Function isn\'t unimodal')


# =======================================================================================
def sven(func: Callable, x0: float, delta0: float):
    """
    :param func: Target unimodal function
    :param x0: Starting point
    :param delta0: Starting step
    :return:
        (a, (a+b)/2, b) - interval that contains the min point
        (f(a), f((a+b)/2), f(b)) - function values
    """

    xs = [x0]
    fs = [func(x0)]

    x_left = x0 - delta0
    x_right = x0 + delta0
    f_left = func(x_left)
    f_right = func(x_right)

    # Choose direction
    delta = delta0
    if f_left >= fs[0] <= f_right:
        return (x_left, x0, x_right), (f_left, fs[0], f_right)
    elif f_left >= fs[0] >= f_right:
        xs.append(x_right)
        fs.append(f_right)
        delta *= 1
    elif f_left <= fs[0] <= f_right:
        xs.append(x_left)
        fs.append(f_left)
        delta *= -1
    else:
        raise NOT_UNIMODAL_ERROR

    # Move forward
    while True:
        delta *= 2

        new_x = xs[-1] + delta
        new_f = func(new_x)

        xs.append(new_x)
        fs.append(new_f)

        if fs[-1] >= fs[-2]:
            break

    # for i in range(len(xs)):
    #     print('xi: %0.3f;\tfi: %0.3f' % (xs[i], fs[i]))

    # Move back and choose the interval
    half_delta_back_x = xs[-1] - delta / 2
    half_delta_back_f = func(half_delta_back_x)

    if half_delta_back_f < fs[-2]:
        result_xs = [xs[-2], half_delta_back_x, xs[-1]]
        result_fs = [fs[-2], half_delta_back_f, fs[-1]]
    else:
        result_xs = [xs[-3], xs[-2], half_delta_back_x]
        result_fs = [fs[-3], fs[-2], half_delta_back_f]

    if delta < 0:
        result_xs.reverse()
        result_fs.reverse()

    return tuple(result_xs), tuple(result_fs)


# =======================================================================================
def golden_section(func: Callable, xs: tuple, fs: tuple, accuracy: float):
    """
    :param func: Target unimodal function
    :param xs: (a, b) - interval that contains the min point
    :param fs: (f(a), f(b)) - function values
    :param accuracy: Target interval length
    :return: x* and f(x*)
    """

    length = xs[-1] - xs[0]

    left_golden_x = xs[0] + GOLDEN_SECTION_A * length
    right_golden_x = xs[0] + (1 - GOLDEN_SECTION_A) * length
    left_golden_f = func(left_golden_x)
    right_golden_f = func(right_golden_x)

    return _golden_section(
        func,
        (xs[0], left_golden_x, right_golden_x, xs[-1]),
        (fs[0], left_golden_f, right_golden_f, fs[-1]),
        length,
        accuracy
    )


def _golden_section(func, xs, fs, length, accuracy):
    # print('xs: %s;\tfs: %s\tL: %0.3f' % (xs, fs, length))

    if length <= accuracy:
        return (xs[0] + xs[-1]) / 2, (fs[0] + fs[-1]) / 2

    if fs[1] < fs[2]:
        return _golden_section_left(
            func,
            xs[0], xs[1], xs[2],
            fs[0], fs[1], fs[2],
            accuracy
        )
    else:
        return _golden_section_right(
            func,
            xs[1], xs[2], xs[3],
            fs[1], fs[2], fs[3],
            accuracy
        )


def _golden_section_left(func, left_x, right_golden_x, right_x, left_f, right_golden_f, right_f, accuracy):
    length = right_x - left_x
    left_golden_x = left_x + 0.382 * length
    left_golden_f = func(left_golden_x)
    return _golden_section(
        func,
        (left_x, left_golden_x, right_golden_x, right_x),
        (left_f, left_golden_f, right_golden_f, right_f),
        length,
        accuracy
    )


def _golden_section_right(func, left_x, left_golden_x, right_x, left_f, left_golden_f, right_f, accuracy):
    length = right_x - left_x
    right_golden_x = left_x + 0.618 * length
    right_golden_f = func(right_golden_x)
    return _golden_section(
        func,
        (left_x, left_golden_x, right_golden_x, right_x),
        (left_f, left_golden_f, right_golden_f, right_f),
        length,
        accuracy
    )


# =======================================================================================
def dsk_powell(func: Callable, xs: tuple, fs: tuple, accuracy: float):
    """
    :param func: Target unimodal function
    :param xs: (a, (a+b)/2, b) - interval that contains the min point
    :param fs: (f(a), f((a+b)/2), f(b))
    :param accuracy: Target interval length
    :return: x* and f(x*)
    """

    approx_x, approx_f = _dsk_powell_approx(func, xs, fs)

    if abs(xs[1] - approx_x) <= accuracy and abs(fs[1] - approx_f) <= accuracy:
        return approx_x, approx_f

    if approx_x < xs[1]:
        return dsk_powell(func, (xs[0], approx_x, xs[1]), (fs[0], approx_f, fs[1]), accuracy)
    else:
        return dsk_powell(func, (xs[1], approx_x, xs[2]), (fs[1], approx_f, fs[2]), accuracy)


def _dsk_powell_approx(func, xs, fs):
    a1 = (fs[1] - fs[0]) / (xs[1] - xs[0])
    a2 = ((fs[2] - fs[0]) / (xs[2] - xs[0]) - a1) / (xs[2] - xs[1])
    approx_x = (xs[0] + xs[1]) / 2 - a1 / 2 / a2
    approx_f = func(approx_x)

    if np.isnan(approx_x):
        i = np.argmin(fs)
        approx_x = xs[i]
        approx_f = fs[i]

    return approx_x, approx_f
