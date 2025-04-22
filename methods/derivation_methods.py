import numpy as np
from copy import deepcopy
from enum import Enum

from custom_types import *


class DerivationMethod(Enum):
    RIGHT_DIFF = 1
    LEFT_DIFF = 2
    SYM_DIFF = 3


def nabla(func: Callable, x0: np.ndarray, h: float, method = DerivationMethod.SYM_DIFF, f0 = None) -> np.ndarray:
    result = np.zeros_like(x0)

    if method != DerivationMethod.SYM_DIFF and f0 is None:
        f0 = func(*x0)

    for i in range(len(x0)):
        if method == DerivationMethod.LEFT_DIFF:
            tmp = deepcopy(x0)
            tmp[i] = tmp[i] - h
            result[i] = (f0 - func(*tmp)) / h
        elif method == DerivationMethod.RIGHT_DIFF:
            tmp = deepcopy(x0)
            tmp[i] = tmp[i] + h
            result[i] = (func(*tmp) - f0) / h
        else:
            tmp1 = deepcopy(x0)
            tmp1[i] = tmp1[i] - h
            tmp2 = deepcopy(x0)
            tmp2[i] = tmp2[i] + h
            result[i] = (func(*tmp1) - func(*tmp2)) / 2 / h

    return result
