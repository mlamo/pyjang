from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d


def compute_upperlimit_from_x_y(
    x_arr: np.ndarray, y_arr: np.ndarray, CL: float = 0.9
) -> float:
    """Compute the upper limit at a confidence level CL for a given posterior y_arr=P(x_arr)."""
    if np.all(y_arr == 0):
        return np.inf
    int_arr = y_arr[:-1] * (x_arr[1:] - x_arr[:-1])
    cum_arr = np.cumsum(int_arr)
    cum_arr = 1 / cum_arr[-1] * cum_arr
    f = interp1d(cum_arr, x_arr[:-1])
    limit = float(f(CL))
    # limit dangerously closed to lkl upper bound
    if limit > 0.9 * CL * x_arr[-1]:
        return np.inf
    return limit


def normalize(x_arr: np.ndarray, y_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize the posterior y_arr=P(x_arr)."""
    if np.all(y_arr == 0):
        return x_arr, y_arr
    integral = np.sum(y_arr[:-1] * (x_arr[1:] - x_arr[:-1]))
    return x_arr, 1 / integral * y_arr
