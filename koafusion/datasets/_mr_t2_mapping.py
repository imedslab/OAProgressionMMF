import numpy as np
from numba import jit, prange


@jit(nopython=True,fastmath=True)
def fit_exp_linear(xs, ys):
    """
    Taken from https://stackoverflow.com/questions/3433486/how-to-do-exponential-and-logarithmic-curve-fitting-in-python-i-found-only-poly
    Fit the function y = A * exp(B * x) to the data, return (A, B)
    From: https://mathworld.wolfram.com/LeastSquaresFittingExponential.html
    """
    S_x2_y = 0.0
    S_y_lny = 0.0
    S_x_y = 0.0
    S_x_y_lny = 0.0
    S_y = 0.0
    for (x, y) in zip(xs, ys):
        S_x2_y += x * x * y
        S_y_lny += y * np.log(y)
        S_x_y += x * y
        S_x_y_lny += x * y * np.log(y)
        S_y += y

    denom = (S_y * S_x2_y - S_x_y * S_x_y)
    if denom == 0.0:
        return np.nan, np.nan
    else:
        a = (S_x2_y * S_y_lny - S_x_y * S_x_y_lny) / denom
        b = (S_y * S_x_y_lny - S_x_y * S_y_lny) / denom
        return np.exp(a), b


@jit(parallel=True, nopython=True)
def fit_t2_map(vol, tes, nan_to=0.0, val_low=0.0, val_high=0.1):
    """
    Args:
        vol: 4D ndarray (slices, rows, cols, echoes), MESE image
        tes: 2D ndarray (slices, echoes), TEs per slice
        nan_to: float, value to set NaNs to
        val_low: lower boundary of expected T2 values, any values beyond are set to 0.0
        val_high: upper boundary of expected T2 values, any values above are set to 0.0

    Returns:
        map_t2: 3D ndarray (slices, rows, cols), T2 map
    """

    map_t2 = np.zeros((vol.shape[0], vol.shape[1], vol.shape[2]), dtype=np.float64)

    for s in range(vol.shape[0]):
        xs = tes[s]
        for i in prange(vol.shape[1]):
            for j in prange(vol.shape[2]):
                ys = vol[s, i, j, :]

                (a, b) = fit_exp_linear(xs, ys)

                if np.isnan(a) or np.isnan(b):
                    map_t2[s, i, j] = 0.0
                else:
                    t = -1.0 / b
                    if np.isnan(t):
                        map_t2[s, i, j] = nan_to
                    else:
                        if t < val_low or t > val_high:
                            map_t2[s, i, j] = 0.0
                        else:
                            map_t2[s, i, j] = t
    return map_t2
