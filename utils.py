from typing import Callable

import numpy as np


def rk4(f: Callable, x: np.ndarray, dt: float, **kwargs) -> np.ndarray:
    k1 = f(x, **kwargs)
    k2 = f(x + dt * k1 / 2, **kwargs)
    k3 = f(x + dt * k2 / 2, **kwargs)
    k4 = f(x + dt * k3, **kwargs)
    x += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x
