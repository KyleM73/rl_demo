import numpy as np


def double_integrator_dynamics(
        x: np.ndarray,
        u: np.ndarray,
        **kwargs,
) -> np.ndarray:
    """Double Integrator Dynamics
    dot x = A @ x + B @ u
    """
    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.float32)
    B = np.array([
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1],
    ], dtype=np.float32)
    return np.einsum("ij,kj->ki", A, x) + np.einsum("ij,kj->ki", B, u - np.array([0, 9.8]))
