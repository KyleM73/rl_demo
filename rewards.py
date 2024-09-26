import numpy as np


def double_integrator_rewards(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Double Integrator Reward Function
    State:
    x: [x, y, dot x, dot y, Tx, Ty, gx, gy] -> [num_envs, 8]
    Action:
    u: [Tx, Ty] -> [num_envs, 2]
    """
    # task rewards
    pose_error = np.linalg.norm(x[:, 6:8] - x[:, 0:2], axis=1)
    tracking_std = 10
    goal_tracking = np.exp(-pose_error ** 2 / tracking_std)
    goal_reached = np.where(pose_error < 0.1, 1, 0)

    # regularizing rewards
    effort_penalty = -np.linalg.norm(u, axis=1) ** 2
    action_rate = -np.linalg.norm(u - x[:, 4:6], axis=1)

    return 5 * goal_tracking + 5 * goal_reached + \
        0.0001 * effort_penalty + 0.0001 * action_rate
