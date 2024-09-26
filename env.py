from typing import Any, Callable, Dict, Iterable, List,  Optional, Type

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv, VecEnvStepReturn, VecEnvObs, VecEnvIndices
)

from config import config


def rk4(f: Callable, x: np.ndarray, dt: float, **kwargs) -> np.ndarray:
    k1 = f(x, **kwargs)
    k2 = f(x + dt * k1 / 2, **kwargs)
    k3 = f(x + dt * k2 / 2, **kwargs)
    k4 = f(x + dt * k3, **kwargs)
    x += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x


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
    return np.einsum("ij,kj->ki", A, x) + np.einsum("ij,kj->ki", B, u)


def reward(x: np.ndarray, u: np.ndarray) -> np.ndarray:
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


class RK4Env(VecEnv):
    metadata = {
        "render_mode": ["human"],
        "render_fps": int(1 / config["policy_dt"])
    }
    actions: np.ndarray

    def __init__(
            self,
            num_envs: int,
            num_obs: int = 8,
            num_actions: int = 2,
            config: Dict = config,
            func: Callable = double_integrator_dynamics,
            rew_func: Callable = reward,
    ) -> None:
        self.cfg = config
        self.f = func
        self.r = rew_func

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_obs,),
            dtype=np.float32,
            seed=self.cfg["seed"],
        )
        self.action_space = gym.spaces.Box(
            low=-self.cfg["max_effort"],
            high=self.cfg["max_effort"],
            shape=(num_actions,),
            dtype=np.float32,
            seed=self.cfg["seed"],
        )

        self.reset_infos: List[Dict[str, Any]] = [{} for _ in range(num_envs)]
        self._seeds: List[Optional[int]] = [None for _ in range(num_envs)]
        self._options: List[Dict[str, Any]] = [{} for _ in range(num_envs)]

        self.num_envs = num_envs
        self.buf_obs = np.zeros((self.num_envs, num_obs))
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos: List[Dict[str, Any]] = [
            {} for _ in range(self.num_envs)
        ]

        self.sim_dt = self.cfg["sim_dt"]
        self.decimation = int(self.cfg["policy_dt"] / self.cfg["sim_dt"])
        self.target_distance = self.cfg["target_distance"]
        self.target_speed = self.cfg["target_speed"]
        self.rng = np.random.default_rng(self.cfg["seed"])
        self.t = np.zeros((self.num_envs,), dtype=np.float32)
        self.max_time = self.cfg["max_time"]

        self.obs_hist = np.zeros(
            (int(self.max_time / self.cfg["policy_dt"]), num_obs),
            dtype=np.float32,
        )
        self.plot = None  # only for env[0]
        self.counter = 0

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        for _ in range(self.decimation):
            self.buf_obs[:, 0:4] = rk4(
                self.f,
                self.buf_obs[:, 0:4],
                self.sim_dt,
                u=self.actions
            )
        self.buf_rews = self.r(self.buf_obs, self.actions)
        self.buf_obs[:, 4:6] = self.actions
        self.obs_hist[self.counter] = self.buf_obs[0]
        terminated = \
            (self.buf_obs[:, 0] < self.cfg["range"]["x"][0]) | \
            (self.buf_obs[:, 0] > self.cfg["range"]["x"][1]) | \
            (self.buf_obs[:, 1] < self.cfg["range"]["y"][0]) | \
            (self.buf_obs[:, 1] > self.cfg["range"]["y"][1])
        truncated = (
            (np.linalg.norm(
                self.buf_obs[:, 0:2], axis=1
            ) < self.target_distance) &
            (np.linalg.norm(
                self.buf_obs[:, 2:4], axis=1
            ) < self.target_speed)
            ) | \
            (self.t > self.max_time)
        self.buf_dones = terminated | truncated
        for idx in range(self.num_envs):
            self.buf_infos[idx]["TimeLimit.truncated"] = \
                truncated[idx] and not terminated[idx]
            if self.buf_dones[idx]:
                self.buf_infos[idx]["terminal_observation"] = self.buf_obs[idx]
                self.buf_infos[idx]["episode"] = {
                    "r": self.buf_rews[idx],
                    "l": self.t[idx]
                }
            self.reset_infos[idx]["success"] = self.buf_dones[idx]
        self.t += self.decimation * self.sim_dt
        self.counter = min(self.counter + 1, self.obs_hist.shape[0] - 1)
        reset_idx = np.argwhere(self.buf_dones).flatten()
        if reset_idx.size > 0:
            self.buf_obs[reset_idx], self.t[reset_idx] = \
                self.reset_idx(reset_idx)

        return (
            np.copy(self.buf_obs),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            self.buf_infos.copy()  # deepcopy(self.buf_infos)
        )

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        self.step_async(actions)
        return self.step_wait()

    def reset_idx(self, indices: VecEnvIndices = None) -> VecEnvObs:
        idx = self._get_indices(indices)
        if 0 in idx and self.counter > 5:
            obs_plot = self.obs_hist[:self.counter]
            fig, ax = plt.subplots()
            ax.plot(obs_plot[:, 0], obs_plot[:, 1])  # trajectory
            ax.scatter(obs_plot[0, 0], obs_plot[0, 1])  # ic
            ax.scatter(obs_plot[0, 6], obs_plot[0, 7])  # goal
            self.plot = fig
            self.counter = 0
        x0 = self.rng.uniform(
            low=self.cfg["range"]["x"][0],
            high=self.cfg["range"]["x"][1],
            size=(len(idx), 1),
        )
        y0 = self.rng.uniform(
            low=self.cfg["range"]["y"][0],
            high=self.cfg["range"]["y"][1],
            size=(len(idx), 1),
        )
        vx0 = self.rng.uniform(
            low=self.cfg["range"]["vx"][0],
            high=self.cfg["range"]["vx"][1],
            size=(len(idx), 1),
        )
        vy0 = self.rng.uniform(
            low=self.cfg["range"]["vy"][0],
            high=self.cfg["range"]["vy"][1],
            size=(len(idx), 1),
        )
        T = np.zeros((len(idx), 2), dtype=np.float32)
        gx = self.rng.uniform(
            low=self.cfg["range"]["x"][0],
            high=self.cfg["range"]["x"][1],
            size=(len(idx), 1),
        )
        gy = self.rng.uniform(
            low=self.cfg["range"]["y"][0],
            high=self.cfg["range"]["y"][1],
            size=(len(idx), 1),
        )
        obs = np.concatenate((x0, y0, vx0, vy0, T, gx, gy), axis=1)
        t = np.zeros((len(idx),), dtype=np.float32)
        return obs, t

    def reset(self, seed=None, options=None) -> VecEnvObs:
        idx = self._get_indices(None)
        self.buf_obs, self.t = self.reset_idx(idx)
        for idx in range(self.num_envs):
            self.reset_infos[idx] = {}
        return self.buf_obs

    def render(self) -> None:
        pass

    def close(self) -> None:
        return

    def _get_indices(self, indices: VecEnvIndices) -> Iterable[int]:
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, np.ndarray):
            indices = indices.flatten().tolist()
        return indices

    def env_is_wrapped(
            self,
            wrapper_class: Type[gym.Wrapper],
            indices: VecEnvIndices = None
    ) -> List[bool]:
        idx = self._get_indices(indices)
        return [False] * len(idx)

    def env_method(
            self,
            method_name: str,
            *method_args,
            indices: VecEnvIndices = None,
            **method_kwargs
    ) -> List[Any]:
        raise NotImplementedError

    def get_attr(
            self,
            attr_name: str,
            indices: VecEnvIndices = None
    ) -> List[Any]:
        raise NotImplementedError

    def set_attr(
            self,
            attr_name: str,
            indices: VecEnvIndices = None
    ) -> List[Any]:
        raise NotImplementedError


if __name__ == "__main__":
    import time
    n = 10_000
    T = 10
    env = RK4Env(n, 8, 2, config)
    u = np.array([[1.]*n, [0.03]*n], dtype=np.float32).T
    now = time.time()
    for _ in range(T):
        obs, rew, done, info = env.step(u)
        print(obs.shape)
    print(f"{n*T/(time.time() - now)} sps")
