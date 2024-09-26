import torch

config = {
    "num_envs": 1_000,
    "device": "cpu",
    "seed": 0,
    "sim_dt": 0.01,
    "policy_dt": 0.01,
    "max_effort": 100,
    "range": {
        "x": (-10, 10),
        "y": (0, 10),
    },
    "ic_range": {
        "x": (-5, 5),
        "y": (8, 10),
        "vx": (-1, 1),
        "vy": (-1, 0),
    },
    "target_distance": 0.1,
    "target_speed": 0.0,
    "max_time": 10,
    "max_steps": 1_000_000_000,
    "policy_cls": "MlpPolicy",
    "policy_kwargs": dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[32, 32],
    ),
    "verbose": 1,
    "rollout_steps": 50,
    "minibatch_size": 1000,
}
