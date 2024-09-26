import torch

config = {
    "num_envs": 5_000,
    "device": "cpu",
    "seed": 0,
    "sim_dt": 0.005,
    "policy_dt": 0.01,
    "max_effort": 10,
    "range": {
        "x": (-10, 10),
        "y": (-10, 10),
        "vx": (-1, 1),
        "vy": (-1, 1),
    },
    "target_distance": 0.1,
    "target_speed": 0.0,
    "max_time": 20,
    "max_steps": 10_000_000,
    "policy_cls": "MlpPolicy",
    "policy_kwargs": dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[32, 32],
    ),
    "verbose": 1,
    "video_interval": 100,
}
