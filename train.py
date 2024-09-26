from stable_baselines3 import PPO
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import EveryNTimesteps

from callbacks import WandBVideoCallback
from config import config
from env import RK4Env

run = wandb.init(
    project="rk4_env",
    config=config,
    sync_tensorboard=True,
    save_code=True,
)

model = PPO(
    config["policy_cls"],
    RK4Env(config["num_envs"], config=config),
    policy_kwargs=config["policy_kwargs"],
    verbose=config["verbose"],
    device=config["device"],
    tensorboard_log=f"runs/{run.id}"
)
try:
    model.learn(
        total_timesteps=config["max_steps"],
        callback=[
            WandbCallback(
                verbose=config["verbose"],
                model_save_path=None,  # f"models/{run.id}"
                model_save_freq=0,  # 100
                gradient_save_freq=0,  # 100
            ),
            EveryNTimesteps(
                n_steps=config["video_interval"],
                callback=WandBVideoCallback(),
            )
        ],
    )
except Exception as e:
    print(f"Exception: {e}")
run.finish()
