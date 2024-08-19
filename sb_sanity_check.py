import numpy as np
import gymnax
import wandb
import jax
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env

env_id = "CartPole-v1"
env_ids = ["CartPole-v1", "Pendulum-v1", "Acrobot-v1" "MountainCar-v0", "bsuite/catch-v0"]
seeds = np.arange(10).tolist()
for seed in seeds:
    ## SB3
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 1_000_000,
        "env_id": env_id,
    }
    wandb_run = wandb.init(project="sb3_sanity_check", config=config, sync_tensorboard=True)
    env = make_vec_env(config["env_id"], n_envs=8, seed=seed)
    model = A2C(
        config["policy_type"], 
        env, 
        stats_window_size=1, # No averaging 
        verbose=1, 
        use_rms_prop=False,
        tensorboard_log=f"runs/{wandb_run.id}",
        policy_kwargs=dict(
            share_features_extractor=False,
            full_std=False,
        ),
        seed=seed
    )
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(verbose=2),
    )
    wandb_run.finish()
    
for seed in seeds:
    ## SB3
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 1_000_000,
        "env_id": env_id,
    }
    wandb_run = wandb.init(project="sb3_sanity_check", config=config, sync_tensorboard=True)
    env = make_vec_env(config["env_id"], n_envs=8, seed=seed)
    model = PPO(
        config["policy_type"], 
        env, 
        stats_window_size=1, # No averaging 
        verbose=1,
        tensorboard_log=f"runs/{wandb_run.id}",
        policy_kwargs=dict(
            share_features_extractor=False,
            full_std=False,
        ),
        seed=seed
    )
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(verbose=2),
    )
    wandb_run.finish()
