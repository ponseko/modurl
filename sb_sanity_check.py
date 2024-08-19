import numpy as np
import gymnax
import wandb
import jax
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env

from gpi_algorithm import GpiAlgorithm, GpiHyperparams
from env_wrappers import GymnaxWrapper, LogWrapper
from default_params import DEFAULT_A2C_PARAMS, DEFAULT_PPO_PARAMS



seeds = np.arange(10).tolist()
for seed in seeds:
    ## SB3
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 500_000,
        "env_id": "CartPole-v1",
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


    # GPI
    env, env_params = gymnax.make("CartPole-v1")
    env = GymnaxWrapper(env)
    env = LogWrapper(env)
    key = jax.random.PRNGKey(seed)

    a2c_agent_params = GpiHyperparams(DEFAULT_A2C_PARAMS)
    config = {"algo": "gpi_a2c", **a2c_agent_params.__dict__}
    wandb_run = wandb.init(project="sb3_sanity_check", config=config)
    
    agent, metrics = GpiAlgorithm.train(key, env, hyperparams=a2c_agent_params)

    wandb_run.finish()

    
for seed in seeds:
    ## SB3
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 500_000,
        "env_id": "CartPole-v1",
    }
    wandb_run = wandb.init(project="sb3_sanity_check", config=config, sync_tensorboard=True)
    env = make_vec_env(config["env_id"], n_envs=8, seed=seed)
    model = PPO(
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


    # GPI
    env, env_params = gymnax.make("CartPole-v1")
    env = GymnaxWrapper(env)
    env = LogWrapper(env)
    key = jax.random.PRNGKey(seed)

    ppo_agent_params = GpiHyperparams(DEFAULT_PPO_PARAMS)
    config = {"algo": "gpi_a2c", **ppo_agent_params.__dict__}
    wandb_run = wandb.init(project="sb3_sanity_check", config=config)
    
    agent, metrics = GpiAlgorithm.train(key, env, hyperparams=ppo_agent_params)

    wandb_run.finish()