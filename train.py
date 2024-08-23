import jax
import numpy as np
import equinox as eqx
import gymnax
# import navix
import wandb
wandb.require("core")
# env = navix.make('Navix-DoorKey-5x5-v0')
# env = NavixWrapper(env)

from gpi_algorithm import GpiAlgorithm, GpiHyperparams
from env_wrappers import GymnaxWrapper, NavixWrapper, LogWrapper
from default_params import DEFAULT_A2C_PARAMS, DEFAULT_PPO_PARAMS
project_name = "default"
def log_multiple_runs(metrics, config):
    num_runs = metrics["timestep"].shape[0]
    num_envs = metrics["num_envs"].flatten()[0]

    for run in range(num_runs):
        rewards = metrics["returned_episode_returns"][run][metrics["returned_episode"][run]]
        timesteps = metrics["timestep"][run][metrics["returned_episode"][run]] * num_envs
        wandb_run = wandb.init(
            entity="FelixAndKoen",
            project=project_name, 
            config=config
        )
        rewards_start = rewards[:len(rewards) % 5]
        timesteps_start = timesteps[:len(rewards) % 5]
        rewards = rewards[len(rewards) % 5:].reshape(-1, 5).mean(axis=1)
        timesteps = timesteps[len(timesteps) % 5:].reshape(-1, 5).mean(axis=1)
        rewards = np.concatenate([rewards_start, rewards])
        timesteps = np.concatenate([timesteps_start, timesteps])
        for i in range(len(rewards)):
            wandb_run.log({
                "reward": rewards[i], 
                "rollout/ep_rew_mean": rewards[i], # SB3 compatibility
                "global_step": timesteps[i],

            })
        wandb_run.finish()


general_params = {
    "total_timesteps": 1_000_000,
    "num_envs": 4,
    "learning_rate_init": 2.5e-3,
    "anneal_lr": True,
    "gamma": 0.995,
    "debug": False,
    "alpha_init": -1.6,
    "entropy_target": 0.5,
    "kl_coef": 0.2,
    "max_grad_norm": 0.5,
    "tau": 0.05,
    "gae_lambda": 0.95,
    "clip_coef": 0.2,
    "clip_coef_vf": 0,
    "vf_coef": 0.25,
    "num_steps": 128,
    # Buffer: Only used when off_policy=True
    "buffer_max_size": 2500,
    "buffer_sample_size": 64,
    "orthogonal_init": True,

    "learn_alpha": False,
    "num_minibatches": 1,
    "update_epochs": 1,
    "dual_critics": False,
    "use_target_networks": False,
    "normalize_advantages": False,
}
ppo_default = general_params.copy()
ppo_default.update({
    "use_policy_network": True,
    "policy_regularizer": ("ppo", "add_entropy"),
    "off_policy": False,
    "use_Q_critic": False,
    "exploration_method": "sampling",
})
a2c_default = general_params.copy()
a2c_default.update({
    "use_policy_network": True,
    "policy_regularizer": ("a2c",),
    "off_policy": False,
    "use_Q_critic": False,
    "exploration_method": "sampling",
})
sac_default = general_params.copy()
sac_default.update({
    "use_policy_network": True,
    "policy_regularizer": ("sac",),
    "off_policy": True,
    "use_Q_critic": True,
    "q_target_objective": "ESARSA",
    "exploration_method": "sampling",
})
dqn_default = general_params.copy()
dqn_default.update({
    "use_policy_network": False,
    "off_policy": True,
    "use_Q_critic": True,
    "q_target_objective": "QLEARNING",
    "exploration_method": "egreedy",
})

env_ids = ["CartPole-v1", "Acrobot-v1", "Catch-bsuite", "MountainCar-v0"]
seed_num = 10
num_timesteps = 1_000_000
for env_id in env_ids:
    env, env_params = gymnax.make(env_id)
    env = GymnaxWrapper(env)
    env = LogWrapper(env)
    key = jax.random.PRNGKey(0)
    seed_keys = jax.random.split(key, seed_num)

    # for algo_name in ["A2C_gpi", "PPO_gpi"]:
    #     project_name = "sb3_sanity_check"
    #     if algo_name == "A2C_gpi":
    #         if env_id == "Catch-bsuite": continue
    #         params = DEFAULT_A2C_PARAMS
    #     elif algo_name == "PPO_gpi":
    #         params = DEFAULT_PPO_PARAMS
    #     params = params.replace(
    #         total_timesteps=num_timesteps,
    #         debug=False,
    #     )
    #     agents, metrics = eqx.filter_vmap(GpiAlgorithm.train, in_axes=(0, None, None))(seed_keys, env, params)
    #     if env_id == "Catch-bsuite": 
    #         env_id = "bsuite/catch-v0"
    #     config = params.__dict__
    #     config.update({"env_id": env_id, "algo": algo_name})
    #     log_multiple_runs(metrics, config)

    for algo_name, params in zip(["A2C_gpi", "PPO_gpi", "SAC_gpi", "DQN_gpi"], [a2c_default, ppo_default, sac_default, dqn_default]):
        algo_name += "_default"
        project_name = "default-vs-engineered"
        params = GpiHyperparams(params)
        agents, metrics = eqx.filter_vmap(GpiAlgorithm.train, in_axes=(0, None, None))(seed_keys, env, params)
        if env_id == "Catch-bsuite": 
            env_id = "bsuite/catch-v0"
        log_multiple_runs(metrics, params.__dict__.update({"env_id": env_id, "algo": algo_name}))