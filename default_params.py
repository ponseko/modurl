"""
    Equivalent params for different algorithms as used by stable-baselines3
"""

from gpi_algorithm import GpiHyperparams

DEFAULT_A2C_PARAMS = GpiHyperparams({
    "use_policy_network": True,
    "policy_regularizer": ("a2c",),
    "dual_critics": False,
    "use_Q_critic": False,
    "use_target_networks": False,
    "off_policy": False,
    "normalize_advantages": False,
    "q_target_objective": "ESARSA",

    "learning_rate_init": 0.0007,
    "anneal_lr": False,
    "num_steps": 5,
    "gamma": 0.99,
    "gae_lambda": 1.0,
    "alpha_init": 0.00,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    
    "num_envs": 8,
    "num_minibatches": 1,
    "update_epochs": 1,
    "clip_coef_vf": 0 # off
})

DEFAULT_PPO_PARAMS = GpiHyperparams({
    "use_policy_network": True,
    "policy_regularizer": ("ppo", "add_entropy"),
    "dual_critics": False,
    "use_Q_critic": False,
    "use_target_networks": False,
    "off_policy": False,
    "normalize_advantages": True,
    "learn_alpha": False,
    "orthogonal_init": True,

    "learning_rate_init": 3e-4,
    "anneal_lr": False,
    "num_steps": 2048,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "alpha_init": 0.00,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "ent_coef": 0.0,
    
    "num_envs": 8,
    "num_minibatches": 32,
    "update_epochs": 10,
    "clip_coef_vf": 0, # off
    "clip_coef": 0.2,
})

DEFAULT_PPO_BUT_NOT_PARAMS = GpiHyperparams({
    "use_policy_network": True,
    "policy_regularizer": ("ppo"),
    "dual_critics": False,
    "use_Q_critic": False,
    "use_target_networks": False,
    "off_policy": False,
    "normalize_advantages": True,
    "q_target_objective": "ESARSA",

    "learning_rate_init": 3e-4,
    "anneal_lr": False,
    "num_steps": 2048,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "alpha_init": 0.00,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "ent_coef": 0.0,
    
    "num_envs": 1,
    "num_minibatches": 32,
    "update_epochs": 10,
    "clip_coef_vf": 0 # off
})