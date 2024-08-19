"""
    Equivalent params for different algorithms as used by stable-baselines3
"""

DEFAULT_A2C_PARAMS = {
    "policy_regularizer": ("a2c"),
    "dual_critics": False,
    "use_Q_critic": False,
    "use_target_networks": False,
    "off_policy": False,
    "normalize_advantages": False,

    "learning_rate": 0.0007,
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
}

DEFAULT_PPO_PARAMS = {
    "policy_regularizer": ("ppo", "add_entropy"),
    "dual_critics": False,
    "use_Q_critic": False,
    "use_target_networks": False,
    "off_policy": False,
    "normalize_advantages": True,

    "learning_rate": 0.0003,
    "anneal_lr": False,
    "num_steps": 2048,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "alpha_init": 0.00,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    
    "num_envs": 8,
    "num_minibatches": 32,
    "update_epochs": 10,
    "clip_coef_vf": 0 # off
}

DEFAULT_SAC_PARAMS = {
    "policy_regularizer": ("sac",),
    "dual_critics": True,
    "use_Q_critic": True,
    "use_target_networks": True,
    "off_policy": True,
    "normalize_advantages": True,

    "learning_rate": 0.0003,
    "buffer_size": 1000000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,

    "init_alpha": 0.2,
    "learn_alpha": True,
    
}
