import jax 
import optax
import flashbax
import jax.numpy as jnp
import chex
import wandb

@chex.dataclass(frozen=True)
class Transition:
    observation: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    value: chex.Array
    log_prob: chex.Array
    info: chex.Array
    next_observation: chex.Array

@chex.dataclass(frozen=True)
class TrainBatch(Transition):
    advantages: chex.Array = None
    returns: chex.Array = None
    targets: chex.Array = None

def log_callback(info):
    num_envs = info["num_envs"]
    return_values = info["returned_episode_returns"][info["returned_episode"]]
    timesteps = info["timestep"][info["returned_episode"]] * num_envs
    for t in range(len(timesteps)): 
        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")

def wandb_callback(info):
    if wandb.run is None:
        raise wandb.Error(
            """
                wandb logging is enabled, but wandb.run is not defined.
                Please initialize wandb before using this callback.
            """
        )
    num_envs = info["num_envs"]
    return_values = info["returned_episode_returns"][info["returned_episode"]]
    if len(return_values) == 0:
        return # no episodes finished
    timesteps = info["timestep"][info["returned_episode"]] * num_envs
    global_timestep = timesteps[0]
    average_return = return_values.mean()
    wandb.log({
        f"global_step": global_timestep,
        f"average episodic return": average_return,
        f"rollout/ep_rew_mean": average_return, # SB3 compatibility
    })
        


def set_up_replay_buffer(hyperparams, env):
    dummy_key = jax.random.PRNGKey(0)
    _, dummy_env_state = env.reset(dummy_key)
    try:
        action_dummy = env.action_space().sample(dummy_key)
    except TypeError:
        action_dummy = env.action_space.sample(dummy_key)
    (obs_dummy, reward_dummy, terminated_dummy, truncated_dummy, info_dummy), env_state = env.step(dummy_key, dummy_env_state, action_dummy)
    dummy_transition = TrainBatch(
        observation=obs_dummy,
        action=action_dummy,
        reward=reward_dummy,
        done=terminated_dummy,
        log_prob=jnp.zeros_like(action_dummy, dtype=jnp.float32),
        value=jnp.zeros_like(reward_dummy, dtype=jnp.float32),
        info=info_dummy,
        next_observation=obs_dummy,
        advantages=jnp.zeros_like(action_dummy, dtype=jnp.float32),
        returns=jnp.zeros_like(reward_dummy, dtype=jnp.float32),
        targets=jnp.zeros_like(reward_dummy, dtype=jnp.float32),
    )
    if not hyperparams.off_policy:
        max_length = hyperparams.num_envs * hyperparams.num_steps
        sample_batch_size = hyperparams.num_envs * hyperparams.num_steps
    else:
        max_length = hyperparams.buffer_max_size
        sample_batch_size = hyperparams.buffer_sample_size
    buffer = flashbax.make_flat_buffer(
        max_length=max_length,
        min_length=hyperparams.num_steps - 1,
        sample_batch_size=sample_batch_size,
        add_sequences=True,
        add_batch_size=hyperparams.num_envs,
    )
    buffer_state = buffer.init(dummy_transition)

    return buffer, buffer_state

def set_up_optimizer(hyperparams, actor, critic, alpha):
    num_update_steps = int(
        hyperparams.total_timesteps 
        * hyperparams.num_minibatches * hyperparams.update_epochs 
        // (hyperparams.num_envs * hyperparams.num_steps)
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(hyperparams.max_grad_norm),
        optax.adam(
            learning_rate=optax.polynomial_schedule(
                init_value=hyperparams.learning_rate, end_value=0., power=1, 
                transition_steps=num_update_steps
            ) if hyperparams.anneal_lr else hyperparams.learning_rate,
            eps=1e-5
        ),
    )
    actor_optimizer_state = optimizer.init(actor)
    critic_optimizer_state = optimizer.init(critic)
    alpha_optimizer_state = optimizer.init(alpha)

    return optimizer, actor_optimizer_state, critic_optimizer_state, alpha_optimizer_state