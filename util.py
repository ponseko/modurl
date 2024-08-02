import jax 
import optax
import flashbax
import jax.numpy as jnp
import chex

@chex.dataclass(frozen=True)
class Transition:
    observation: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    value: chex.Array
    log_prob: chex.Array
    info: chex.Array

@chex.dataclass(frozen=True)
class TrainBatch(Transition):
    next_observations: chex.Array = None
    advantages: chex.Array = None
    returns: chex.Array = None
    targets: chex.Array = None

def log_callback(info):
    num_envs = info["timestep"].shape[1]
    return_values = info["returned_episode_returns"][info["returned_episode"]]
    timesteps = info["timestep"][info["returned_episode"]] * num_envs
    for t in range(len(timesteps)): 
        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")


def set_up_replay_buffer(hyperparams, env, env_params = None):
    dummy_key = jax.random.PRNGKey(0)
    if env_params is None:
        env_params = env.default_params()
    _, dummy_env_state = env.reset(dummy_key, env_params)
    action_dummy = env.action_space(env_params).sample(dummy_key)
    obs_dummy, _, reward_dummy, done_dummy, info_dummy = env.step(dummy_key, dummy_env_state, action_dummy, env_params)
    dummy_transition = Transition(
        observation=obs_dummy,
        action=action_dummy,
        reward=reward_dummy,
        done=done_dummy,
        log_prob=jnp.zeros_like(action_dummy, dtype=jnp.float32),
        value=jnp.zeros_like(reward_dummy, dtype=jnp.float32),
        info=info_dummy
    )
    buffer = flashbax.make_flat_buffer(
        max_length=hyperparams.num_envs * hyperparams.num_steps * 2,
        min_length=0,
        sample_batch_size=hyperparams.buffer_sample_size,
        add_sequences=False,
        add_batch_size=hyperparams.num_envs,
    )
    buffer_state = buffer.init(dummy_transition)

    return buffer, buffer_state

def set_up_optimizer(hyperparams, actor, critic, critic_dual = None):
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

    return optimizer, actor_optimizer_state, critic_optimizer_state