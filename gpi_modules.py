import jax 
import jax.numpy as jnp
import equinox as eqx
from util import TrainBatch

def calculate_gae_returns(trajectory_batch, last_value, hyperparams):
    def _calculate_gae(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        value, reward, done = (
            transition.value,
            transition.reward,
            transition.done,
        )
        advantage = reward + hyperparams.gamma * next_value * (1 - done) - value
        gae = advantage + hyperparams.gamma * hyperparams.gae_lambda * (1 - done) * gae
        return (gae, value), (gae, gae + value)
    
    _, (advantages, returns) = jax.lax.scan(
        _calculate_gae,
        (jnp.zeros_like(last_value), last_value),
        trajectory_batch,
        reverse=True,
        unroll=16
    )

    return advantages, returns

@eqx.filter_grad
def q_loss(critics, hyperparams, train_data: TrainBatch):
    total_q_loss = 0

    for critic in critics:
        q_out = jax.vmap(critic)(train_data.observation)
        idx1 = jnp.arange(q_out.shape[0])
        selected_q_values = q_out[idx1, train_data.action]
        q_loss = jnp.mean((selected_q_values - train_data.targets) ** 2)
        total_q_loss += q_loss

    return total_q_loss



@eqx.filter_grad
def value_loss(critics, hyperparams, train_data: TrainBatch):
    
    total_value_loss = 0
    for critic in critics:
        value = jax.vmap(critic)(train_data.observation)
        value_loss = jnp.square(value - train_data.returns).mean()

        if hyperparams.clip_coef_vf > 0:
            assert train_data.value is not None
            value_clipped = train_data.value + (
                jnp.clip(
                    value - train_data.value, -hyperparams.clip_coef_vf, hyperparams.clip_coef_vf
                )
            )
            value_clipped_losses = jnp.square(value_clipped - train_data.returns)

            # TODO: Should this be an optional component? (e.g. pessimistic updates)
            value_loss = jnp.maximum(value_loss, value_clipped_losses).mean()

        total_value_loss += value_loss

    return hyperparams.vf_coef * total_value_loss

@eqx.filter_grad
def policy_loss(params, hyperparams, train_data: TrainBatch, critic_output, old_policy, alpha):
    if hyperparams.normalize_advantages:
        advantages = (train_data.advantages - train_data.advantages.mean()) / (train_data.advantages.std() + 1e-8)
    else: 
        advantages = train_data.advantages

    action_dist = jax.vmap(params)(train_data.observation)

    actor_loss = 0

    if "a2c" in hyperparams.policy_regularizer:
        log_prob = action_dist.log_prob(train_data.action)
        a2c_loss = -(advantages * log_prob).mean()
        actor_loss += a2c_loss

    if "ppo" in hyperparams.policy_regularizer:
        log_prob = action_dist.log_prob(train_data.action)
        ratio = jnp.exp(log_prob - train_data.log_prob)
        ppo_loss1 = advantages * ratio
        ppo_loss2 = (
            jnp.clip(
                ratio, 1.0 - hyperparams.clip_coef, 1.0 + hyperparams.clip_coef
            ) * advantages
        )
        actor_loss += -jnp.minimum(ppo_loss1, ppo_loss2).mean()

    if "sac" in hyperparams.policy_regularizer:
        curr_action_probs = action_dist.probs
        curr_action_probs_log = jnp.log(curr_action_probs + 1e-8)
        q_values_curr = critic_output
        sac_loss = -jnp.mean(
            (curr_action_probs * (q_values_curr - (alpha() * curr_action_probs_log))).sum(axis=-1)
        )
        actor_loss += sac_loss

    if "kl" in hyperparams.policy_regularizer:
        old_action_dist = jax.vmap(old_policy)(train_data.observation)
        kl_div = jnp.mean(old_action_dist.kl_divergence(action_dist))
        actor_loss += hyperparams.kl_coef * kl_div

    if "add_entropy" in hyperparams.policy_regularizer:
        entropy = action_dist.entropy().mean()
        actor_loss -= alpha() * entropy

    return actor_loss

@eqx.filter_grad
def alpha_loss(params, hyperparams, action_dist):
    # def target_entropy_scale(count):
    #     frac = (
    #         1.0
    #         - count / (hyperparams.total_timesteps // hyperparams.update_every)
    #     )
    #     return (hyperparams.target_entropy_scale_start * frac) + 0.01
    
    action_probs = action_dist.probs
    action_probs_log = jnp.log(action_probs + 1e-8)
    target_entropy = -(hyperparams.alpha_init) * jnp.log(1 / params.num_actions)
    return -jnp.mean(
        jnp.log(params()) 
        * ((action_probs * action_probs_log) 
        + target_entropy)
    )