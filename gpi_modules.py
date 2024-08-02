import jax 
import jax.numpy as jnp
import equinox as eqx
from util import TrainBatch

def calculate_gae(trajectory_batch, last_value, hyperparams):
    def _calculate_gae(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        value, reward, done = (
            transition.value,
            transition.reward,
            transition.done,
        )
        delta = reward + hyperparams.gamma * next_value * (1 - done) - value
        gae = delta + hyperparams.gamma * hyperparams.gae_lambda * (1 - done) * gae
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

        if hyperparams.clip_coef_vf != 0:
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
def policy_loss_ppo(params, hyperparams, trajectory_batch, advantages):
    
    action_dist = jax.vmap(params)(trajectory_batch.observation)
    log_prob = action_dist.log_prob(trajectory_batch.action)

    ratio = jnp.exp(log_prob - trajectory_batch.log_prob)
    _advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    actor_loss1 = _advantages * ratio
    actor_loss2 = (
        jnp.clip(
            ratio, 1.0 - hyperparams.clip_coef, 1.0 + hyperparams.clip_coef
        ) * _advantages
    )
    actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()

    if True: # use entropy
        entropy = action_dist.entropy().mean()
        actor_loss -= hyperparams.ent_coef * entropy

    return actor_loss

@eqx.filter_grad
def policy_loss_a2c(params, hyperparams, trajectory_batch, advantages):
    
    action_dist = jax.vmap(params)(trajectory_batch.observation)
    log_prob = action_dist.log_prob(trajectory_batch.action)

    if False: # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    actor_loss = -(advantages * log_prob).mean()

    if True: # use entropy
        entropy = action_dist.entropy().mean()
        actor_loss -= hyperparams.ent_coef * entropy

    return actor_loss

@eqx.filter_grad
def policy_loss_sac(params, hyperparams, train_data: TrainBatch, critic_output):
    
    curr_action_dist = jax.vmap(params)(train_data.observation)
    curr_action_probs = curr_action_dist.probs
    curr_action_probs_log = jnp.log(curr_action_probs + 1e-8)
    # q_1_outputs_curr = jax.vmap(new_critic1)(batch["observation"])
    # q_2_outputs_curr = jax.vmap(new_critic2)(batch["observation"])
    # q_values_curr = jnp.minimum(q_1_outputs_curr, q_2_outputs_curr)
    q_values_curr = critic_output

    loss = -jnp.mean(
        (curr_action_probs * (q_values_curr - (0.2 * curr_action_probs_log))).sum(axis=-1)
    )

    return loss