from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
from typing import Tuple
import chex
import gymnax
from gymnax.environments.environment import Environment
from functools import partial
from env_wrappers import LogWrapper
import flashbax
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState as BufferState, TrajectoryBuffer as Buffer
from gpi_modules import *
from util import log_callback, set_up_replay_buffer, set_up_optimizer, Transition, TrainBatch
from networks import ActorNetwork, ValueNetwork, Q_Network, Alpha

@chex.dataclass(frozen=True)
class TrainState:
    actor: ActorNetwork
    actor_optimizer_state: optax.OptState
    buffer_state: BufferState # NOTE: Maybe also optional (None?)
    alpha: Alpha
    alpha_optimizer_state: optax.OptState

    critic: tuple[eqx.Module] | tuple[eqx.Module, eqx.Module]
    critic_optimizer_state: tuple[optax.OptState] | tuple[optax.OptState, optax.OptState]

    critic_target: tuple[eqx.Module] | tuple[eqx.Module, eqx.Module] | None = None

@chex.dataclass(frozen=True)
class GPI_hyperparams:
    """
    Hyperparameters for the any algorithm
    There will be some reduncy in this class, but it will be easier to manage
    """
    # General
    total_timesteps: int = 1e6
    learning_rate: float = 1e-3
    anneal_lr: bool = True
    gamma: float = 0.99
    num_envs: int = 6
    backend: str = "cpu" # "cpu" or "gpu"
    debug: bool = True # log values during training
    alpha_init: float = 0.2 # entropy coefficient
    learn_alpha: bool = True 
    kl_coef: float = 0.2

    # PPO
    gae_lambda: float = 0.95
    max_grad_norm: float = 100.0
    clip_coef: float = 0.2
    clip_coef_vf: float = 10.0 # Depends on the reward scaling !
    vf_coef: float = 0.25
    num_steps: int = 128 # steps per environment
    num_minibatches: int = 4 # Number of mini-batches
    update_epochs: int = 4 # K epochs to update the policy
    # batch_size: int = 0 # batch size (num_envs * num_steps)
    # minibatch_size: int = 0 # mini-batch size (batch_size / num_minibatches)
    # num_iterations: int = 0 # number of iterations (total_timesteps / num_steps / num_envs)

    # Buffer
    buffer_max_size: int = 1000
    buffer_sample_size: int = 128 * 6

    # Configuration options
    policy_regularizer: str = ("ppo", "add_entropy") # "sac", "a2c", "ppo", "kl", "add_entropy"
    dual_critics: bool = True
    use_Q_critic: bool = True # if false: use value network, else Q network
    use_gae: bool = True
    use_target_networks: bool = True
    off_policy: bool = True
    normalize_advantages: bool = True

class GpiAlgorithm(eqx.Module):

    train_state_checkpoint: TrainState
    hyperparams: GPI_hyperparams
    optimizer: optax.GradientTransformation
    buffer: Buffer

    def __init__(
        self,
        key: chex.PRNGKey,
        env: gymnax.environments.environment.Environment, # env is a gym environment in jax style TODO
        hyperparams: GPI_hyperparams = GPI_hyperparams(),
    ):
        self.hyperparams = hyperparams
        env_params = env.default_params
        observation_space = env.observation_space(env_params)
        action_space = env.action_space(env_params)
        num_actions = action_space.n

        actor_key, critic_key, dual_critic_key = jax.random.split(key, 3)
        actor = ActorNetwork(actor_key, observation_space.shape[0], [64, 64], num_actions)

        if hyperparams.use_Q_critic: 
            CriticNetwork = Q_Network
        else: CriticNetwork = ValueNetwork

        alpha = Alpha(hyperparams.alpha_init, num_actions)


        if hyperparams.dual_critics:
            critic = (
                CriticNetwork(critic_key, observation_space.shape[0], [64, 64], num_actions=num_actions),
                CriticNetwork(dual_critic_key, observation_space.shape[0], [64, 64], num_actions=num_actions)
            )
        else:
            critic = (CriticNetwork(critic_key, observation_space.shape[0], [64, 64], num_actions=num_actions),)

        if hyperparams.use_target_networks:
            critic_target = jax.tree.map(lambda x: x, critic)
        else: critic_target = None

        self.optimizer, actor_optimizer_state, critic_optimizer_state, alpha_optimizer_state = set_up_optimizer(hyperparams, actor, critic, alpha)
        self.buffer, buffer_state = set_up_replay_buffer(hyperparams, env, env_params)
        
        self.train_state_checkpoint = TrainState(
            actor=actor,
            critic=critic,
            alpha=alpha,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
            alpha_optimizer_state=alpha_optimizer_state,
            critic_target=critic_target,
            buffer_state=buffer_state
        )

    def __call__(self, key: chex.PRNGKey, observation):
        action_dist = jax.vmap(self.train_state_checkpoint.actor)(observation)
        action = action_dist.sample(seed=key)
        return action
    
    def save(self, path: str) -> None:
        eqx.tree_serialise_leaves(path, self)

    def load(self, path: str) -> GpiAlgorithm:
        return eqx.tree_deserialise_leaves(path)
    

    @classmethod
    def train(cls, rng: chex.PRNGKey, env: Environment, *, agent: GpiAlgorithm = None, hyperparams: GPI_hyperparams = None):
        """
            This method trains an agent instance and returns the trained agent.
            The number of training steps are determined by the provided hyperparameters.
            if an agent object is provided, it will continue training from its checkpointed train state --
            Creating a new agent will also provide an initial train state, so a new agent can be trained from scratch.
            Alternatively, the train method may be provided an instance of hyperparemeters, in which case 
            a new agent will be trained from scratch

            @example use:
                env = x.make("some_env")
                agent = GPIAlgorithm.train(rng, env)

            @alternative use:
                env = x.make("some_env")
                agent = GPIAlgorithm(rng, env)
                agent = GPIAlgorithm.train(rng, agent, env)
        """

        def get_critic_output(critics: Tuple, observation):
            """ 
                Get value from critic network, either a single critic or the minimum of two critics 
                Provide target networks if required
            """
            for critic in critics:
                assert isinstance(critic, eqx.Module)
            if len(critics) == 1:
                return jax.vmap(critics[0])(observation)
            else: # dualing critic
                return jnp.minimum(jax.vmap(critics[0])(observation), jax.vmap(critics[1])(observation))
            

        def sample_data(runner_state, _) -> Tuple[Tuple, Transition]:
            train_state, env_state, last_obs, rng = runner_state
            rng, sample_key, step_key = jax.random.split(rng, 3)

            action_dist = jax.vmap(train_state.actor)(last_obs)
            action, log_prob = action_dist.sample_and_log_prob(seed=sample_key)
            value = get_critic_output(train_state.critic, last_obs)
            if hyperparams.use_Q_critic:
                value = (value * action_dist.probs).sum(axis=-1)

            # take a step in the environment
            rng, key = jax.random.split(rng)
            step_key = jax.random.split(key, hyperparams.num_envs)
            obsv, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(step_key, env_state, action, env_params)
            
            transition = Transition(
                observation=last_obs,
                action=action,
                reward=reward,
                done=done,
                value=value,
                log_prob=log_prob,
                info=info,
                next_observation=obsv
            )
            
            runner_state = (train_state, env_state, obsv, rng)
            return runner_state, transition
        
        def process_sampled_data(train_state, trajectory_batch, last_obs, rng):
            if hyperparams.use_Q_critic: # we need Q targets for the critic(s)
                trajectory_batch_flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), trajectory_batch)
                next_action_dist = jax.vmap(train_state.actor)(trajectory_batch_flat.next_observation)
                next_action_probs = next_action_dist.probs
                next_action_probs_log = jnp.log(next_action_probs + 1e-8)
                q_targets = get_critic_output(train_state.critic_target, trajectory_batch_flat.next_observation)
                flat_targets = (next_action_probs * (q_targets - train_state.alpha() * next_action_probs_log)).sum(axis=-1)
                flat_targets = trajectory_batch_flat.reward + ~trajectory_batch_flat.done * hyperparams.gamma * flat_targets
                targets = flat_targets.reshape(trajectory_batch.reward.shape)

            last_value = get_critic_output(train_state.critic, last_obs)
            if hyperparams.use_Q_critic:
                action_dist = jax.vmap(train_state.actor)(last_obs)
                last_value = (last_value * action_dist.probs).sum(axis=-1)
            if hyperparams.use_gae: # use GAE
                advantages, returns = calculate_gae_returns(trajectory_batch, last_value, hyperparams)
            else:
                advantages, returns = calculate_n_step_returns(trajectory_batch, last_value, hyperparams)
            
            train_batch = TrainBatch(
                **trajectory_batch,
                advantages=advantages,
                returns=returns,
                targets=targets
            )
            train_batch = jax.tree.map( # flatten over the envs
                lambda x: x.reshape((-1,) + x.shape[2:]), train_batch
            )
            buffer_state = jax.jit(agent.buffer.add, donate_argnums=(0,))(train_state.buffer_state, train_batch)
            train_state = eqx.tree_at(lambda x: x.buffer_state, train_state, buffer_state)

            def create_minibatches(rng, _):
                rng, shuffle_key, sample_key = jax.random.split(rng, 3)
                train_batch = agent.buffer.sample(train_state.buffer_state, sample_key).experience.first
                batch_size = train_batch.observation.shape[0]

                batch_idx = jax.random.permutation(shuffle_key, batch_size)
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, batch_idx, axis=0), train_batch
                )
                minibatches = jax.tree.map( # If minibatches=False, num_minibatches=1
                    lambda x: x.reshape((hyperparams.num_minibatches, -1) + x.shape[1:]), shuffled_batch
                )
                return rng, minibatches

            rng, minibatched_train_data = jax.lax.scan(
                create_minibatches, rng, None, hyperparams.update_epochs, unroll=True
            )
            train_batch = jax.tree.map( # flatten the epochs, so we have one single batch to scan over
                lambda x: x.reshape((-1,) + x.shape[2:]), minibatched_train_data
            )

            return train_state, train_batch, rng

        def update_over_data(update_state, train_data: TrainBatch):

            def _policy_evaluation():
                if hyperparams.use_Q_critic:
                    critic_grads = q_loss(train_state.critic, hyperparams, train_data)
                else: # value network
                    critic_grads = value_loss(train_state.critic, hyperparams, train_data)
                updates, critic_opt_state = agent.optimizer.update(critic_grads, train_state.critic_optimizer_state)
                critic = optax.apply_updates(train_state.critic, updates)
                return critic, critic_opt_state
            
            def _policy_improvement():
                critic_output = get_critic_output(train_state.critic, train_data.observation)
                actor_grads = policy_loss(train_state.actor, hyperparams, train_data, critic_output, old_actor, train_state.alpha)
                updates, actor_opt_state = agent.optimizer.update(actor_grads, train_state.actor_optimizer_state)
                actor = optax.apply_updates(train_state.actor, updates)
                return actor, actor_opt_state
            
            train_state, old_actor = update_state

            updated_critic, updated_critic_opt_state = _policy_evaluation()
            updated_actor, updated_actor_opt_state = _policy_improvement()

            # update target network
            if hyperparams.use_target_networks:
                critic_target = jax.tree.map(lambda x, y: y * 0.9 + x * 0.10, train_state.critic_target, train_state.critic)
            else: critic_target = None

            if hyperparams.learn_alpha:
                action_dist = jax.vmap(train_state.actor)(train_data.observation)
                alpha_grads = alpha_loss(train_state.alpha, hyperparams, action_dist)
                updates, updated_alpha_opt_state = agent.optimizer.update(alpha_grads, train_state.alpha_optimizer_state)
                updated_alpha = optax.apply_updates(train_state.alpha, updates)
            else: updated_alpha, updated_alpha_opt_state = train_state.alpha, train_state.alpha_optimizer_state

            train_state = TrainState(
                actor=updated_actor,
                critic=updated_critic,
                alpha=updated_alpha,
                actor_optimizer_state=updated_actor_opt_state,
                critic_optimizer_state=updated_critic_opt_state,
                alpha_optimizer_state=updated_alpha_opt_state,
                buffer_state=train_state.buffer_state,
                critic_target=critic_target
            )

            update_state = (train_state, old_actor)

            return update_state, 0.0 # TODO: return loss info

        @partial(jax.jit, donate_argnums=(0,))
        def train_step(runner_state, _):
            
            # sample
            (train_state, env_state, last_obs, rng), trajectory_batch = jax.lax.scan(
                sample_data, runner_state, None, hyperparams.num_steps
            )

            # process
            train_state, train_batch, rng = process_sampled_data(train_state, trajectory_batch, last_obs, rng)

            # train
            update_state = (train_state, train_state.actor)
            update_state, loss_info = jax.lax.scan(
                update_over_data, update_state, train_batch, unroll=4
            )

            runner_state = (update_state[0], env_state, last_obs, rng)

            if hyperparams.debug:
                metric = trajectory_batch.info
                metric["loss"] = loss_info
                metric["num_envs"] = agent.hyperparams.num_envs
                jax.debug.callback(log_callback, metric)

            return runner_state, trajectory_batch.info

        if agent is None:
            hyperparams = hyperparams or GPI_hyperparams()
            agent = GpiAlgorithm(rng, env, hyperparams)

        env_params = env.default_params
        hyperparams = agent.hyperparams
        batch_size = int(hyperparams.num_envs * hyperparams.num_steps)
        num_iterations = int(hyperparams.total_timesteps // batch_size)
        latest_or_new_train_state = agent.train_state_checkpoint

        # reset the environment
        rng, reset_key = jax.random.split(rng)
        reset_key = jax.random.split(reset_key, hyperparams.num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)

        runner_state = (latest_or_new_train_state, env_state, obsv, rng)
        runner_state, metrics = jax.lax.scan(
            train_step, runner_state, None, num_iterations
        )

        return agent, metrics


if __name__ == "__main__":
    env, env_params = gymnax.make("CartPole-v1")
    # env = navix.make('MiniGrid-Empty-8x8-v0')
    env = LogWrapper(env)
    key = jax.random.PRNGKey(0)
    agent = GpiAlgorithm(key, env)
    GpiAlgorithm.train(key, env, agent=agent)
