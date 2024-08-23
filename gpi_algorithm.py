from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
from typing import Tuple, Optional
from functools import partial
import chex
import gymnax
from gymnax.environments.environment import Environment
from env_wrappers import LogWrapper, GymnaxWrapper, NavixWrapper
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState as BufferState, TrajectoryBuffer as Buffer
from gpi_modules import *
from util import log_callback, wandb_callback, set_up_replay_buffer, set_up_optimizer, Transition, TrainBatch
from networks import ActorNetwork, ValueNetwork, Q_Network, Alpha
# import navix
import wandb
import pandas as pd
from jax_tqdm import scan_tqdm
import distrax

@chex.dataclass(frozen=True)
class TrainState:
    actor: Optional[ActorNetwork]
    actor_optimizer_state:  Optional[optax.OptState]
    buffer_state: BufferState
    alpha: Alpha
    alpha_optimizer_state: optax.OptState

    critic: Tuple[eqx.Module] | Tuple[eqx.Module, eqx.Module]
    critic_optimizer_state: Tuple[optax.OptState] | Tuple[optax.OptState, optax.OptState]

    critic_target: Tuple[eqx.Module] | Tuple[eqx.Module, eqx.Module] | None = None

@chex.dataclass(frozen=True)
class GpiHyperparams:
    """
    Hyperparameters for the any algorithm
    There will be some reduncy in this class, but it will be easier to manage
    """
    # General
    total_timesteps: int = 1e6
    learning_rate_init: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    num_envs: int = 4
    debug: bool = True # log values during training
    ent_coef: float = 0.01
    alpha_init: float = 0.0 # 
    entropy_target: float = 0.5
    learn_alpha: bool = True 
    kl_coef: float = 0.2
    max_grad_norm: float = 0.5
    tau: float = 0.05 # target network update rate

    # PPO
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_coef_vf: float = 10.0 # Depends on the reward scaling !
    vf_coef: float = 0.25
    num_steps: int = 50 # steps per environment
    num_minibatches: int = 1 # Number of mini-batches
    update_epochs: int = 1 # K epochs to update the policy
    # batch_size: int = 0 # batch size (num_envs * num_steps)
    # minibatch_size: int = 0 # mini-batch size (batch_size / num_minibatches)
    # num_iterations: int = 0 # number of iterations (total_timesteps / num_steps / num_envs)

    # Buffer: Only used when off_policy=True
    buffer_max_size: int = 2500
    buffer_sample_size: int = 64

    # Configuration options
    dual_critics: bool = False
    use_Q_critic: bool = True # if false: use value network, else Q network
    use_target_networks: bool = True
    off_policy: bool = True # else -> sample_batch_size = num_steps (* num_envs)
    normalize_advantages: bool = True
    policy_regularizer: Tuple[str] = ("sac",) # "ppo", "a2c", "sac", "kl", "add_entropy"
    q_target_objective: str = "QLEARNING" # "QLEARNING", "SARSA", "ESARSA"
    exploration_method: str = "sampling" # "sampling", "egreedy"
    epsilon_init: float = 0.3
    use_policy_network: bool = True
    orthogonal_init: bool = True

    @property
    def num_rollouts(self):
        rollout_length = int(self.num_envs * self.num_steps)
        return int(self.total_timesteps // rollout_length)

    @property
    def num_training_iterations(self):
        updates_per_rollout = int(self.num_minibatches * self.update_epochs)
        return int(self.num_rollouts * updates_per_rollout)

    @property
    def learning_rate(self):
        if not self.anneal_lr:
            return self.learning_rate_init
        return optax.linear_schedule(
            init_value=self.learning_rate_init,
            end_value=0.,
            transition_steps=self.num_training_iterations
        )
    
    @property
    def epsilon(self):
        return optax.linear_schedule(
            init_value=self.epsilon_init,
            end_value=0.01,
            transition_steps=self.num_training_iterations
        )
    

class GpiAlgorithm(eqx.Module):

    train_state_checkpoint: TrainState
    hyperparams: GpiHyperparams = GpiHyperparams() # = eqx.field(static=True)
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    buffer: Buffer = eqx.field(static=True)

    def __init__(
        self,
        key: chex.PRNGKey,
        env: gymnax.environments.environment.Environment, # env is a gym environment in jax style TODO
        hyperparams: GpiHyperparams = GpiHyperparams(),
    ):
        self.hyperparams = hyperparams

        # Assume environments implement spaces as a property
        observation_space = env.observation_space
        action_space = env.action_space
        num_actions = action_space.n

        actor_key, critic_key, dual_critic_key = jax.random.split(key, 3)
        if hyperparams.use_policy_network:
            actor = ActorNetwork(actor_key, observation_space.shape[0], [64, 64], hyperparams.orthogonal_init, num_actions)
        else: actor = None

        if hyperparams.use_Q_critic: 
            CriticNetwork = Q_Network
        else: CriticNetwork = ValueNetwork

        alpha = Alpha(hyperparams.alpha_init, num_actions)

        if hyperparams.dual_critics:
            critic = (
                CriticNetwork(critic_key, observation_space.shape[0], [64, 64], hyperparams.orthogonal_init, num_actions=num_actions),
                CriticNetwork(dual_critic_key, observation_space.shape[0], [64, 64], hyperparams.orthogonal_init, num_actions=num_actions)
            )
        else:
            critic = (CriticNetwork(critic_key, observation_space.shape[0], [64, 64], hyperparams.orthogonal_init, num_actions=num_actions),)

        if hyperparams.use_target_networks:
            critic_target = jax.tree.map(lambda x: x, critic)
        else: critic_target = None

        self.optimizer, actor_optimizer_state, critic_optimizer_state, alpha_optimizer_state = set_up_optimizer(hyperparams, actor, critic, alpha)
        self.buffer, buffer_state = set_up_replay_buffer(hyperparams, env)
        
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
        # NOTE: always uses sampling for now
        logits = self.train_state_checkpoint.actor(observation)
        action_dist = distrax.Categorical(logits=logits)
        action = action_dist.sample(seed=key)
        return action
    
    def save(self, path: str) -> None:
        eqx.tree_serialise_leaves(path, self)

    def load(self, path: str) -> GpiAlgorithm:
        return eqx.tree_deserialise_leaves(path, self)
    

    @classmethod
    def train(cls, rng: chex.PRNGKey, env: Environment, hyperparams: Optional[GpiHyperparams] = None, agent: Optional[GpiAlgorithm] = None):
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

        def get_curr_train_iter(train_state: TrainState):
            return train_state.critic_optimizer_state[1][0].count


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
            
        def get_action_dist(observation, train_state: Optional[TrainState] = None, *, actor = None, critic = None) -> distrax.Distribution:
            if train_state is not None:
                actor = train_state.actor if actor is None else actor
                critic = train_state.critic if critic is None else critic
            if actor is not None: # We use a policy network
                logits = jax.vmap(actor)(observation)
            else: # We use a Q network
                assert critic is not None and hyperparams.use_Q_critic, "If not using policy network, must use Q network"
                logits = get_critic_output(critic, observation)
            if hyperparams.exploration_method == "sampling":
                return distrax.Categorical(logits=logits)
            elif hyperparams.exploration_method == "egreedy":
                return distrax.EpsilonGreedy(logits, epsilon=hyperparams.epsilon(get_curr_train_iter(train_state)))
            else:
                raise ValueError(f"Invalid exploration method: {hyperparams.exploration_method}")
            
        def sample_data(runner_state, _) -> Tuple[Tuple, Transition]:
            train_state, env_state, last_obs, rng = runner_state
            rng, sample_key, step_key = jax.random.split(rng, 3)

            action_dist = get_action_dist(last_obs, train_state)
            action, log_prob = action_dist.sample_and_log_prob(seed=sample_key)
            value = get_critic_output(train_state.critic, last_obs)
            if hyperparams.use_Q_critic:
                value = (value * action_dist.probs).sum(axis=-1)

            # take a step in the environment
            rng, key = jax.random.split(rng)
            step_key = jax.random.split(key, hyperparams.num_envs)
            (obsv, reward, terminated, truncated, info), env_state = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(step_key, env_state, action)

            done = terminated | truncated

            # # SB3 hack: (would like something else, but lets leave it for now)
            next_value = get_critic_output(train_state.critic, obsv)
            next_value = hyperparams.gamma * next_value
            reward = reward + (truncated * next_value)
            
            transition = Transition(
                observation=last_obs,
                action=action,
                reward=reward,
                done=done,
                value=value,
                log_prob=log_prob,
                info=info,
                next_observation=info["terminal_observation"]
            )
            
            runner_state = (train_state, env_state, obsv, rng)
            return runner_state, transition
        
        def process_sampled_data(train_state, trajectory_batch, last_obs, rng):
            if hyperparams.use_Q_critic: # we need Q targets for the critic(s)
                trajectory_batch_flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), trajectory_batch)
                next_action_dist = get_action_dist(trajectory_batch_flat.next_observation, train_state)
                next_action_probs = next_action_dist.probs
                next_action_probs_log = jnp.log(next_action_probs + 1e-8)
                if hyperparams.use_target_networks:
                    q_targets = get_critic_output(train_state.critic_target, trajectory_batch_flat.next_observation)
                else:
                    q_targets = get_critic_output(train_state.critic, trajectory_batch_flat.next_observation)
                # flat_targets = (next_action_probs * (q_targets - train_state.alpha() * next_action_probs_log)).sum(axis=-1)


                if True: # soft targets TODO option in config, or just leave as alpha=0
                    q_targets = q_targets - train_state.alpha() * next_action_probs_log
                if hyperparams.q_target_objective == "QLEARNING":
                    flat_targets = jnp.max(q_targets, axis=-1)
                elif hyperparams.q_target_objective == "SARSA":
                    idx1 = np.arange(q_targets.shape[0])
                    flat_targets = q_targets[idx1, trajectory_batch.action]
                elif hyperparams.q_target_objective == "ESARSA":
                    flat_targets = (next_action_probs * q_targets).sum(axis=-1)
                # TODO: sampled ESARSA for cont action spaces
                flat_targets = trajectory_batch_flat.reward + ~trajectory_batch_flat.done * hyperparams.gamma * flat_targets
                targets = flat_targets.reshape(trajectory_batch.reward.shape)
            else: 
                targets = np.zeros(trajectory_batch.reward.shape)

            last_value = get_critic_output(train_state.critic, last_obs)
            if hyperparams.use_Q_critic:
                action_dist = get_action_dist(last_obs, train_state)
                last_value = (last_value * action_dist.probs).sum(axis=-1)
            advantages, returns = calculate_gae_returns(trajectory_batch, last_value, hyperparams)
            
            train_batch = TrainBatch(
                **trajectory_batch,
                advantages=advantages,
                returns=returns,
                targets=targets 
            )

            train_batch = jax.tree.map( # align with flashbax: (envs, time_length, ...)
                lambda x: x.swapaxes(0,1), train_batch
            )
            buffer_state = jax.jit(agent.buffer.add, donate_argnums=(0,))(train_state.buffer_state, train_batch)

            train_state = train_state.replace(
                buffer_state=buffer_state
            )
            # train_state = eqx.tree_at(lambda x: x.buffer_state, train_state, buffer_state)

            def create_minibatches(rng, _):
                rng, shuffle_key, sample_key = jax.random.split(rng, 3)
                train_batch = agent.buffer.sample(train_state.buffer_state, sample_key).experience#.first
                # indices = jax.random.permutation(shuffle_key, train_batch.reward.shape[0])
                # train_batch = jax.tree_util.tree_map(
                #     lambda x: jnp.take(x, indices, axis=0), train_batch
                # )
                minibatches = jax.tree.map( # If minibatches=False, num_minibatches=1
                    lambda x: x.reshape((hyperparams.num_minibatches, -1) + x.shape[1:]), train_batch
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
                if not hyperparams.use_policy_network:
                    return train_state.actor, train_state.actor_optimizer_state
                critic_output = get_critic_output(train_state.critic, train_data.observation)
                actor_grads = policy_loss(train_state.actor, hyperparams, train_data, critic_output, old_actor, train_state.alpha, get_action_dist)
                updates, actor_opt_state = agent.optimizer.update(actor_grads, train_state.actor_optimizer_state)
                actor = optax.apply_updates(train_state.actor, updates)
                return actor, actor_opt_state
            
            train_state, old_actor = update_state 

            updated_critic, updated_critic_opt_state = _policy_evaluation()
            updated_actor, updated_actor_opt_state = _policy_improvement()

            # update target network
            if hyperparams.use_target_networks:
                critic_target = jax.tree.map(lambda x, y: y * (1 - hyperparams.tau) + x * hyperparams.tau, train_state.critic_target, train_state.critic)
            else: critic_target = None

            # update alpha
            if hyperparams.learn_alpha:
                # NOTE: updated train state?
                action_dist = get_action_dist(train_data.observation, train_state)
                alpha_grads = alpha_loss(train_state.alpha, hyperparams, action_dist, step=train_state.alpha_optimizer_state[1][0].count)
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

        @eqx.filter_jit(donate="warn")
        def train_step(runner_state, _):
            # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
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

            metric = trajectory_batch.info
            metric["loss"] = loss_info
            metric["num_envs"] = agent.hyperparams.num_envs

            if hyperparams.debug:
                jax.debug.callback(log_callback, metric)

            if wandb.run:
                jax.debug.callback(wandb_callback, metric)

            return runner_state, metric

        if agent is None:
            hyperparams = hyperparams or GpiHyperparams()
            agent = GpiAlgorithm(rng, env, hyperparams)

        hyperparams = agent.hyperparams
        batch_size = int(hyperparams.num_envs * hyperparams.num_steps)
        num_iterations = int(hyperparams.total_timesteps // batch_size)
        latest_or_new_train_state = agent.train_state_checkpoint

        # reset the environment
        rng, reset_key = jax.random.split(rng)
        reset_key = jax.random.split(reset_key, hyperparams.num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_key)

        runner_state = (latest_or_new_train_state, env_state, obsv, rng)
        if not hyperparams.debug:
            train_step = scan_tqdm(num_iterations)(train_step)
        runner_state, metrics = jax.lax.scan(
            train_step, runner_state, jnp.arange(num_iterations)
        )

        return agent, metrics


if __name__ == "__main__":
    env, env_params = gymnax.make("CartPole-v1")
    env = GymnaxWrapper(env)
    env = LogWrapper(env)
    key = jax.random.PRNGKey(0)
    from default_params import DEFAULT_PPO_PARAMS, DEFAULT_A2C_PARAMS
    agent, metrics = GpiAlgorithm.train(key, env, DEFAULT_PPO_PARAMS)
    # agent, metrics = GpiAlgorithm.train(key, env)
    avg = metrics["returned_episode_returns"][metrics["returned_episode"]].mean()
    print(avg)
