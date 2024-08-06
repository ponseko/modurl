import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
from typing import Tuple
import chex
import gymnax
from functools import partial
from env_wrappers import LogWrapper
import flashbax
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState as BufferState, TrajectoryBuffer as Buffer
from gpi_modules import *
from util import log_callback, set_up_replay_buffer, set_up_optimizer, Transition, TrainBatch
import navix
from networks import ActorNetwork, ValueNetwork, Q_Network, Alpha
from __future__ import annotations



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
    
    def get_critic_output(self, critics: tuple, observation):
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

    def sample_data(self, runner_state):

        def _sample_step(runner_state, _) -> Tuple[Tuple, Transition]:
            train_state, env_state, last_obs, rng = runner_state
            rng, sample_key, step_key = jax.random.split(rng, 3)

            action_dist = jax.vmap(train_state.actor)(last_obs)
            action, log_prob = action_dist.sample_and_log_prob(seed=sample_key)
            value = self.get_critic_output(train_state.critic, last_obs)
            if self.hyperparams.use_Q_critic:
                value = (value * action_dist.probs).sum(axis=-1)

            # take a step in the environment
            rng, key = jax.random.split(rng)
            step_key = jax.random.split(key, self.hyperparams.num_envs)
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
        
        """" First we do a rollout of `num_steps` steps in the environment """
        (train_state, env_state, last_obs, rng), trajectory_batch = jax.lax.scan(
            _sample_step, runner_state, None, self.hyperparams.num_steps
        )

        """ 
            Then we calculate the training targets (advantages, returns, q_targets) 
            This introduces some reduncancy as not all targets are used in all cases
            For the time being we accept this small inefficiency for the sake of a simpler generalization 
        """
        if self.hyperparams.use_Q_critic: # we need Q targets for the critic(s)
            trajectory_batch_flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), trajectory_batch)
            next_action_dist = jax.vmap(train_state.actor)(trajectory_batch_flat.next_observation)
            next_action_probs = next_action_dist.probs
            next_action_probs_log = jnp.log(next_action_probs + 1e-8)
            q_targets = self.get_critic_output(train_state.critic_target, trajectory_batch_flat.next_observation)
            flat_targets = (next_action_probs * (q_targets - train_state.alpha() * next_action_probs_log)).sum(axis=-1)
            flat_targets = trajectory_batch_flat.reward + ~trajectory_batch_flat.done * self.hyperparams.gamma * flat_targets
            targets = flat_targets.reshape(trajectory_batch.reward.shape)

        last_value = self.get_critic_output(train_state.critic, last_obs)
        if self.hyperparams.use_Q_critic:
            action_dist = jax.vmap(train_state.actor)(last_obs)
            last_value = (last_value * action_dist.probs).sum(axis=-1)
        if self.hyperparams.use_gae: # use GAE
            advantages, returns = calculate_gae_returns(trajectory_batch, last_value, self.hyperparams)
        else:
            advantages, returns = calculate_n_step_returns(trajectory_batch, last_value, self.hyperparams)
        
        train_batch = TrainBatch(
            **trajectory_batch,
            advantages=advantages,
            returns=returns,
            targets=targets
        )
        train_batch = jax.tree.map( # flatten over the envs
            lambda x: x.reshape((-1,) + x.shape[2:]), train_batch
        )
        buffer_state = jax.jit(self.buffer.add, donate_argnums=(0,))(train_state.buffer_state, train_batch)
        train_state = eqx.tree_at(lambda x: x.buffer_state, train_state, buffer_state)

        runner_state = (train_state, env_state, last_obs, rng)
        return runner_state, train_batch
    
    def preprocess_sampled_data(self, runner_state):
        batch_size = int(self.hyperparams.num_envs * self.hyperparams.num_steps)
        train_state, env_state, last_obs, rng = runner_state
        rng, shuffle_key, sample_key = jax.random.split(rng, 3)
        train_batch = self.buffer.sample(train_state.buffer_state, sample_key).experience.first

        batch_idx = jax.random.permutation(shuffle_key, batch_size)
        shuffled_batch = jax.tree_util.tree_map(
            lambda x: jnp.take(x, batch_idx, axis=0), train_batch
        )
        minibatches = jax.tree_util.tree_map( # If minibatches=False, num_minibatches=1
            lambda x: x.reshape((self.hyperparams.num_minibatches, -1) + x.shape[1:]), shuffled_batch
        )
        # repeat the data for each update epoch #TODO: this is not consistent with actually doing multiple epochs, as in a normal sense we get different minibatches each time.
        train_data = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x, self.hyperparams.update_epochs, axis=0), minibatches
        )

        return train_data
    
    def update_over_data(self, runner_state, train_batch: TrainBatch):

        def _update(runner_state, train_data: TrainBatch):

            def __policy_evaluation():
                if self.hyperparams.use_Q_critic:
                    critic_grads = q_loss(train_state.critic, self.hyperparams, train_data)
                else: # value network
                    critic_grads = value_loss(train_state.critic, self.hyperparams, train_data)
                updates, critic_opt_state = self.optimizer.update(critic_grads, train_state.critic_optimizer_state)
                critic = optax.apply_updates(train_state.critic, updates)
                return critic, critic_opt_state
            
            def __policy_improvement():
                critic_output = self.get_critic_output(train_state.critic, train_data.observation)
                actor_grads = policy_loss(train_state.actor, self.hyperparams, train_data, critic_output, train_state_before_update.actor, train_state.alpha)
                updates, actor_opt_state = self.optimizer.update(actor_grads, train_state.actor_optimizer_state)
                actor = optax.apply_updates(train_state.actor, updates)
                return actor, actor_opt_state
            
            train_state, env_state, last_obs, rng = runner_state

            # update target network
            if self.hyperparams.use_target_networks:
                critic_target = jax.tree.map(lambda x, y: y * 0.995 + x * 0.005, train_state.critic_target, train_state.critic)
            else: critic_target = None

            if self.hyperparams.learn_alpha:
                action_dist = jax.vmap(train_state.actor)(train_data.observation)
                alpha_grads = alpha_loss(train_state.alpha, self.hyperparams, action_dist)
                updates, updated_alpha_opt_state = self.optimizer.update(alpha_grads, train_state.alpha_optimizer_state)
                updated_alpha = optax.apply_updates(train_state.alpha, updates)
            else: updated_alpha, updated_alpha_opt_state = train_state.alpha, train_state.alpha_optimizer_state

            updated_critic, updated_critic_opt_state = __policy_evaluation()
            updated_actor, updated_actor_opt_state = __policy_improvement()
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

            runner_state = (train_state, env_state, last_obs, rng)

            return runner_state, 0.0 # TODO: return loss info
        
        train_state_before_update, env_state, last_obs, rng = runner_state

        runner_state, loss_info = jax.lax.scan(_update, runner_state, train_batch, unroll=4)

        return runner_state, loss_info

def train(rng: chex.PRNGKey, agent: GpiAlgorithm, env: gymnax.environments.environment.Environment):

    @partial(jax.jit, donate_argnums=(0,))
    def train_step(runner_state, _):

        runner_state, trajectory_batch = agent.sample_data(runner_state)
        # train_batch = agent.get_critic_targets(runner_state, trajectory_batch)
        train_batch = agent.preprocess_sampled_data(runner_state)
        runner_state, loss_info = agent.update_over_data(runner_state, train_batch)

        if agent.hyperparams.debug:
            metric = trajectory_batch.info
            metric["loss"] = loss_info
            metric["num_envs"] = agent.hyperparams.num_envs
            jax.debug.callback(log_callback, metric)

        return runner_state, trajectory_batch.info

    env_params = env.default_params
    rng, reset_key = jax.random.split(rng)
    reset_key = jax.random.split(reset_key, agent.hyperparams.num_envs)
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)

    batch_size = int(agent.hyperparams.num_envs * agent.hyperparams.num_steps)
    num_iterations = int(agent.hyperparams.total_timesteps // batch_size)
    # minibatch_size = int(batch_size // agent.hyperparams.num_minibatches)

    runner_state = (agent.train_state_checkpoint, env_state, obsv, rng)
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
    train(key, agent, env)
