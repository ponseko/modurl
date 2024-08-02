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

from networks import ActorNetwork, ValueNetwork, Q_Network


@chex.dataclass(frozen=True)
class TrainState:
    actor: eqx.Module
    actor_optimizer_state: optax.OptState
    buffer_state: BufferState # NOTE: Maybe also optional (None?)

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

    # PPO
    gae_lambda: float = 0.95
    max_grad_norm: float = 100.0
    clip_coef: float = 0.2
    clip_coef_vf: float = 10.0 # Depends on the reward scaling !
    ent_coef: float = 0.01
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
    policy_regularizer: str = "sac" # "ppo", "sac", "a2c"
    dual_critics: bool = True
    use_Q_critic: bool = True # if false: use value network, else Q network
    use_gae: bool = False
    use_target_networks: bool = True
    off_policy: bool = True

class GpiAlgorithm(eqx.Module):

    train_state: TrainState
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

        self.optimizer, actor_optimizer_state, critic_optimizer_state = set_up_optimizer(hyperparams, actor, critic)
        self.buffer, buffer_state = set_up_replay_buffer(hyperparams, env, env_params)
        
        # This train state is not updated during training, only at checkpoints such that it can be saved
        self.train_state = TrainState(
            actor=actor,
            critic=critic,
            actor_optimizer_state=actor_optimizer_state,
            critic_optimizer_state=critic_optimizer_state,
            critic_target=critic_target,
            buffer_state=buffer_state
        )

    def __call__(self, key: chex.PRNGKey, observation):
        action_dist = jax.vmap(self.train_state.actor)(observation)
        action = action_dist.sample(seed=key)
        return action
    
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


    
    def save(self, path: str):
        eqx.tree_serialise_leaves(path, self)

    def load(self, path: str):
        return eqx.tree_deserialise_leaves(path)

    def sample_data(self, runner_state):

        def _sample_step(runner_state, _):
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
                info=info
            )
            buffer_state = jax.jit(self.buffer.add, donate_argnums=(0,))(train_state.buffer_state, transition)
            train_state = eqx.tree_at(lambda x: x.buffer_state, train_state, buffer_state)
            runner_state = (train_state, env_state, obsv, rng)
            return runner_state, transition
        
        runner_state, trajectory_batch = jax.lax.scan(
            _sample_step, runner_state, None, self.hyperparams.num_steps
        )

        return runner_state, trajectory_batch
    
    def get_critic_targets(self, runner_state, trajectory_batch) -> TrainBatch:
        train_state, env_state, last_obs, rng = runner_state
        rng, shuffle_key, sample_key = jax.random.split(rng, 3)

        if self.hyperparams.off_policy:
            trajectory_batch = self.buffer.sample(train_state.buffer_state, sample_key).experience
            train_batch = TrainBatch(trajectory_batch.first)
            train_batch = eqx.tree_at( # inserting the next observation in the sampled batch
                lambda x: x.next_observations, train_batch, trajectory_batch.second.observation, is_leaf=lambda x: x is None
            )

        if self.hyperparams.use_Q_critic: # we need Q targets for the critic(s)
            assert train_batch.next_observations is not None, "Next observation is required for Q learning (only data from the buffer supported for now)"
            next_action_dist = jax.vmap(train_state.actor)(train_batch.next_observations)
            next_action_probs = next_action_dist.probs
            next_action_probs_log = jnp.log(next_action_probs + 1e-8)
            q_targets = self.get_critic_output(train_state.critic_target, train_batch.next_observations)
            targets = (next_action_probs * (q_targets - 0.2 * next_action_probs_log)).sum(axis=-1)
            targets = train_batch.reward + ~train_batch.done * self.hyperparams.gamma * targets
            train_batch = eqx.tree_at(
                lambda x: x.targets, train_batch, targets, is_leaf=lambda x: x is None
            )

        else: # we need V targets (returns and advantages)
            last_value = self.get_critic_output(train_state.critic, last_obs)
            if self.hyperparams.use_gae: # use GAE
                advantages, returns = calculate_gae(trajectory_batch, last_value, self.hyperparams)
                train_batch = TrainBatch(
                    **trajectory_batch,
                    advantages=advantages,
                    returns=returns
                )
            else: # use n-step returns, or nothing
                train_batch = TrainBatch(trajectory_batch=trajectory_batch)

        return train_batch
    
    def preprocess_sampled_data(self, runner_state, train_batch: TrainBatch):
        batch_size = int(self.hyperparams.num_envs * self.hyperparams.num_steps)
        train_state, env_state, last_obs, rng = runner_state
        rng, shuffle_key, sample_key = jax.random.split(rng, 3)

        batch_idx = jax.random.permutation(shuffle_key, batch_size)
        # reshape (flatten the data gathered by different envs)
        if not self.hyperparams.off_policy: # NOTE: this should be different
            train_batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), train_batch
            )
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
                if self.hyperparams.policy_regularizer == "ppo":
                    actor_grads = policy_loss_ppo(train_state.actor, self.hyperparams, train_data, train_data.advantages)
                elif self.hyperparams.policy_regularizer == "a2c":
                    actor_grads = policy_loss_a2c(train_state.actor, self.hyperparams, train_data, train_data.advantages)
                elif self.hyperparams.policy_regularizer == "sac":
                    critic_output = self.get_critic_output(train_state.critic, train_data.observation)
                    actor_grads = policy_loss_sac(train_state.actor, self.hyperparams, train_data, critic_output)
                updates, actor_opt_state = self.optimizer.update(actor_grads, train_state.actor_optimizer_state)
                actor = optax.apply_updates(train_state.actor, updates)
                return actor, actor_opt_state
            
            train_state, env_state, last_obs, rng = runner_state

            # update target network
            if self.hyperparams.use_target_networks:
                critic_target = jax.tree.map(lambda x, y: y * 0.995 + x * 0.005, train_state.critic_target, train_state.critic)
            else: critic_target = None

            updated_critic, updated_critic_opt_state = __policy_evaluation()
            updated_actor, updated_actor_opt_state = __policy_improvement()
            train_state = TrainState(
                actor=updated_actor,
                critic=updated_critic,
                actor_optimizer_state=updated_actor_opt_state,
                critic_optimizer_state=updated_critic_opt_state,
                buffer_state=train_state.buffer_state,
                critic_target=critic_target
            )

            runner_state = (train_state, env_state, last_obs, rng)

            return runner_state, 0.0 # TODO: return loss info

        runner_state, loss_info = jax.lax.scan(_update, runner_state, train_batch, unroll=4)

        return runner_state, loss_info

def train(rng: chex.PRNGKey, agent: GpiAlgorithm, env: gymnax.environments.environment.Environment):

    @partial(jax.jit, donate_argnums=(0,))
    def train_step(runner_state, _):

        runner_state, trajectory_batch = agent.sample_data(runner_state)
        train_batch = agent.get_critic_targets(runner_state, trajectory_batch)
        train_batch = agent.preprocess_sampled_data(runner_state, train_batch)
        runner_state, loss_info = agent.update_over_data(runner_state, train_batch)

        if agent.hyperparams.debug:
            metric = trajectory_batch.info
            metric["loss"] = loss_info
            jax.debug.callback(log_callback, metric)

        return runner_state, trajectory_batch.info

    env_params = env.default_params
    rng, reset_key = jax.random.split(rng)
    reset_key = jax.random.split(reset_key, agent.hyperparams.num_envs)
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)

    batch_size = int(agent.hyperparams.num_envs * agent.hyperparams.num_steps)
    num_iterations = int(agent.hyperparams.total_timesteps // batch_size)
    # minibatch_size = int(batch_size // agent.hyperparams.num_minibatches)

    runner_state = (agent.train_state, env_state, obsv, rng)
    runner_state, metrics = jax.lax.scan(
        train_step, runner_state, None, num_iterations
    )

    return agent, metrics


if __name__ == "__main__":
    env, env_params = gymnax.make("CartPole-v1")
    env = LogWrapper(env)
    key = jax.random.PRNGKey(0)
    agent = GpiAlgorithm(key, env)
    train(key, agent, env)