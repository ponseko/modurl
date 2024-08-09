from typing import Optional, Tuple, Union, NamedTuple
import chex
from gymnax.environments import spaces
import numpy as np
import equinox as eqx

@chex.dataclass(frozen=True)
class EnvState:
    pass

@chex.dataclass(frozen=True)
class LogEnvState:
    env_state: EnvState
    episode_returns: float
    returned_episode_returns: float
    timestep: int

class TimeStep(NamedTuple):
    observation: chex.Array
    reward: chex.Array
    terminated: chex.Array
    truncated: chex.Array
    info: chex.Array

class JaxEnvWrapper(object):
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)
    
class GymnaxWrapper(JaxEnvWrapper):
    """ 
        Converts a Gymnax(-like) environment to a standard JAX environment,
        by converting the step and reset output format
    """

    def reset(self, key, params=None):
        obs, env_state = self._env.reset(key, params)
        return obs, env_state # no changes ...
    
    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state, action, params
        )
        timestep = TimeStep(
            observation=obs,
            reward=reward,
            terminated=done,
            truncated=False, # gymnax API does not distinguish between truncated and terminated
            info=info
        )
        return timestep, env_state
    
class NavixWrapper(JaxEnvWrapper):
    """ 
        Converts a Navix(-like) environment to a standard JAX environment,
        by converting the step and reset output format
        also flattens the observation space
    """

    def reset(self, key):
        timestep = self._env.reset(key)
        flat_obs = timestep.observation.flatten()
        return flat_obs, timestep

    def step(self, key, state, action):
        navix_timestep = self._env.step(state, action)
        flat_obs = navix_timestep.observation.flatten()
        timestep = (flat_obs, navix_timestep.reward, navix_timestep.is_termination(), navix_timestep.is_truncation(), navix_timestep.info)
        return timestep, navix_timestep # navix requires its entire timestep instead of just the state
    
    @property
    def observation_space(self):
        return spaces.Box(
            low=self._env.observation_space.minimum,
            high=self._env.observation_space.maximum,
            shape=(np.prod(self._env.observation_space.shape),),
            dtype=self._env.observation_space.dtype,
        )

class LogWrapper(JaxEnvWrapper):
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, LogEnvState]:
        obs, env_state = self._env.reset(key)
        
        if hasattr(env_state, "info"):
            # navix fix:
            info = {
                **env_state.info,
                "returned_episode_returns": 0.0,
                "timestep": 0,
                "returned_episode": False,
            }
            env_state = eqx.tree_at(lambda x: x.info, env_state, info)
        state = LogEnvState(
            env_state=env_state,
            episode_returns=0.0,
            returned_episode_returns=0.0,
            timestep=0,
        )
        
        return obs, state

    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float, chex.Array],
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        (obs, reward, terminated, truncated, info), env_state = self._env.step(
            key, state.env_state, action
        )
        done = terminated | truncated
        new_episode_return = state.episode_returns + reward
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode"] = done
        info["timestep"] = state.timestep
        return TimeStep(obs, reward, terminated, truncated, info), state
