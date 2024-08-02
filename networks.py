import jax
import equinox as eqx
from typing import List
import distrax
import jax.numpy as jnp

class ActorNetwork(eqx.Module):
    """Actor network"""

    layers: list

    def __init__(self, key, in_shape, hidden_features: List[int], num_actions):
        keys = jax.random.split(key, len(hidden_features))
        self.layers = [
            eqx.nn.Linear(in_shape, hidden_features[0], key=keys[0])
        ] + [
            eqx.nn.Linear(hidden_features[i], hidden_features[i+1], key=keys[i+1])
            for i in range(len(hidden_features)-1)
        ] + [
            eqx.nn.Linear(hidden_features[-1], num_actions, key=keys[-1])
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        return distrax.Categorical(logits=self.layers[-1](x))
    
class ValueNetwork(eqx.Module):
    """
        Value (critic) network with a single output
        Used to output V when given a state
    """
    layers: list

    def __init__(self, key, in_shape, hidden_layers: List[int], **kwargs):
        keys = jax.random.split(key, len(hidden_layers))
        self.layers = [
            eqx.nn.Linear(in_shape, hidden_layers[0], key=keys[0])
        ] + [
            eqx.nn.Linear(hidden_layers[i], hidden_layers[i+1], key=keys[i+1])
            for i in range(len(hidden_layers)-1)
        ] + [
            eqx.nn.Linear(hidden_layers[-1], 1, key=keys[-1])
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        return jnp.squeeze(self.layers[-1](x), axis=-1)

class Q_Network(eqx.Module):
    """'
        Q (critic) network that outputs values for each action
        e.g. a list of Q-values
    """

    layers: list

    def __init__(self, key, in_shape, hidden_layers: List[int], num_actions):
        keys = jax.random.split(key, len(hidden_layers))
        self.layers = [
            eqx.nn.Linear(in_shape, hidden_layers[0], key=keys[0])
        ] + [
            eqx.nn.Linear(hidden_layers[i], hidden_layers[i+1], key=keys[i+1])
            for i in range(len(hidden_layers)-1)
        ] + [
            eqx.nn.Linear(hidden_layers[-1], num_actions, key=keys[-1])
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        return self.layers[-1](x)