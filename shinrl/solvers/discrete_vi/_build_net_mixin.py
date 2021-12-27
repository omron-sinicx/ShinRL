""" MixIn to prepare networks.
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from copy import deepcopy
from typing import Optional

import gym
import jax
import jax.numpy as jnp
import optax

import shinrl as srl

from .config import ViConfig


class BuildNetMixIn:
    def initialize(self, env: gym.Env, config: Optional[ViConfig] = None) -> None:
        super().initialize(env, config)

        q_net, q_opt = self._build_net()
        self.q_net = q_net
        self.q_opt = q_opt

        net_param, opt_state = self._build_net_data()
        self.data["QNetParams"] = net_param
        self.data["QNetTargParams"] = deepcopy(net_param)
        self.data["QOptState"] = opt_state

    def _build_net(self):
        n_out = self.env.action_space.n
        depth, hidden = self.config.depth, self.config.hidden
        act_layer = getattr(jax.nn, self.config.activation.name)
        if len(self.env.observation_space.shape) == 1:
            net = srl.build_obs_forward_fc(n_out, depth, hidden, act_layer)
        else:
            net = srl.build_obs_forward_conv(n_out, depth, hidden, act_layer)
        optimizer = getattr(optax, self.config.optimizer.name)

        q_net = net
        q_opt = optimizer(learning_rate=self.config.lr)
        return q_net, q_opt

    def _build_net_data(self):
        self.key, key = jax.random.split(self.key)
        sample_init = jnp.ones([1, *self.env.observation_space.shape])
        net_param = self.q_net.init(key, sample_init)
        opt_state = self.q_opt.init(net_param)
        return net_param, opt_state
