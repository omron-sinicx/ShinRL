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
from chex import Array

import shinrl as srl

from .config import DdpgConfig


class BuildNetMixIn:
    def initialize(self, env: gym.Env, config: Optional[DdpgConfig] = None) -> None:
        super().initialize(env, config)

        q_net, q_opt, pol_net, pol_opt = self._build_net()
        self.q_net = q_net
        self.q_opt = q_opt
        self.pol_net = pol_net
        self.pol_opt = pol_opt

        q_param, q_state, pol_param, pol_state = self._build_net_data()
        self.data["QNetParams"] = q_param
        self.data["QNetTargParams"] = deepcopy(q_param)
        self.data["QOptState"] = q_state
        self.data["PolNetParams"] = pol_param
        self.data["PolOptState"] = pol_state
        self.data["PolNetTargParams"] = deepcopy(pol_param)

    def _build_net(self):
        depth, hidden = self.config.depth, self.config.hidden
        optimizer = getattr(optax, self.config.optimizer.name)
        act_layer = getattr(jax.nn, self.config.activation.name)

        # build q-net
        if len(self.env.observation_space.shape) == 1:
            builder = srl.build_obs_act_forward_fc
        else:
            builder = srl.build_obs_act_forward_conv
        q_net = builder(1, depth, hidden, act_layer)
        q_opt = optimizer(learning_rate=self.config.q_lr)

        # build pol-net
        if len(self.env.observation_space.shape) == 1:
            builder = srl.build_obs_forward_fc
        else:
            builder = srl.build_obs_forward_conv

        n_out = self.env.action_space.shape[0]
        pol_net = builder(n_out, depth, hidden, act_layer, last_layer=jax.nn.tanh)
        pol_opt = optimizer(learning_rate=self.config.pol_lr)
        return q_net, q_opt, pol_net, pol_opt

    def _build_net_data(self):
        sample_obs = jnp.ones([1, *self.env.observation_space.shape])
        sample_act = jnp.ones([1, *self.env.action_space.shape])

        self.key, key = jax.random.split(self.key)
        q_param = self.q_net.init(key, sample_obs, sample_act)
        q_state = self.q_opt.init(q_param)

        self.key, key = jax.random.split(self.key)
        pol_param = self.pol_net.init(key, sample_obs)
        pol_state = self.pol_opt.init(pol_param)

        return q_param, q_state, pol_param, pol_state
