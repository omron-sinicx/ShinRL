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

from .config import PiConfig


class BuildNetMixIn:
    def initialize(self, env: gym.Env, config: Optional[PiConfig] = None) -> None:
        super().initialize(env, config)

        q_net, q_opt, log_pol_net, log_pol_opt = self._build_net()
        self.q_net = q_net
        self.q_opt = q_opt
        self.log_pol_net = log_pol_net
        self.log_pol_opt = log_pol_opt

        q_param, q_state, log_pol_param, log_pol_state = self._build_net_data()
        self.data["QNetParams"] = q_param
        self.data["QNetTargParams"] = deepcopy(q_param)
        self.data["QOptState"] = q_state
        self.data["LogPolNetParams"] = log_pol_param
        self.data["LogPolOptState"] = log_pol_state

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
        q_opt = optimizer(learning_rate=self.config.q_lr)
        log_pol_net = net
        log_pol_opt = optimizer(learning_rate=self.config.pol_lr)
        return q_net, q_opt, log_pol_net, log_pol_opt

    def _build_net_data(self):
        self.key, key = jax.random.split(self.key)
        sample_init = jnp.ones([1, *self.env.observation_space.shape])

        q_param = self.q_net.init(key, sample_init)
        q_state = self.q_opt.init(q_param)

        self.key, key = jax.random.split(self.key)
        log_pol_param = self.log_pol_net.init(key, sample_init)
        log_pol_state = self.log_pol_opt.init(log_pol_param)

        return q_param, q_state, log_pol_param, log_pol_state
