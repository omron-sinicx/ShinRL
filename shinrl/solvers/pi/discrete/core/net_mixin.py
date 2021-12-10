"""MixIns for deep algorithms, i.e., config.approx == APPROX.nn.

* NetInitMixIn: Initialize neural networks.
* NetActMixIn: For exploration and exploitation with neural networks.
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from __future__ import annotations

from copy import deepcopy
from typing import Optional

import distrax
import gym
import jax
import jax.numpy as jnp
import optax

import shinrl as srl

from .config import PiConfig


class NetInitMixIn:
    """MixIn to prepare networks"""

    def initialize(self, env: gym.Env, config: Optional[PiConfig] = None) -> None:
        super().initialize(env, config)

        # make networks
        n_out = env.action_space.n
        depth, hidden = self.config.depth, self.config.hidden
        act_layer = getattr(jax.nn, self.config.activation.name)
        if len(env.observation_space.shape) == 1:
            net = srl.build_forward_fc(n_out, depth, hidden, act_layer)
        else:
            net = srl.build_forward_conv(n_out, depth, hidden, act_layer)

        # make q network
        self.q_net = net
        self.key, key = jax.random.split(self.key)
        sample_init = jnp.ones([1, *env.observation_space.shape])
        net_param = net.init(key, sample_init)
        self.prms_dict.set("QNet", net_param)
        self.prms_dict.set("TargQNet", deepcopy(net_param))

        optimizer = getattr(optax, config.optimizer.name)
        opt = optimizer(learning_rate=config.q_lr)
        self.q_opt = opt
        opt_state = opt.init(net_param)
        self.prms_dict.set("QOpt", opt_state)

        # make policy network
        self.pol_net = net
        self.key, key = jax.random.split(self.key)
        sample_init = jnp.ones([1, *env.observation_space.shape])
        net_param = net.init(key, sample_init)
        self.prms_dict.set("PolNet", net_param)

        optimizer = getattr(optax, config.optimizer.name)
        opt = optimizer(learning_rate=config.pol_lr)
        self.pol_opt = opt
        opt_state = opt.init(net_param)
        self.prms_dict.set("PolOpt", opt_state)


Distributions = {
    "oracle": distrax.Greedy,
    "greedy": distrax.Greedy,
    "eps_greedy": distrax.EpsilonGreedy,
    "softmax": distrax.Softmax,
}


class NetActMixIn:
    """MixIn with act_functions for GymExplore and GymEval MixIns."""

    def initialize(self, env: gym.Env, config: Optional[PiConfig] = None) -> None:
        super().initialize(env, config)
        ExploreDist = Distributions[self.config.explore.name]
        self._explore_act = srl.build_net_act(ExploreDist, self.q_net)
        ExploitDist = Distributions[self.config.exploit.name]
        self._exploit_act = srl.build_net_act(ExploitDist, self.q_net)

    def make_explore_act(self) -> srl.ACT_FN:
        params = self.prms_dict["PolNet"]
        kwargs = {
            "eps_greedy": {
                "epsilon": srl.calc_eps(
                    self.n_step,
                    self.config.eps_decay,
                    self.config.eps_warmup,
                    self.config.eps_end,
                ),
            },
            "softmax": {"temperature": self.config.max_tmp},
        }[self.config.explore.name]
        return lambda key, obs: self._explore_act(key, obs, params, **kwargs)

    def make_exploit_act(self) -> srl.ACT_FN:
        params = self.prms_dict["PolNet"]
        kwargs = {
            "greedy": {},
            "softmax": {"temperature": self.config.max_tmp},
        }[self.config.exploit.name]
        return lambda key, obs: self._exploit_act(key, obs, params, **kwargs)
