"""MixIn building act_functions for GymExplore and GymEval MixIns.
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from typing import Optional

import gym

import shinrl as srl

from .config import ViConfig


class BuildNetActMixIn:
    def initialize(self, env: gym.Env, config: Optional[ViConfig] = None) -> None:
        super().initialize(env, config)
        self.explore_act = self._build_act_fn(self.config.explore.name)
        self.eval_act = self._build_act_fn(self.config.evaluate.name)

    def _build_act_fn(self, flag) -> srl.ACT_FN:
        net = self.q_net
        if flag == "oracle" or flag == "greedy":
            net_act = srl.build_discrete_greedy_net_act(net)

            def act_fn(key, obs):
                return net_act(key, obs, self.data["QNetParams"])

        elif flag == "eps_greedy":
            net_act = srl.build_eps_greedy_net_act(net)

            def act_fn(key, obs):
                n_step = self.n_step
                decay, warmup, end = (
                    self.config.eps_decay,
                    self.config.eps_warmup,
                    self.config.eps_end,
                )
                params = self.data["QNetParams"]
                return net_act(key, obs, params, n_step, decay, warmup, end)

        elif flag == "softmax":
            net_act = srl.build_softmax_net_act(net)

            def act_fn(key, obs):
                params = self.data["QNetParams"]
                return net_act(key, obs, params, self.config.softmax_tmp)

        else:
            raise NotImplementedError

        return act_fn
