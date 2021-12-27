"""MixIn building act_functions for GymExplore and GymEval MixIns.
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from typing import Callable, Optional

import gym

import shinrl as srl

from .config import PiConfig


class BuildNetActMixIn:
    def initialize(self, env: gym.Env, config: Optional[PiConfig] = None) -> None:
        super().initialize(env, config)
        self.explore_act = self._build_act_fn(self.config.explore.name)
        self.eval_act = self._build_act_fn(self.config.exploit.name)

    def _build_act_fn(self, flag) -> Callable[[], srl.ACT_FN]:
        net = self.log_pol_net
        if flag == "oracle" or flag == "greedy":
            net_act = srl.build_discrete_greedy_net_act(net)

            def act_fn(key, obs):
                return net_act(key, obs, self.data["LogPolNetParams"])

        elif flag == "eps_greedy":
            net_act = srl.build_eps_greedy_net_act(net)

            def act_fn(key, obs):
                n_step = self.n_step
                decay, warmup, end = (
                    self.config.eps_decay,
                    self.config.eps_warmup,
                    self.config.eps_end,
                )
                params = self.data["LogPolNetParams"]
                return net_act(key, obs, params, n_step, decay, warmup, end)

        elif flag == "identity":
            net_act = srl.build_softmax_net_act(net)

            def act_fn(key, obs):
                params = self.data["LogPolNetParams"]
                return net_act(key, obs, params, 1)

        else:
            raise NotImplementedError

        return act_fn
