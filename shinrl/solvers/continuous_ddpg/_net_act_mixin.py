"""MixIn building act_functions for GymExplore and GymEval MixIns.
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from typing import Callable, Optional

import gym

import shinrl as srl

from .config import DdpgConfig


class NetActMixIn:
    def initialize(self, env: gym.Env, config: Optional[DdpgConfig] = None) -> None:
        super().initialize(env, config)
        self.explore_act = self._build_act_fn(self.config.explore.name)
        self.eval_act = self._build_act_fn(self.config.exploit.name)

    def _build_act_fn(self, flag) -> Callable[[], srl.ACT_FN]:
        net = self.pol_net
        if flag == "oracle" or flag == "greedy":
            net_act = srl.build_continuous_greedy_net_act(net)

            def act_fn(key, obs):
                params = self.data["PolNetParams"]
                return net_act(key, obs, params)

        elif flag == "normal":
            net_act = srl.build_fixed_scale_normal_net_act(net)

            def act_fn(key, obs):
                scale = self.config.normal_scale
                params = self.data["PolNetParams"]
                return net_act(key, obs, params, scale)

        else:
            raise NotImplementedError

        return act_fn
