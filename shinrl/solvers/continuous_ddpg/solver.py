"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from __future__ import annotations

from typing import List, Type

import gym

import shinrl as srl

from ._build_net_mixin import BuildNetMixIn
from ._build_table_mixin import BuildTableMixIn
from ._calc_params_mixin import CalcParamsDpMixIn, CalcParamsRlMixIn
from ._net_act_mixin import NetActMixIn
from ._target_mixin import QTargetMixIn
from .config import DdpgConfig
from .step_mixin import DeepDpStepMixIn, DeepRlStepMixIn


class ContinuousDdpgSolver(srl.BaseSolver):
    DefaultConfig = DdpgConfig

    @staticmethod
    def make_mixins(env: gym.Env, config: DdpgConfig) -> List[Type[object]]:
        approx, explore = config.approx, config.explore
        APPROX, EXPLORE = config.APPROX, config.EXPLORE
        mixin_list: List[Type[object]] = [ContinuousDdpgSolver]
        if isinstance(env, gym.Wrapper):
            is_shin_env = isinstance(env.unwrapped, srl.ShinEnv)
        else:
            is_shin_env = isinstance(env, srl.ShinEnv)

        # Set evaluation and exploration mixins
        if is_shin_env:
            mixin_list += [srl.BaseShinEvalMixIn, srl.BaseShinExploreMixIn]
        else:
            mixin_list += [srl.BaseGymEvalMixIn, srl.BaseGymExploreMixIn]

        if approx == APPROX.nn:
            mixin_list += [BuildNetMixIn, NetActMixIn]

        if is_shin_env:
            mixin_list += [BuildTableMixIn]

        # Add algorithm mixins to compute Q-targets
        mixin_list += [QTargetMixIn]

        # Branch to tabular DP, deep DP, tabular RL, or deep RL
        if approx == APPROX.nn and explore == EXPLORE.oracle:
            mixin_list += [DeepDpStepMixIn, CalcParamsDpMixIn]
        elif config.approx == APPROX.nn and config.explore != EXPLORE.oracle:
            mixin_list += [DeepRlStepMixIn, CalcParamsRlMixIn]
        else:
            raise NotImplementedError

        # Reverse the order. The latter classes have the higher priorities.
        mixin_list = mixin_list[::-1]
        return mixin_list
