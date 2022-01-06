"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from __future__ import annotations

from typing import List, Type

import gym

import shinrl as srl

from ._build_calc_params_mixin import BuildCalcParamsDpMixIn, BuildCalcParamsRlMixIn
from ._build_net_act_mixin import BuildNetActMixIn
from ._build_net_mixin import BuildNetMixIn
from ._build_table_mixin import BuildTableMixIn
from ._step_mixin import DeepDpStepMixIn, DeepRlStepMixIn
from ._target_mixin import QTargetMixIn
from .config import DdpgConfig


class ContinuousDdpgSolver(srl.BaseSolver):
    """Deep Deterministic Policy Gradient (DDPG) solver.

    This solver implements variants of DDPG algorithm for a continuous action space.
    For example, ContinuousDdpgSolver with explore == "oracle" uses all the state and action pairs.
    """

    DefaultConfig = DdpgConfig

    @staticmethod
    def make_mixins(env: gym.Env, config: DdpgConfig) -> List[Type[object]]:
        approx, explore = config.approx, config.explore
        APPROX, EXPLORE = config.APPROX, config.EXPLORE
        if isinstance(env, gym.Wrapper):
            is_shin_env = isinstance(env.unwrapped, srl.ShinEnv)
        else:
            is_shin_env = isinstance(env, srl.ShinEnv)

        mixin_list: List[Type[object]] = [ContinuousDdpgSolver]

        # Add base mixins for evaluation and exploration.
        if is_shin_env:
            mixin_list += [srl.BaseShinEvalMixIn, srl.BaseShinExploreMixIn]
        else:
            mixin_list += [srl.BaseGymEvalMixIn, srl.BaseGymExploreMixIn]

        # Add mixins to prepare networks.
        if approx == APPROX.nn:
            mixin_list += [BuildNetMixIn, BuildNetActMixIn]

        # Add mixins to prepare tables.
        if is_shin_env:
            mixin_list += [BuildTableMixIn]

        # Add algorithm mixins to compute Q-targets
        mixin_list += [QTargetMixIn]

        # Branch to deep DP and deep RL
        if approx == APPROX.nn and explore == EXPLORE.oracle:
            mixin_list += [DeepDpStepMixIn, BuildCalcParamsDpMixIn]
        elif config.approx == APPROX.nn and config.explore != EXPLORE.oracle:
            mixin_list += [DeepRlStepMixIn, BuildCalcParamsRlMixIn]
        else:
            raise NotImplementedError

        # Reverse the order. The latter classes have the higher priorities.
        mixin_list = mixin_list[::-1]
        return mixin_list
