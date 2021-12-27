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
from ._target_mixin import QTargetMixIn, SoftQTargetMixIn
from .config import PiConfig
from .step_mixin import (
    DeepDpStepMixIn,
    DeepRlStepMixIn,
    TabularDpStepMixIn,
    TabularRlStepMixIn,
)


class DiscretePiSolver(srl.BaseSolver):
    """Policy iteration solver. """

    DefaultConfig = PiConfig

    @staticmethod
    def make_mixins(env: gym.Env, config: PiConfig) -> List[Type[object]]:
        approx, explore = config.approx, config.explore
        APPROX, EXPLORE = config.APPROX, config.EXPLORE
        mixin_list: List[Type[object]] = [DiscretePiSolver]
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
        is_q_learning = config.er_coef == 0.0
        if is_q_learning:  # Vanilla Q target
            mixin_list += [QTargetMixIn]
            config.pol_loss_fn = config.LOSS.cross_entropy_loss
        else:  # Soft Q target
            mixin_list += [SoftQTargetMixIn]
            config.pol_loss_fn = config.LOSS.kl_loss

        # Branch to tabular DP, deep DP, tabular RL, or deep RL
        if approx == APPROX.tabular and explore == EXPLORE.oracle:
            mixin_list += [TabularDpStepMixIn]
        elif approx == APPROX.nn and explore == EXPLORE.oracle:
            mixin_list += [DeepDpStepMixIn, CalcParamsDpMixIn]
        elif config.approx == APPROX.tabular and explore != EXPLORE.oracle:
            mixin_list += [TabularRlStepMixIn]
        elif config.approx == APPROX.nn and config.explore != EXPLORE.oracle:
            mixin_list += [DeepRlStepMixIn, CalcParamsRlMixIn]
        else:
            raise NotImplementedError

        # Reverse the order. The latter classes have the higher priorities.
        mixin_list = mixin_list[::-1]
        return mixin_list
