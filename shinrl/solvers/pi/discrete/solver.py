"""
Author: Toshinori Kitamura
Affiliation: NAIST
"""
from __future__ import annotations

from typing import List, Type

import gym

import shinrl as srl

from .core.config import PiConfig
from .core.net_mixin import NetActMixIn, NetInitMixIn
from .core.step_mixin import (
    DeepDpStepMixIn,
    DeepRlStepMixIn,
    TabularDpStepMixIn,
    TabularRlStepMixIn,
)
from .core.target_mixin import QTargetMixIn, SoftQTargetMixIn
from .core.tb_mixin import TbInitMixIn


class DiscretePiSolver(srl.Solver):
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

        # Prepare evaluation and exploration functions
        if is_shin_env:
            mixin_list += [srl.ShinEvalMixIn, srl.ShinExploreMixIn]
        else:
            mixin_list += [srl.GymEvalMixIn, srl.GymExploreMixIn]

        # Initialize networks & Prepare functions to take actions
        if approx == APPROX.nn:
            mixin_list += [NetInitMixIn, NetActMixIn]

        # Initialize tables
        if is_shin_env:
            mixin_list += [TbInitMixIn]

        # Add algorithm mixins to compute Q-targets
        is_q_learning = config.er_coef == 0.0
        if is_q_learning:  # Vanilla Q target
            mixin_list += [QTargetMixIn]
            config.pol_loss_fn = config.LOSS.cross_entropy_loss
        else:
            mixin_list += [SoftQTargetMixIn]
            config.pol_loss_fn = config.LOSS.kl_loss

        # Branch to tabular DP, deep DP, tabular RL, or deep RL
        if approx == APPROX.tabular and explore == EXPLORE.oracle:
            mixin_list += [TabularDpStepMixIn]
        elif approx == APPROX.nn and explore == EXPLORE.oracle:
            mixin_list += [DeepDpStepMixIn]
        elif config.approx == APPROX.tabular and explore != EXPLORE.oracle:
            mixin_list += [TabularRlStepMixIn]
        elif config.approx == APPROX.nn and config.explore != EXPLORE.oracle:
            mixin_list += [DeepRlStepMixIn]
        else:
            raise NotImplementedError

        # Reverse the order. The latter classes have the higher priorities.
        mixin_list = mixin_list[::-1]
        return mixin_list
