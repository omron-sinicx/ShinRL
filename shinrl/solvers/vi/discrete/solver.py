"""
Author: Toshinori Kitamura
Affiliation: NAIST
"""
from __future__ import annotations

from typing import List, Type

import gym

import shinrl as srl

from .core.config import ViConfig
from .core.net_mixin import NetActMixIn, NetInitMixIn
from .core.step_mixin import (
    DeepDpStepMixIn,
    DeepRlStepMixIn,
    TabularDpStepMixIn,
    TabularRlStepMixIn,
)
from .core.target_mixin import DoubleQTargetMixIn, MunchausenTargetMixIn, QTargetMixIn
from .core.tb_mixin import TbInitMixIn


class DiscreteViSolver(srl.Solver):
    """Value iteration solver.

    Value iteration (VI) is a classical approach to find the optimal policy and its values in a MDP.
    Many recent RL algorithms have been extended from classical VI.
    For example, the popular DQN algorithm can be considered as
    a VI extension with function approximation and exploration.

    This ViSolver changes its behavior from VI to Munchausen-DQN according to 'config':

    * Value Iteration: approx == APPROX.tabular & explore == EXPLORE.oracle.
    * Tabular Q learning: approx == APPROX.tabular & explore != EXPLORE.oracle.
    * DQN: approx == APPROX.nn & explore != EXPLORE.oracle.
    * Deep VI: approx == APPROX.nn & explore == EXPLORE.oracle. Approximate computed table with NN. Usefule to evaluate the NN capacity.
    * Munchausen VI, tabular Q or DQN: same as above except kl_coef != 0 or er_coef != 0.
    """

    DefaultConfig = ViConfig

    @staticmethod
    def make_mixins(env: gym.Env, config: ViConfig) -> List[Type[object]]:
        approx, explore = config.approx, config.explore
        APPROX, EXPLORE = config.APPROX, config.EXPLORE
        mixin_list: List[Type[object]] = [DiscreteViSolver]
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
        is_q_learning = (config.er_coef == 0.0) * (config.kl_coef == 0.0)
        use_double_q = config.use_double_q
        if is_q_learning and not use_double_q:  # Vanilla Q target
            mixin_list += [QTargetMixIn]
        elif is_q_learning and use_double_q:  # Double Q target
            mixin_list += [DoubleQTargetMixIn]
        elif not is_q_learning and not use_double_q:  # Munchausen Q target
            mixin_list += [MunchausenTargetMixIn]
        else:
            raise NotImplementedError

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
