"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from typing import List, Type

import gym

import shinrl as srl

from ._build_calc_params_mixin import BuildCalcParamsDpMixIn, BuildCalcParamsRlMixIn
from ._build_net_act_mixin import BuildNetActMixIn
from ._build_net_mixin import BuildNetMixIn
from ._build_table_mixin import BuildTableMixIn
from ._step_mixin import (
    DeepDpStepMixIn,
    DeepRlStepMixIn,
    TabularDpStepMixIn,
    TabularRlStepMixIn,
)
from ._target_mixin import DoubleQTargetMixIn, MunchausenTargetMixIn, QTargetMixIn
from .config import ViConfig


class DiscreteViSolver(srl.BaseSolver):
    """Value iteration (VI) solver.

    This solver implements some basic VI-based algorithms.
    For example, DiscreteViSolver turns into DQN when approx == "nn" and explore != "oracle".
    """

    DefaultConfig = ViConfig

    @staticmethod
    def make_mixins(env: gym.Env, config: ViConfig) -> List[Type[object]]:
        approx, explore = config.approx, config.explore
        APPROX, EXPLORE = config.APPROX, config.EXPLORE
        if isinstance(env, gym.Wrapper):
            is_shin_env = isinstance(env.unwrapped, srl.ShinEnv)
        else:
            is_shin_env = isinstance(env, srl.ShinEnv)

        mixin_list: List[Type[object]] = [DiscreteViSolver]

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
            mixin_list += [DeepDpStepMixIn, BuildCalcParamsDpMixIn]
        elif config.approx == APPROX.tabular and explore != EXPLORE.oracle:
            mixin_list += [TabularRlStepMixIn]
        elif config.approx == APPROX.nn and config.explore != EXPLORE.oracle:
            mixin_list += [DeepRlStepMixIn, BuildCalcParamsRlMixIn]
        else:
            raise NotImplementedError

        # Reverse the order. The latter classes have the higher priorities.
        mixin_list = mixin_list[::-1]
        return mixin_list
