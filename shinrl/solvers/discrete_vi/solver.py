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
from ._target_mixin import DoubleQTargetMixIn, MunchausenTargetMixIn, QTargetMixIn
from .config import ViConfig
from .step_mixin import (
    DeepDpStepMixIn,
    DeepRlStepMixIn,
    TabularDpStepMixIn,
    TabularRlStepMixIn,
)


class DiscreteViSolver(srl.BaseSolver):
    """Value iteration solver.

    Value iteration (VI) is a classical approach to find the optimal policy and its values in a MDP.
    Many recent RL algorithms have been extended from classical VI.
    For example, the popular DQN algorithm can be considered as
    a VI extension with function approximation and exploration.

    This ViSolver changes its behavior according to 'config':

    * Value Iteration: approx == "tabular" & explore == "oracle".
    * Tabular Q learning: approx == "tabular" & explore != "oracle".
    * DQN: approx == "nn" & explore != "oracle".
    * Deep VI: approx == "nn" & explore == "oracle". Approximate computed table with NN. Usefule to evaluate the NN capacity.
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

        # Set base mixins for evaluation and exploration.
        if is_shin_env:
            mixin_list += [srl.BaseShinEvalMixIn, srl.BaseShinExploreMixIn]
        else:
            mixin_list += [srl.BaseGymEvalMixIn, srl.BaseGymExploreMixIn]

        if approx == APPROX.nn:
            # Prepare networks: "q_net"
            mixin_list += [BuildNetMixIn, NetActMixIn]

        if is_shin_env:
            # Prepare tables: "Q"
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
