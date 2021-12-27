"""Mixins for evaluation and exploration.
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, Tuple

import distrax
import gym
import jax
from chex import Array, PRNGKey

import shinrl as srl
from shinrl import Sample, collect_samples

OBS, ACT, LOG_PROB = Array, Array, Array


class BaseGymEvalMixIn:
    """Base mixin for gym.Env evaluation."""

    def initialize(self, env, config=None) -> None:
        super().initialize(env, config)
        self._eval_env = deepcopy(self.env)

    # ########## YOU NEED TO IMPLEMENT HERE ##########

    def eval_act(self, key: PRNGKey, obs: OBS) -> Tuple[PRNGKey, ACT, LOG_PROB]:
        """Act function for exploitation. Used in evaluation.

        Args:
            key (PRNGKey)
            obs (Array): observation

        Returns:
            new_key (PRNGKey)
            act (Array)
            log_prob (Array)
        """

        raise NotImplementedError

    # ################################################

    def evaluate(self) -> Dict[str, float]:
        """
        Do sampling-based evaluation.

        Returns:
            Dict[str, float]: Dict of evaluation results.
        """
        self._eval_env.obs = self._eval_env.reset()
        self.key, samples = collect_samples(
            key=self.key,
            env=self._eval_env,
            act_fn=self.eval_act,
            num_episodes=self.config.eval_trials,
            use_state=False,
        )
        ret = (samples.rew.sum() / self.config.eval_trials).item()
        return {"Return": ret}


class BaseGymExploreMixIn:
    """Base mixin for gym.Env exploration."""

    # ########## YOU NEED TO IMPLEMENT HERE ##########

    def explore_act(self, key: PRNGKey, obs: OBS) -> Tuple[PRNGKey, ACT, LOG_PROB]:
        """Act function for exploration.

        Args:
            key (PRNGKey)
            obs (Array): observation

        Returns:
            new_key (PRNGKey)
            act (Array)
            log_prob (Array)
        """
        raise NotImplementedError

    # ################################################

    def explore(self, store_to_buffer=True) -> Sample:
        """Collect samples using explore_act function.

        Returns:
            Sample
        """
        buffer = self.buffer if store_to_buffer else None
        self.key, samples = collect_samples(
            key=self.key,
            env=self.env,
            act_fn=self.explore_act,
            num_samples=self.config.num_samples,
            buffer=buffer,
            use_state=False,
        )
        return samples


class BaseShinEvalMixIn:
    """Base mixin for ShinEnv evaluation."""

    # ########## self.data NEED TO HAVE "EvaluatePolicy" Array ##########

    def evaluate(self) -> Dict[str, float]:
        """Do oracle evaluation with `EvaluatePolicy` table.

        Returns:
            Dict[str, float]: Dict of evaluation results.
        """
        assert self.is_shin_env
        assert "EvaluatePolicy" in self.data, "EvaluatePolicy is not set."
        pol = self.data["EvaluatePolicy"]
        ret = self.env.calc_return(pol)
        return {"Return": ret}


class BaseShinExploreMixIn:
    """Base mixin for ShinEnv exploration."""

    # ########## self.data NEED TO HAVE "ExplorePolicy" Array ##########

    def initialize(self, env, config=None) -> None:
        super().initialize(env, config)
        assert self.is_shin_env
        is_continuous = isinstance(self.env.action_space, gym.spaces.Box)
        self.tb_act = _build_c_tb_act(self.env) if is_continuous else _build_tb_act()

    def explore(self, store_to_buffer=True) -> Sample:
        """Collect samples with `ExplorePolicy` table.

        Returns:
            Sample
        """
        policy = self.data["ExplorePolicy"]
        tb_explore_act = lambda key, state: self.tb_act(key, state, policy)
        buffer = self.buffer if store_to_buffer else None
        self.key, samples = collect_samples(
            key=self.key,
            env=self.env,
            act_fn=tb_explore_act,
            num_samples=self.config.num_samples,
            buffer=buffer,
            use_state=True,
        )
        return samples


def _build_tb_act() -> srl.ACT_FN:
    def tb_act(key, state, policy):
        # Act according to a policy table.
        key, new_key = jax.random.split(key)
        dist = distrax.Categorical(probs=policy[state])
        act = dist.sample(seed=key)
        log_prob = dist.log_prob(act)
        return new_key, act, log_prob

    return jax.jit(tb_act)


def _build_c_tb_act(env: srl.ShinEnv) -> srl.ACT_FN:
    def continuous_tb_act(key, state, policy):
        # Act according to a policy table.
        key, new_key = jax.random.split(key)
        dist = distrax.Categorical(probs=policy[state])
        act = dist.sample(seed=key)
        log_prob = dist.log_prob(act)
        act = env.continuous_action(act)
        return new_key, act, log_prob

    return jax.jit(continuous_tb_act)
