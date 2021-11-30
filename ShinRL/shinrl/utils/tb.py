from __future__ import annotations

from typing import Tuple

import numpy as np


def calc_eps(
    step: int, decay_period: float, warmup_steps: int, eps_end: float
) -> float:
    """Same as the one of dopamine(https://github.com/google/dopamine/blob/master/dopamine/agents/dqn/dqn_agent.py#L43).

    Args:
        step (int): the current step
        decay_period (float): the period over which epsilon is decayed.
        warmup_steps (int): the number of steps taken before epsilon is decayed.
        eps_end (float): the final value to which to decay the epsilon parameter.

    Returns:
        float: the epsilon value computed according to the schedule.
    """

    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - eps_end) * steps_left / decay_period
    bonus = np.clip(bonus, 0.0, 1.0 - eps_end)
    return eps_end + bonus


def eps_greedy_policy(values: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Return the epsilon-greedy policy
    Args:
        values (np.ndarray): the state-action value
        eps (float, optional): the epsilon value for the randomness

    Returns:
        np.ndarray: the epsilon-greedy policy
    """
    policy_probs = np.zeros_like(values)
    policy_probs[np.arange(values.shape[0]), np.argmax(values, axis=-1)] = 1.0 - eps
    policy_probs += eps / (policy_probs.shape[-1])
    return policy_probs


def get_tb_act(env, policy: np.ndarray) -> Tuple[int, float]:
    """Return the discrete action and the log-policy from a tabular policy

    Args:
        env (shinrl.envs.ShinEnv):
        policy (np.ndarray): A tabular policy

    Returns:
        Tuple[int, float]: an action and its log-probability
    """
    probs = policy[env.get_state()]
    if np.sum(probs) != 1:
        probs /= np.sum(probs)
    act = np.random.choice(np.arange(0, env.dA), p=probs)
    return act, np.log(probs[act])
