""" JAX functions to collect samples. 
Author: Toshinori Kitamura
Affiliation: NAIST
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import gym
import numpy as np
from chex import Array, PRNGKey
from cpprb import ReplayBuffer

OBS, ACT, LOG_PROB = Array, Array, Array
ACT_FN = Callable[[PRNGKey, OBS], Tuple[PRNGKey, ACT, LOG_PROB]]


class Sample(NamedTuple):
    obs: Optional[Array] = None
    next_obs: Optional[Array] = None
    rew: Optional[Array] = None
    done: Optional[Array] = None
    log_prob: Optional[Array] = None
    act: Optional[Array] = None
    timeout: Optional[Array] = None
    state: Optional[Array] = None
    next_state: Optional[Array] = None


def act_and_step(
    key: PRNGKey,
    env: gym.Env,
    act_fn: ACT_FN,
    use_state: bool,
) -> Tuple[PRNGKey, Dict[str, Array]]:
    """Do one-step and return the results.

    Args:
        key (PRNGKey)
        env (GymEnv)
        act_fn (ACT_FN)
        use_state (bool): Use a state if True. Use an obs otherwise.

    Returns:
        new_key (PRNGKey): generated new key.
        sample (Sample): One step sample.
    """
    is_shin_env = hasattr(env, "mdp")

    obs = env.obs
    if is_shin_env:
        state = env.get_state()
    act_fn_input = state if use_state and is_shin_env else obs
    new_key, act, log_prob = act_fn(key, act_fn_input)
    act = np.asarray(act)
    next_obs, rew, done, info = env.step(act)
    env.obs = next_obs
    timeout = info["TimeLimit.truncated"] if "TimeLimit.truncated" in info else False
    sample = {
        "obs": obs,
        "next_obs": next_obs,
        "rew": rew,
        "done": done,
        "log_prob": log_prob,
        "act": act,
        "timeout": timeout,
    }
    if is_shin_env:
        sample["state"] = state
        sample["next_state"] = env.get_state()
    return new_key, sample


def collect_samples(
    key: PRNGKey,
    env: gym.Env,
    act_fn: ACT_FN,
    num_samples: Optional[int] = None,
    num_episodes: Optional[int] = None,
    buffer: Optional[ReplayBuffer] = None,
    use_state: bool = False,
    render: bool = False,
) -> Sample:

    """
    Args:
        key (PRNGKey)
        env (Union[ShinEnv, gym.Env])
        act_fn (GYM_ACT_FN): A function taking an observation and return action and log_prob.
        num_samples (Optional[int], optional): Number of samples to collect.
        num_episodes (Optional[int], optional): Number of episodes to collect.
        buffer (Optional[ReplayBuffer], optional): Do buffer.add and call on_episode_end if not None.
        use_state (bool, optional): Use state instead of obs for act_fn.
        render (bool, optional):

    Returns:
        new_key (PRNGKey): generated new key.
        samples (Samples): Collected samples.
    """
    assert not (num_samples is None and num_episodes is None)
    num_samples = -1 if num_episodes is not None else num_samples

    assert hasattr(
        env, "obs"
    ), 'env has no attribute "obs". Do env.obs = env.reset() before collect_samples.'
    done_count, step_count = 0, 0
    samples: Dict[str, List[Array]] = defaultdict(lambda: [])

    while True:
        if render:
            time.sleep(1 / 20)
            env.render()

        key, sample = act_and_step(key, env, act_fn, use_state)
        if buffer is not None:
            buffer.add(**sample)
        step_count += 1
        if sample["done"]:
            env.obs = env.reset()
            done_count += 1
            if buffer is not None:
                buffer.on_episode_end()
        for _key, _val in sample.items():
            samples[_key].append(_val)

        if step_count == num_samples or done_count == num_episodes:
            break

    for _key, _val in samples.items():
        _val = np.array(_val)
        _val = _val[:, None] if len(_val.shape) == 1 else _val
        samples[_key] = _val
    return key, Sample(**samples)


def make_replay_buffer(env: gym.Env, size: int) -> ReplayBuffer:
    """Make a replay buffer.
    If not ShinEnv:
        Returns a ReplayBuffer with ("rew", "done", "obs", "act", "log_prob", "timeout").

    If ShinEnv:
        Returns a ReplayBuffer with ("rew", "done", "obs", "act", "log_prob", "timeout", "state").
    """
    is_shin_env = hasattr(env, "mdp")
    if isinstance(env.action_space, gym.spaces.Discrete):
        act_type, act_shape = int, 1
    elif isinstance(env.action_space, gym.spaces.Box):
        act_type, act_shape = float, env.action_space.shape
    env_dict = {
        "rew": {"dtype": float, "shape": 1},
        "done": {"dtype": bool, "shape": 1},
        "obs": {"dtype": float, "shape": env.observation_space.shape},
        "act": {"dtype": act_type, "shape": act_shape},
        "log_prob": {"dtype": float, "shape": 1},
        "timeout": {"dtype": bool, "shape": 1},
    }
    if is_shin_env:
        env_dict.update({"state": {"dtype": int, "shape": 1}})
        return ReplayBuffer(size, env_dict, next_of=("obs", "state"))
    return ReplayBuffer(size, env_dict, next_of=("obs",))
