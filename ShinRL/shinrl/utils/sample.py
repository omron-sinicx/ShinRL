from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, Optional, Tuple, Union

import gym
import numpy as np
import torch
from cpprb import ReplayBuffer
from torch import Tensor

Array = Union[Tensor, np.ndarray]


TNSR_TYPES = {
    "obs": torch.float32,
    "next_obs": torch.float32,
    "rew": torch.float32,
    "done": torch.bool,
    "act_dsc": torch.long,
    "act_cont": torch.float32,
    "log_prob": torch.float32,
    "timeout": torch.bool,
    "state": torch.long,
    "next_state": torch.long,
}


NP_TYPES = {
    "obs": np.float32,
    "next_obs": np.float32,
    "rew": np.float32,
    "done": np.bool_,
    "act_dsc": np.int32,
    "act_cont": np.float32,
    "log_prob": np.float32,
    "timeout": np.bool_,
    "state": np.int32,
    "next_state": np.int32,
}


@dataclass
class Samples:
    obs: Array
    next_obs: Array
    rew: Array
    done: Array
    log_prob: Array
    act: Array
    timeout: Array
    state: Optional[Array] = None
    next_state: Optional[Array] = None

    def __post_init__(self) -> None:
        """Cast types and reshape the arrays"""
        self._i = 0
        for field in fields(self):
            name = field.name
            val = getattr(self, name)
            if isinstance(val, np.ndarray):
                if name == "act":
                    if val.dtype in [np.int32, np.int64]:
                        val = val.astype(NP_TYPES["act_dsc"])
                    else:
                        val = val.astype(NP_TYPES["act_cont"])
                else:
                    val = val.astype(NP_TYPES[name])
                if val.shape[-1] == 1:
                    val = np.squeeze(val, axis=-1)
            elif isinstance(val, Tensor):
                if name == "act":
                    if val.dtype in [torch.int32, torch.long]:
                        val = val.type(TNSR_TYPES["act_dsc"])
                    else:
                        val = val.type(TNSR_TYPES["act_cont"])
                else:
                    val = val.type(TNSR_TYPES[name])
                if val.shape[-1] == 1:
                    val = val.squeeze(dim=-1)
            if val is not None:
                setattr(self, name, val)

    def np_to_tnsr(self, device: str = "cpu") -> Samples:
        samples = {}
        for field in fields(self):
            name = field.name
            val = getattr(self, name)
            if val is not None:
                samples[name] = torch.tensor(val, device=device)
        return Samples(**samples)

    def tnsr_to_np(self) -> Samples:
        samples = {}
        for field in fields(self):
            name = field.name
            val = getattr(self, name)
            if val is not None:
                samples[name] = val.cpu().detach().numpy()
        return Samples(**samples)

    def __next__(self) -> Samples:
        if self._i == len(self.rew):
            raise StopIteration()
        vals = {field.name: getattr(self, field.name) for field in fields(self)}
        vals = {name: val[self._i] for name, val in vals.items() if val is not None}
        self._i += 1
        return Samples(**vals)

    def __iter__(self):
        return self


def collect_samples(
    env: gym.Env,
    get_act: Callable[[gym.Env], Tuple[float, float]],
    num_samples: Optional[int] = None,
    num_episodes: Optional[int] = None,
    render: bool = False,
    buffer: Optional[ReplayBuffer] = None,
    get_act_args: Dict[str, Any] = {},
) -> Samples:
    """

    Args:
        env (gym.Env):
        get_act (Callable[[gym.Env], Tuple[float, float]]): a function taking an env and return action with log_prob
        num_samples (Optional[int], optional): Number of samples to collect
        num_episodes (Optional[int], optional): Number of episodes to collect.
        render (bool, optional):
        buffer (Optional[ReplayBuffer], optional): Do buffer.add and call on_episode_end if not None.
        get_act_args (Dict[str, Any], optional): Args for get_act method.

    Returns:
        samples (Samples): collected samples
    """
    is_shin_env = hasattr(env, "transition_matrix")
    assert hasattr(
        env, "obs"
    ), 'env has no attribute "obs". Run env.obs = env.reset() before collect_samples.'
    samples = defaultdict(lambda: [])
    done, done_count, step_count = False, 0, 0

    # does not allow num_samples != None and num_episodes != None
    assert not (num_samples is None and num_episodes is None)
    num_samples = -1 if num_episodes is not None else num_samples

    while True:
        if render:
            time.sleep(1 / 20)
            env.render()
        sample = {}
        # add current info
        if is_shin_env:
            s = env.get_state()
            obs = env.observation(s)
        else:
            obs = env.obs
        sample["obs"] = obs

        # do one step
        action, log_prob = get_act(env=env, **get_act_args)
        next_obs, rew, done, info = env.step(action)
        env.obs = next_obs
        if is_shin_env and env.action_mode == "continuous":
            action = env.to_continuous_action(action)
        step_count += 1

        # add next info
        timeout = (
            info["TimeLimit.truncated"] if "TimeLimit.truncated" in info else False
        )
        sample.update(
            {
                "next_obs": next_obs,
                "rew": rew,
                "done": done,
                "log_prob": log_prob,
                "act": action,
                "timeout": timeout,
            }
        )
        if is_shin_env:
            sample.update({"state": s, "next_state": env.get_state()})

        if buffer is not None:
            buffer.add(**sample)

        if done:
            env.obs = env.reset()
            done_count += 1
            if buffer is not None:
                buffer.on_episode_end()

        for key, val in sample.items():
            samples[key].append(val)
        if step_count == num_samples or done_count == num_episodes:
            break
    return Samples(**{key: np.array(val) for key, val in samples.items()})


def make_replay_buffer(env: gym.Env, size: int) -> ReplayBuffer:
    is_shin_env = hasattr(env, "transition_matrix")
    if isinstance(env.action_space, gym.spaces.Discrete):
        act_type, act_shape = "act_dsc", 1
    elif isinstance(env.action_space, gym.spaces.Box):
        act_type, act_shape = "act_cont", env.action_space.shape
    env_dict = {
        "rew": {"dtype": NP_TYPES["rew"], "shape": 1},
        "done": {"dtype": NP_TYPES["done"], "shape": 1},
        "obs": {"dtype": NP_TYPES["obs"], "shape": env.observation_space.shape},
        "act": {"dtype": NP_TYPES[act_type], "shape": act_shape},
        "log_prob": {"dtype": NP_TYPES["log_prob"], "shape": 1},
        "timeout": {"dtype": NP_TYPES["timeout"], "shape": 1},
    }
    if is_shin_env:
        env_dict.update({"state": {"dtype": NP_TYPES["state"], "shape": 1}})
        return ReplayBuffer(size, env_dict, next_of=("obs", "state"))
    return ReplayBuffer(size, env_dict, next_of=("obs",))
