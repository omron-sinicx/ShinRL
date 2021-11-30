import random
from typing import Callable, Literal

import gym
import numpy as np
import torch
from scipy import special

from shinrl import utils

from .config import ViConfig

Net = Callable[[torch.Tensor], torch.Tensor]

# ----- tabular policy -----
def get_q_to_pol(_type: Literal["greedy", "eps_greedy", "softmax"] = "greedy"):
    return {
        "greedy": to_greedy_tb,
        "eps_greedy": to_eps_greedy_tb,
        "softmax": to_softmax_tb,
    }[_type]


def to_greedy_tb(value: np.ndarray, *args, **kwargs):
    return utils.eps_greedy_policy(value, eps=0)


def to_eps_greedy_tb(value: np.ndarray, config: ViConfig, step: int, *args, **kwargs):
    eps = utils.calc_eps(
        step,
        config.eps_decay,
        config.eps_warmup,
        config.eps_end,
    )
    return utils.eps_greedy_policy(value, eps=eps)


def to_softmax_tb(value: np.ndarray, config: ViConfig, *args, **kwargs):
    return special.softmax(value * config.max_tmp, axis=-1)


# ----- nn policy -----
def get_net_act(_type: Literal["greedy", "eps_greedy", "softmax"] = "greedy"):
    return {
        "greedy": act_greedy_net,
        "eps_greedy": act_eps_greedy_net,
        "softmax": act_softmax_net,
    }[_type]


def act_greedy_net(env: gym.Env, net: Net, config: ViConfig, *args, **kwargs):
    with torch.no_grad():
        obs = torch.as_tensor(
            env.obs, dtype=torch.float32, device=config.device
        ).unsqueeze(0)
        action = net(obs).argmax(1).item()
        return action, 0.0


def act_eps_greedy_net(
    env: gym.Env, net: Net, config: ViConfig, step: int, *args, **kwargs
):
    with torch.no_grad():
        eps = random.random()
        eps_thresh = utils.calc_eps(
            step,
            config.eps_decay,
            config.eps_warmup,
            config.eps_end,
        )
        if eps > eps_thresh:
            obs = torch.tensor(
                env.obs, dtype=torch.float32, device=config.device
            ).unsqueeze(0)
            action = net(obs).argmax(1).item()
        else:
            action = env.action_space.sample()
        return action, 0.0


def act_softmax_net(env: gym.Env, net: Net, config: ViConfig, *args, **kwargs):
    with torch.no_grad():
        obs = torch.as_tensor(
            env.obs, dtype=torch.float32, device=config.device
        ).unsqueeze(0)
        out = net(obs).reshape(-1).detach().cpu().numpy()
        probs = to_softmax_tb(out, config)
        log_probs = np.log(probs)
        action = np.random.choice(np.arange(0, env.action_space.n), p=probs)
        log_prob = log_probs[action]
        return action, log_prob
