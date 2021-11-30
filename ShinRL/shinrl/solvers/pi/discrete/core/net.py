from typing import Tuple

import gym
import torch
from torch import nn, optim

from .config import PiConfig


def make_net_opt(
    env: gym.Env, lr: float, config: PiConfig
) -> Tuple[nn.Module, optim.Optimizer]:
    if len(env.observation_space.shape) == 1:
        _net = fc_net
    elif env.observation_space.shape == (1, 28, 28):
        _net = conv_net
    elif env.observation_space.shape[:2] == (10, 10):
        # for minatar
        _net = minatar_net
    net = _net(env, config).to(config.device)
    _opt = getattr(torch.optim, config.optimizer)
    opt = _opt(net.parameters(), lr=lr)
    return net, opt


def fc_net(env: gym.Env, config: PiConfig) -> nn.Module:
    """For Env with tuple observation.

    Args:
        env (gym.Env)
        config (PiConfig)

    Returns:
        nn.Module
    """
    modules = []
    obs_shape = env.observation_space.shape[0]
    n_acts = env.action_space.n
    depth, hidden = config.depth, config.hidden
    act_layer = getattr(nn, config.activation)
    if depth > 0:
        modules.append(nn.Linear(obs_shape, hidden))
        for _ in range(depth - 1):
            modules += [act_layer(), nn.Linear(hidden, hidden)]
        modules += [act_layer(), nn.Linear(hidden, n_acts)]
    else:
        modules.append(nn.Linear(obs_shape, n_acts))
    return nn.Sequential(*modules)


def conv_net(env: gym.Env, config: PiConfig) -> nn.Module:
    """For env with image shape == (1, 28, 28).
    Especially for ShinEnv with image observation.

    Args:
        env (gym.Env): env with (1, 28, 28) observation space.
        config (PiConfig)

    Returns:
        nn.Module
    """
    depth, hidden = config.depth, config.hidden
    act_layer = getattr(nn, config.activation)
    n_acts = env.action_space.n
    conv_modules = [
        nn.Conv2d(1, 10, kernel_size=5, stride=2),
        nn.Conv2d(10, 20, kernel_size=5, stride=2),
        nn.Flatten(),
    ]
    fc_modules = []
    if depth > 0:
        fc_modules.append(nn.Linear(320, hidden))
        for _ in range(depth - 1):
            fc_modules += [act_layer(), nn.Linear(hidden, hidden)]
        fc_modules += [act_layer(), nn.Linear(hidden, n_acts)]
    else:
        fc_modules.append(nn.Linear(320, n_acts))
    return nn.Sequential(*(conv_modules + fc_modules))


class Permute(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


def minatar_net(env: gym.Env, config: PiConfig) -> nn.Module:
    """For MinAtar env.

    Args:
        env (gym.Env)
        config (PiConfig)

    Returns:
        nn.Module
    """
    hidden = config.hidden
    act_layer = getattr(nn, config.activation)
    in_channels = env.observation_space.shape[2]

    def size_linear_unit(size: int, kernel_size: int = 3, stride: int = 1) -> int:
        return (size - (kernel_size - 1) - 1) // stride + 1

    num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
    return nn.Sequential(
        Permute(),
        nn.Conv2d(in_channels, 16, kernel_size=3, stride=1),
        act_layer(),
        nn.Flatten(),
        nn.Linear(num_linear_units, hidden),
        act_layer(),
        nn.Linear(hidden, env.action_space.n),
    )
