from dataclasses import dataclass
from typing import Literal

import torch

from shinrl.solvers.base import BaseConfig


@dataclass
class PiConfig(BaseConfig):
    """Config for PiSolver.

    Args:
        explore (Literal["oracle", "eps_greedy", "softmax"]):
            Type of the policy in exploration. The solver uses all the state-action pairs if 'oracle'.
        exploit (Literal["greedy", "softmax"]): Type of the policy in evaluation.
        approx (Literal["tabular", "nn"]): Type of the function approximation.
        loss_fn (str): Loss function for critic.
        num_samples (int): Number of samples to collect when explore != 'oracle'
        minibatch_size (int): Size of the minibatch
        buffer_size (int): Replay buffer capacity.
        target_update_interval (int): Interval to update the target network.
        er_coef (float): Coefficient for entropy regularization.
        noise_scale (float): Scaling factor for error tolerant analysis.
        eps_end (float): Epsilon value at the end of the eps-greedy exploration.
        eps_warmup (int): Epsilon value is set 1.0 until 'eps_warmup'.
        eps_decay (int): Interval to decrease the epsilon value.
        max_tmp (float): temperature parameter for softmax and mellowmax
        hidden (int): Size of the linear layer
        depth (int): Depth of the linear layer
        activation (str): Type of the activation layer
        optimizer (str): Type of the optimizer
        actor_lr (float): Learning rate of the actor.
        ciritic_lr (float): Learning rate of the critic.
        device (str): Device for torch.
    """

    explore: Literal["oracle", "eps_greedy", "softmax"] = "oracle"
    exploit: Literal["greedy", "softmax"] = "greedy"
    approx: Literal["tabular", "nn"] = "tabular"
    # algorithm settings
    loss_fn: str = "mse_loss"
    num_samples: int = 4
    minibatch_size: int = 32
    buffer_size: int = int(1e5)
    target_update_interval: int = 1000
    er_coef: float = 0.0
    noise_scale: float = 0.0
    # policy settings
    eps_end: float = 0.1
    eps_warmup: int = 0
    eps_decay: int = 10 ** 5
    max_tmp: float = 1.0
    # network settings
    hidden: int = 128
    depth: int = 2
    activation: str = "ReLU"
    optimizer: str = "Adam"
    actor_lr: float = 0.001
    critic_lr: float = 0.001
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
