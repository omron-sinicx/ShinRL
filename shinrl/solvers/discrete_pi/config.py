"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
import enum
from enum import auto
from inspect import getmembers
from typing import ClassVar, Type

import chex
import jax
import optax

import shinrl as srl
from shinrl import SolverConfig


class EXPLORE(enum.IntEnum):
    oracle = auto()
    eps_greedy = auto()
    softmax = auto()


class EXPLOIT(enum.IntEnum):
    softmax = auto()
    greedy = auto()


class APPROX(enum.IntEnum):
    tabular = auto()
    nn = auto()


ACTIVATION = enum.IntEnum(
    "ACTIVATION",
    {fnc[0]: enum.auto() for fnc in getmembers(jax.nn) if callable(fnc[1])},
)


OPTIMIZER = enum.IntEnum(
    "OPTIMIZER",
    {fnc[0]: enum.auto() for fnc in getmembers(optax._src.alias) if callable(fnc[1])},
)


LOSS = enum.IntEnum(
    "LOSS",
    {fnc[0]: enum.auto() for fnc in getmembers(srl._calc.loss) if callable(fnc[1])},
)


@chex.dataclass
class PiConfig(SolverConfig):
    """Config for PiSolver.

    Args:
        explore (EXPLORE):
            Type of the policy in exploration. The solver uses all the state-action pairs if 'oracle'.
        exploit (EXPLOIT): Type of the policy in evaluation.
        approx (APPROX): Type of the function approximation.

        eps_end (float): Epsilon value at the end of the eps-greedy exploration.
        eps_warmup (int): Epsilon value is set 1.0 until 'eps_warmup'.
        eps_decay (int): Interval to decrease the epsilon value.
        softmax_tmp (float): temperature parameter for softmax.

        pol_lr (float): Learning rate of the pol.
        q_lr (float): Learning rate of the q.
        num_samples (int): Number of samples to collect when explore != 'oracle'.
        batch_size (int): Size of the minibatch.
        buffer_size (int): Replay buffer capacity.
        er_coef (float): Coefficient for entropy regularization.

        hidden (int): Size of the linear layer.
        depth (int): Depth of the linear layer.
        target_update_interval (int): Interval to update the target network.
        activation (str): Activation function.
        optimizer (str): Optimizer for nn.
        q_loss_fn (str): Loss function for q.
        pol_loss_fn (str): Loss function for policy. Automatically changed in make_mixin.
    """

    # class variables
    EXPLORE: ClassVar[Type[EXPLORE]] = EXPLORE
    EXPLOIT: ClassVar[Type[EXPLOIT]] = EXPLOIT
    APPROX: ClassVar[Type[APPROX]] = APPROX
    ACTIVATION: ClassVar[Type[ACTIVATION]] = ACTIVATION
    LOSS: ClassVar[Type[LOSS]] = LOSS
    OPTIMIZER: ClassVar[Type[OPTIMIZER]] = OPTIMIZER

    explore: EXPLORE = EXPLORE.oracle
    exploit: EXPLOIT = EXPLOIT.greedy
    approx: APPROX = APPROX.tabular

    # algorithm configs
    pol_lr: float = 1e-3
    q_lr: float = 1e-3
    num_samples: int = 4
    buffer_size: int = int(1e6)
    batch_size: int = 32
    er_coef: float = 0.0

    # policy settings
    eps_end: float = 0.1
    eps_warmup: int = 0
    eps_decay: int = 10 ** 5
    softmax_tmp: float = 1.0

    # network configs
    hidden: int = 128
    depth: int = 2
    target_update_interval: int = 1000
    activation: ACTIVATION = ACTIVATION.relu
    optimizer: OPTIMIZER = OPTIMIZER.adam
    q_loss_fn: LOSS = LOSS.l2_loss
    pol_loss_fn: LOSS = LOSS.cross_entropy_loss  # Automatically changed in make_mixin.
