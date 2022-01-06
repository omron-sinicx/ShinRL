"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
import chex

from shinrl import Config


@chex.dataclass
class SolverConfig(Config):
    """
    Args:
        seed (int): random seed for the solver
        discount (float): discount factor
        eval_trials (int): number of trials for evaluation
        eval_interval (int): interval to evaluate
        add_interval (int): interval to add a scalar to the history
        steps_per_epoch (int): number of steps per one epoch
        verbose (bool): flag to show logged information on stdout
    """

    seed: int = 0
    discount: float = 0.99
    eval_trials: int = 10
    eval_interval: int = 100
    add_interval: int = 100
    steps_per_epoch: int = 1000
    verbose: bool = True
