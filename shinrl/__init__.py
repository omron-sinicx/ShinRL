from gym.envs.registration import register

# Common mathmetical (jitted) functions & classes.
from ._calc.backup_dp import (
    calc_optimal_q,
    calc_q,
    calc_return,
    calc_visit,
    double_backup_dp,
    expected_backup_dp,
    munchausen_backup_dp,
    optimal_backup_dp,
    soft_expected_backup_dp,
)
from ._calc.backup_rl import (
    double_backup_rl,
    expected_backup_rl,
    munchausen_backup_rl,
    optimal_backup_rl,
    soft_expected_backup_rl,
)
from ._calc.build_net import build_forward_conv, build_forward_fc, build_net_act
from ._calc.collect_samples import ACT_FN, Sample, collect_samples, make_replay_buffer
from ._calc.draw import draw_circle, line_aa
from ._calc.epsilon_greedy import calc_eps
from ._calc.loss import cross_entropy_loss, huber_loss, kl_loss, l2_loss
from ._calc.mdp import MDP
from ._calc.moving_average import calc_ma
from ._calc.sparse import SparseMat, sp_mul, sp_mul_t

# Common useful functions & classes.
from ._utils.config import Config
from ._utils.jittable import DictJittable
from ._utils.log import add_logfile_handler, initialize_log_style
from ._utils.minatar import make_minatar
from ._utils.params import ParamsDict
from ._utils.prepare_history_dir import prepare_history_dir
from ._utils.scalars import Scalars
from ._utils.tables import TbDict

# Implemented environments with access to the *oracle* quantities.
from .envs.base.config import EnvConfig
from .envs.base.env import ShinEnv
from .envs.cartpole.env import CartPole
from .envs.maze.env import Maze
from .envs.mountaincar.env import MountainCar
from .envs.pendulum.env import Pendulum

# Implemented environments with access to the *oracle* quantities.
from .solvers.base.core.config import SolverConfig
from .solvers.base.core.history import History
from .solvers.base.core.mixin import (
    GymEvalMixIn,
    GymExploreMixIn,
    ShinEvalMixIn,
    ShinExploreMixIn,
)
from .solvers.base.solver import Solver
from .solvers.pi.discrete.solver import DiscretePiSolver
from .solvers.vi.discrete.solver import DiscreteViSolver

initialize_log_style()


# ----- register envs -----


register(
    id="ShinMaze-v0",
    entry_point="shinrl:Maze",
    kwargs={"config": Maze.DefaultConfig()},
)

register(
    id="ShinPendulum-v0",
    entry_point="shinrl:Pendulum",
    kwargs={"config": Pendulum.DefaultConfig()},
)

register(
    id="ShinPendulumContinuous-v0",
    entry_point="shinrl:Pendulum",
    kwargs={"config": Pendulum.DefaultConfig(act_mode="continuous", dA=50)},
)

register(
    id="ShinPendulumImage-v0",
    entry_point="shinrl:Pendulum",
    kwargs={"config": Pendulum.DefaultConfig(obs_mode="image")},
)

register(
    id="ShinPendulumImageContinuous-v0",
    entry_point="shinrl:Pendulum",
    kwargs={
        "config": Pendulum.DefaultConfig(act_mode="continuous", dA=50, obs_mode="image")
    },
)

register(
    id="ShinMountainCar-v0",
    entry_point="shinrl:MountainCar",
    kwargs={"config": MountainCar.DefaultConfig()},
)

register(
    id="ShinMountainCarContinuous-v0",
    entry_point="shinrl:MountainCar",
    kwargs={"config": MountainCar.DefaultConfig(act_mode="continuous", dA=50)},
)

register(
    id="ShinMountainCarImage-v0",
    entry_point="shinrl:MountainCar",
    kwargs={"config": MountainCar.DefaultConfig(obs_mode="image")},
)

register(
    id="ShinMountainCarImageContinuous-v0",
    entry_point="shinrl:MountainCar",
    kwargs={
        "config": MountainCar.DefaultConfig(
            act_mode="continuous", dA=50, obs_mode="image"
        )
    },
)

register(
    id="ShinCartPole-v0",
    entry_point="shinrl:CartPole",
    kwargs={"config": CartPole.DefaultConfig()},
)


register(
    id="ShinCartPoleContinuous-v0",
    entry_point="shinrl:CartPole",
    kwargs={"config": CartPole.DefaultConfig(act_mode="continuous", dA=50)},
)
