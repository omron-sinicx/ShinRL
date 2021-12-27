import chex
import jax.numpy as jnp

import shinrl as srl


def test_build_table_mixin():
    from shinrl.solvers.discrete_pi._build_table_mixin import BuildTableMixIn
    from shinrl.solvers.discrete_pi.config import PiConfig

    class MockSolver(srl.BaseSolver):
        @staticmethod
        def make_mixins():
            return [BuildTableMixIn, MockSolver]

        def step(self):
            pass

        def evaluate(self):
            pass

    env = srl.Pendulum()
    config = PiConfig()
    mixins = MockSolver.make_mixins()
    solver = MockSolver.factory(env, config, mixins)
    chex.assert_rank(solver.data["Q"], 2)
    chex.assert_rank(solver.data["Logits"], 2)
    chex.assert_rank(solver.data["ExplorePolicy"], 2)
    chex.assert_rank(solver.data["EvaluatePolicy"], 2)


def test_build_net_mixin():
    from shinrl.solvers.discrete_pi._build_net_mixin import BuildNetMixIn
    from shinrl.solvers.discrete_pi.config import PiConfig

    class MockSolver(srl.BaseSolver):
        @staticmethod
        def make_mixins():
            return [BuildNetMixIn, MockSolver]

        def step(self):
            pass

        def evaluate(self):
            pass

    env = srl.Pendulum()
    config = PiConfig()
    mixins = MockSolver.make_mixins()
    solver = MockSolver.factory(env, config, mixins)
    obs = jnp.expand_dims(env.observation_space.sample(), axis=0)
    output = solver.q_net.apply(solver.data["QNetParams"], obs)
    chex.assert_shape(output, (1, env.dA))


def test_target_mixin():
    from shinrl.solvers.discrete_pi._build_table_mixin import BuildTableMixIn
    from shinrl.solvers.discrete_pi._target_mixin import QTargetMixIn
    from shinrl.solvers.discrete_pi.config import PiConfig

    class MockSolver(srl.BaseSolver):
        @staticmethod
        def make_mixins():
            return [QTargetMixIn, BuildTableMixIn, MockSolver]

        def step(self):
            pass

        def evaluate(self):
            pass

    env = srl.Pendulum()
    config = PiConfig()
    mixins = MockSolver.make_mixins()
    solver = MockSolver.factory(env, config, mixins)
    pol_dist = solver.target_pol_dist(solver.data["Q"])
    q_targ = solver.target_q_tabular_dp(solver.data, pol_dist)
    chex.assert_shape(q_targ, (env.dS, env.dA))


def test_net_act_mixin():
    from shinrl.solvers.discrete_pi._build_net_mixin import BuildNetMixIn
    from shinrl.solvers.discrete_pi._net_act_mixin import NetActMixIn
    from shinrl.solvers.discrete_pi.config import PiConfig

    class MockSolver(srl.BaseSolver):
        @staticmethod
        def make_mixins():
            return [NetActMixIn, BuildNetMixIn, MockSolver]

        def step(self):
            pass

        def evaluate(self):
            pass

    env = srl.Pendulum()
    config = PiConfig()
    mixins = MockSolver.make_mixins()
    solver = MockSolver.factory(env, config, mixins)
    obs = env.observation_space.sample()
    _, act, log_p = solver.explore_act(solver.key, obs)
    chex.assert_rank(act, 0)
    chex.assert_rank(log_p, 0)
