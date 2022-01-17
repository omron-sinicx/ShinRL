"""
Author: Toshinori Kitamura
Affiliation: NAIST & OSX
"""
import gym
import jax
import numpy as np

import shinrl as srl


class MockActMixIn:
    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        env = self.env
        if self.is_shin_env:
            key = jax.random.PRNGKey(0)
            pol = jax.random.uniform(key, shape=(env.mdp.dS, env.mdp.dA))
            pol /= pol.sum(axis=1, keepdims=True)
            self.data.update({"EvaluatePolicy": pol, "ExplorePolicy": pol})

    def eval_act(self, key, obs):
        act = self.env.action_space.sample()
        log_prob = 0.0
        return key, act, log_prob

    def explore_act(self, key, obs):
        act = self.env.action_space.sample()
        log_prob = 0.0
        return key, act, log_prob

    def step(self):
        return {}


def test_gym_eval():
    class MockSolver(srl.BaseSolver):
        @staticmethod
        def make_mixins():
            return [MockActMixIn, srl.BaseGymEvalMixIn, MockSolver]

    env = srl.NormalizeActionWrapper(gym.make("Pendulum-v0"))
    eval_env = srl.NormalizeActionWrapper(gym.make("Pendulum-v0"))
    config = MockSolver.DefaultConfig()
    mixins = MockSolver.make_mixins()
    solver = MockSolver.factory(env, config, mixins)
    solver.set_eval_env(eval_env)
    res = solver.evaluate()
    assert np.isscalar(res["Return"])


def test_gym_explore():
    class MockSolver(srl.BaseSolver):
        @staticmethod
        def make_mixins():
            return [MockActMixIn, srl.BaseGymExploreMixIn, MockSolver]

        def evaluate():
            pass

    env = srl.NormalizeActionWrapper(gym.make("Pendulum-v0"))
    config = MockSolver.DefaultConfig()
    config.num_samples = 10
    mixins = MockSolver.make_mixins()
    solver = MockSolver.factory(env, config, mixins)
    samples = solver.explore()
    assert isinstance(samples, srl.Sample)


def test_shin_eval():
    class MockSolver(srl.BaseSolver):
        @staticmethod
        def make_mixins():
            return [MockActMixIn, srl.BaseShinEvalMixIn, MockSolver]

    env = gym.make("ShinPendulum-v0")
    config = MockSolver.DefaultConfig()
    mixins = MockSolver.make_mixins()
    solver = MockSolver.factory(env, config, mixins)
    res = solver.evaluate()
    assert np.isscalar(res["Return"])


def test_shin_explore():
    class MockSolver(srl.BaseSolver):
        @staticmethod
        def make_mixins():
            return [MockActMixIn, srl.BaseShinExploreMixIn, MockSolver]

        def evaluate():
            pass

    env = gym.make("ShinPendulum-v0")
    config = MockSolver.DefaultConfig()
    config.num_samples = 10
    mixins = MockSolver.make_mixins()
    solver = MockSolver.factory(env, config, mixins)
    samples = solver.explore()
    assert isinstance(samples, srl.Sample)
