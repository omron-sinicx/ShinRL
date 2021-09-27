from shinrl import DiscretePiSolver, Pendulum


def test_pi_tabular_dp():
    pend_env = Pendulum()
    pend_env.reset()
    config = DiscretePiSolver.DefaultConfig(
        explore="oracle",
        approx="tabular",
        eval_interval=3,
        add_interval=1,
        steps_per_epoch=10,
    )
    mixins = DiscretePiSolver.make_mixins(pend_env, config)
    solver = DiscretePiSolver.factory(pend_env, config, mixins)
    solver.run()
    assert solver.config == config
    assert solver.n_step == 10


def test_pi_deep_dp():
    pend_env = Pendulum()
    pend_env.reset()
    config = DiscretePiSolver.DefaultConfig(
        explore="oracle",
        approx="nn",
        eval_interval=3,
        add_interval=1,
        steps_per_epoch=10,
        hidden=5,
        depth=1,
    )
    mixins = DiscretePiSolver.make_mixins(pend_env, config)
    solver = DiscretePiSolver.factory(pend_env, config, mixins)
    solver.run()
    assert solver.config == config
    assert solver.n_step == 10


def test_pi_tabular_rl():
    pend_env = Pendulum()
    pend_env.reset()
    config = DiscretePiSolver.DefaultConfig(
        explore="eps_greedy",
        approx="tabular",
        eval_interval=3,
        add_interval=1,
        steps_per_epoch=10,
    )
    mixins = DiscretePiSolver.make_mixins(pend_env, config)
    solver = DiscretePiSolver.factory(pend_env, config, mixins)
    solver.run()
    assert solver.config == config
    assert solver.n_step == 10


def test_pi_deep_rl():
    pend_env = Pendulum()
    pend_env.reset()
    config = DiscretePiSolver.DefaultConfig(
        explore="eps_greedy",
        approx="nn",
        eval_interval=3,
        add_interval=1,
        steps_per_epoch=10,
    )
    mixins = DiscretePiSolver.make_mixins(pend_env, config)
    solver = DiscretePiSolver.factory(pend_env, config, mixins)
    solver.run()
    assert solver.config == config
    assert solver.n_step == 10


def test_image_input():
    pend_env = Pendulum(Pendulum.DefaultConfig(obs_mode="image"))
    pend_env.reset()
    config = DiscretePiSolver.DefaultConfig(
        explore="eps_greedy",
        approx="nn",
        eval_interval=3,
        add_interval=1,
        steps_per_epoch=10,
    )
    mixins = DiscretePiSolver.make_mixins(pend_env, config)
    solver = DiscretePiSolver.factory(pend_env, config, mixins)
    solver.run()
    assert solver.config == config
    assert solver.n_step == 10
