from shinrl import ContinuousDdpgSolver, Pendulum


def test_ddpg_deep_dp():
    pend_env = Pendulum(Pendulum.DefaultConfig(act_mode="continuous"))
    pend_env.reset()
    config = ContinuousDdpgSolver.DefaultConfig(
        explore="oracle",
        approx="nn",
        eval_interval=3,
        add_interval=1,
        steps_per_epoch=10,
        hidden=5,
        depth=1,
    )
    mixins = ContinuousDdpgSolver.make_mixins(pend_env, config)
    solver = ContinuousDdpgSolver.factory(pend_env, config, mixins)
    solver.run()
    assert solver.config == config
    assert solver.n_step == 10


def test_ddpg_deep_rl():
    pend_env = Pendulum(Pendulum.DefaultConfig(act_mode="continuous"))
    pend_env.reset()
    config = ContinuousDdpgSolver.DefaultConfig(
        explore="normal",
        approx="nn",
        eval_interval=3,
        add_interval=1,
        steps_per_epoch=10,
    )
    mixins = ContinuousDdpgSolver.make_mixins(pend_env, config)
    solver = ContinuousDdpgSolver.factory(pend_env, config, mixins)
    solver.run()
    assert solver.config == config
    assert solver.n_step == 10


def test_image_input():
    pend_env = Pendulum(Pendulum.DefaultConfig(act_mode="continuous", obs_mode="image"))
    pend_env.reset()
    config = ContinuousDdpgSolver.DefaultConfig(
        explore="normal",
        approx="nn",
        eval_interval=3,
        add_interval=1,
        steps_per_epoch=10,
    )
    mixins = ContinuousDdpgSolver.make_mixins(pend_env, config)
    solver = ContinuousDdpgSolver.factory(pend_env, config, mixins)
    solver.run()
    assert solver.config == config
    assert solver.n_step == 10
