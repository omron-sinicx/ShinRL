from gym.envs.registration import register

register(
    id="ShinMaze-v0",
    entry_point="shinrl.envs:Maze",
    kwargs={
        "spec": None,
        "horizon": 30,
        "maze_size": 5,
        "trans_eps": 0.0,
        "obs_mode": "random",
        "obs_dim": 10,
    },
)

register(
    id="ShinPendulum-v0",
    entry_point="shinrl.envs:Pendulum",
    kwargs={
        "horizon": 200,
        "action_mode": "discrete",
        "dA": 5,
        "state_disc": 32,
        "obs_mode": "tuple",
    },
)

register(
    id="ShinPendulumContinuous-v0",
    entry_point="shinrl.envs:Pendulum",
    kwargs={
        "horizon": 200,
        "action_mode": "continuous",
        "dA": 50,
        "state_disc": 32,
        "obs_mode": "tuple",
    },
)

register(
    id="ShinPendulumImage-v0",
    entry_point="shinrl.envs:Pendulum",
    kwargs={
        "horizon": 200,
        "action_mode": "discrete",
        "dA": 5,
        "state_disc": 32,
        "obs_mode": "image",
    },
)

register(
    id="ShinPendulumImageContinuous-v0",
    entry_point="shinrl.envs:Pendulum",
    kwargs={
        "horizon": 200,
        "action_mode": "continuous",
        "dA": 50,
        "state_disc": 32,
        "obs_mode": "image",
    },
)

register(
    id="ShinMountainCar-v0",
    entry_point="shinrl.envs:MountainCar",
    kwargs={
        "horizon": 200,
        "action_mode": "discrete",
        "dA": 3,
        "state_disc": 32,
        "obs_mode": "tuple",
    },
)

register(
    id="ShinMountainCarContinuous-v0",
    entry_point="shinrl.envs:MountainCar",
    kwargs={
        "horizon": 200,
        "action_mode": "continuous",
        "dA": 50,
        "state_disc": 32,
        "obs_mode": "tuple",
    },
)

register(
    id="ShinMountainCarImage-v0",
    entry_point="shinrl.envs:MountainCar",
    kwargs={
        "horizon": 200,
        "action_mode": "discrete",
        "dA": 3,
        "state_disc": 32,
        "obs_mode": "image",
    },
)

register(
    id="ShinMountainCarImageContinuous-v0",
    entry_point="shinrl.envs:MountainCar",
    kwargs={
        "horizon": 200,
        "action_mode": "continuous",
        "dA": 50,
        "state_disc": 32,
        "obs_mode": "image",
    },
)

register(
    id="ShinCartPole-v0",
    entry_point="shinrl.envs:CartPole",
    kwargs={
        "horizon": 100,
        "action_mode": "discrete",
        "dA": 5,
        "x_disc": 128,
        "x_dot_disc": 4,
        "th_disc": 64,
        "th_dot_disc": 4,
        "obs_mode": "tuple",
    },
)
