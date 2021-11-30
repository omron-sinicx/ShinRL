import os
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pytest

from shinrl import utils


@dataclass
class MockConfig(utils.Config):
    test_int: int = 123
    test_str: str = "test"


def test_config():
    config = MockConfig()
    assert config.test_int == 123
    assert config.test_str == "test"
    cdict = config.asdict()
    assert cdict["test_int"] == 123


def test_config_update():
    config = MockConfig()
    config.update({"test_int": 467})
    assert config.test_int == 467
    assert config.test_str == "test"
    config1 = MockConfig()
    config2 = MockConfig(test_int=89)
    config1.update(config2)
    assert config1.test_int == 89


def test_config_save_load(tmpdir):
    path = tmpdir.mkdir("tmp")
    config = MockConfig()
    config.save_as_yaml(os.path.join(path, "config.yaml"))
    new_config = config.load_from_yaml(os.path.join(path, "config.yaml"))
    assert config == new_config


def test_history():
    history = utils.History(add_interval=10)
    for i in range(30):
        history.add_scalar("test1", i)
        history.add_scalar("test2", 10 * i)
        history.step += 1
    history.set_tb("test_array", np.array([1, 2, 3]))
    assert history.scalars["test1"] == {"x": [0, 10, 20], "y": [0, 10, 20]}
    assert history.scalars["test2"] == {"x": [0, 10, 20], "y": [0, 100, 200]}
    assert np.all(history.tbs["test_array"] == np.array([1, 2, 3]))
    assert history.step == 30


def test_history_save_load(tmpdir):
    history = utils.History(add_interval=10)
    for i in range(30):
        history.add_scalar("test1", i)
        history.add_scalar("test2", 10 * i)
        history.step += 1
    history.epoch += 1
    path = tmpdir.mkdir("tmp")
    scalars = deepcopy(history.scalars)
    history.save_all(path)
    assert history.scalars == {}

    new_history = utils.History()
    new_history.load_all(path)
    assert new_history.scalars == scalars
    new_history.scalars = {}
    assert history == new_history
