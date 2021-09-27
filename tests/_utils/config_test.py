import os

import chex

import shinrl as srl


@chex.dataclass
class MockConfig(srl.Config):
    test_int: int = 123
    test_str: str = "test"


def test_config(tmpdir):
    config = MockConfig()
    assert config.test_int == 123
    assert config.test_str == "test"
    cdict = config.asdict()
    assert cdict["test_int"] == 123

    # test update
    config = MockConfig()
    config.update({"test_int": 467})
    assert config.test_int == 467
    assert config.test_str == "test"
    config1 = MockConfig()
    config2 = MockConfig(test_int=89)
    config1.update(config2)
    assert config1.test_int == 89

    # save & load
    path = tmpdir.mkdir("tmp")
    config.save_as_yaml(os.path.join(path, "config.yaml"))
    new_config = config.load_from_yaml(os.path.join(path, "config.yaml"))
    assert config == new_config
