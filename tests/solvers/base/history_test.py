from copy import deepcopy

import chex
import haiku as hk
import jax
import jax.numpy as jnp

from shinrl import History


def test_history(tmpdir):
    history = History()
    history.config.add_interval = 10
    for i in range(30):
        history.add_scalar("test1", i)
        history.add_scalar("test2", 10 * i)
        history.n_step += 1
    assert history.scalars["test1"] == {"x": [0, 10, 20], "y": [0, 10, 20]}
    assert history.scalars["test2"] == {"x": [0, 10, 20], "y": [0, 100, 200]}
    assert history.n_step == 30

    history.data["test_tb"] = jnp.ones(12)
    mock_net = lambda input: hk.Linear(1)(input)
    net = hk.without_apply_rng(hk.transform(mock_net))
    key = jax.random.PRNGKey(0)
    sample = jnp.ones(10)
    net_params = net.init(key, sample)
    history.data["test_param"] = net_params

    # save & load
    history.n_epoch += 1
    path = tmpdir.mkdir("tmp")
    scalars = deepcopy(history.scalars)
    history.save(path)
    assert history.scalars == {}

    new_history = History()
    new_history.load(path)
    assert new_history.scalars == scalars
    assert jnp.all(new_history.data["test_tb"] == jnp.ones(12))
    chex.assert_tree_all_close(
        history.data["test_param"], new_history.data["test_param"]
    )
