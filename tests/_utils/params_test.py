import os

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax

from shinrl import ParamsDict


def test_params(tmpdir):
    mock_net = lambda input: hk.Linear(1)(input)
    net = hk.without_apply_rng(hk.transform(mock_net))
    opt = optax.adam(1e-3)
    key = jax.random.PRNGKey(0)
    sample = jnp.ones(10)
    net_params = net.init(key, sample)
    opt_state = opt.init(net_params)
    params = ParamsDict()
    params.set("params", net_params)
    params.set("state", opt_state)

    # jit
    hoge = jax.jit(lambda p: p["params"])
    chex.assert_tree_all_close(hoge(params), net_params)

    # save & load
    path = tmpdir.mkdir("tmp")
    path = os.path.join(path, "tmp.pkl")
    params.save(path)
    new_params = ParamsDict()
    new_params.load(path)

    chex.assert_tree_all_close(params["params"], new_params["params"])
    chex.assert_tree_all_close(params["state"], new_params["state"])
