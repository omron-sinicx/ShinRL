import os

import jax
import jax.numpy as jnp

from shinrl import TbDict


def test_tables(tmpdir):
    tables = TbDict()
    tables.set("test", jnp.ones((10, 10)))
    assert jnp.all(tables["test"] == jnp.ones((10, 10)))

    x = 1
    test_map = lambda: jnp.ones((15, 15)) * x
    tables.set("test_map", test_map)
    assert jnp.all(tables["test_map"] == jnp.ones((15, 15)))
    x = 2
    assert jnp.all(tables["test_map"] == jnp.ones((15, 15)) * 2)

    # jit
    hoge = jax.jit(lambda tb: tb["test_map"])
    assert jnp.all(hoge(tables) == jnp.ones((15, 15)) * 2)

    # save & load
    path = tmpdir.mkdir("tmp")
    path = os.path.join(path, "tmp.pkl")
    tables.save(path)
    new_tables = TbDict()
    new_tables.load(path)
    assert jnp.all(tables["test"] == new_tables["test"])
