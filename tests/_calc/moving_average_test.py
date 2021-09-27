import jax.numpy as jnp
import numpy.testing as npt

import shinrl as srl


def test_calc_ma():
    lr = 0.1
    state = jnp.array([6, 5, 2])
    act = jnp.array([1, 2, 3])
    q = jnp.zeros((10, 5))
    q_targ = jnp.ones(3)
    jnp_q = srl.calc_ma(lr, state, act, q, q_targ)

    import numpy as np

    np_q = np.zeros((10, 5))
    for ss, aa, target in zip(state, act, q_targ):
        np_q[ss, aa] = (1 - lr) * np_q[ss, aa] + lr * target

    npt.assert_allclose(jnp_q, np_q)
