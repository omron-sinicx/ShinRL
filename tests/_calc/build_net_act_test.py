import chex
import haiku as hk
import jax
import jax.numpy as jnp


def build_obs_forward_fc(n_out, last_layer=None):
    @jax.vmap
    def forward(input):
        modules = [hk.Linear(n_out)]
        if last_layer is not None:
            modules.append(last_layer)
        return hk.Sequential(modules)(input.astype(float))

    return hk.without_apply_rng(hk.transform(forward))


def test_discrete_greedy_net_act():
    from shinrl import build_discrete_greedy_net_act

    net = build_obs_forward_fc(5)
    net_act = build_discrete_greedy_net_act(net)

    key = jax.random.PRNGKey(0)
    sample_init = jnp.ones([1, 3])
    params = net.init(key, sample_init)

    obs = jnp.ones([3])
    _, act, log_p = net_act(key, obs, params)
    chex.assert_rank(act, 0)
    chex.assert_rank(log_p, 0)


def test_eps_greedy_net_act():
    from shinrl import build_eps_greedy_net_act

    net = build_obs_forward_fc(5)
    net_act = build_eps_greedy_net_act(net)

    key = jax.random.PRNGKey(0)
    sample_init = jnp.ones([1, 3])
    params = net.init(key, sample_init)

    obs = jnp.ones([3])
    _, act, log_p = net_act(key, obs, params, 1, 10, 100, 0.01)
    chex.assert_rank(act, 0)
    chex.assert_rank(log_p, 0)


def test_softmax_net_act():
    from shinrl import build_softmax_net_act

    net = build_obs_forward_fc(5)
    net_act = build_softmax_net_act(net)

    key = jax.random.PRNGKey(0)
    sample_init = jnp.ones([1, 3])
    params = net.init(key, sample_init)

    obs = jnp.ones([3])
    _, act, log_p = net_act(key, obs, params, 0.1)
    chex.assert_rank(act, 0)
    chex.assert_rank(log_p, 0)


def test_fixed_scale_normal_net_act():
    from shinrl import build_fixed_scale_normal_net_act

    net = build_obs_forward_fc(2)
    net_act = build_fixed_scale_normal_net_act(net)

    key = jax.random.PRNGKey(0)
    sample_init = jnp.ones([1, 3])
    params = net.init(key, sample_init)

    obs = jnp.ones([3])
    _, act, log_p = net_act(key, obs, params, 1.0)
    chex.assert_shape(act, (2,))
    chex.assert_shape(log_p, (2,))


def test_continuous_greedy_net_act():
    from shinrl import build_continuous_greedy_net_act

    net = build_obs_forward_fc(2)
    net_act = build_continuous_greedy_net_act(net)

    key = jax.random.PRNGKey(0)
    sample_init = jnp.ones([1, 3])
    params = net.init(key, sample_init)

    obs = jnp.ones([3])
    _, act, log_p = net_act(key, obs, params)
    chex.assert_shape(act, (2,))
    chex.assert_shape(log_p, (2,))
