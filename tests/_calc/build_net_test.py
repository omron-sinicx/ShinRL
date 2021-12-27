import chex
import jax
import jax.numpy as jnp


def test_obs_forward_fc():
    from shinrl import build_obs_forward_fc

    batch = 10
    n_out = 5
    depth = 2
    hidden = 32
    net = build_obs_forward_fc(n_out, depth, hidden, jax.nn.relu)
    key = jax.random.PRNGKey(0)
    sample_init = jnp.ones([batch, 3])
    params = net.init(key, sample_init)
    output = net.apply(params, sample_init)
    chex.assert_shape(output, (batch, n_out))


def test_obs_forward_conv():
    from shinrl import build_obs_forward_conv

    batch = 10
    n_out = 5
    depth = 2
    hidden = 32
    net = build_obs_forward_conv(n_out, depth, hidden, jax.nn.relu)
    key = jax.random.PRNGKey(0)
    sample_init = jnp.ones([batch, 28, 28, 1])
    params = net.init(key, sample_init)
    output = net.apply(params, sample_init)
    chex.assert_shape(output, (batch, n_out))


def test_obs_act_forward_fc():
    from shinrl import build_obs_act_forward_fc

    batch = 10
    n_out = 5
    depth = 2
    hidden = 32
    net = build_obs_act_forward_fc(n_out, depth, hidden, jax.nn.relu)
    key = jax.random.PRNGKey(0)
    sample_obs = jnp.ones([batch, 3])
    sample_act = jnp.ones([batch, 2])
    params = net.init(key, sample_obs, sample_act)
    output = net.apply(params, sample_obs, sample_act)
    chex.assert_shape(output, (batch, n_out))


def test_obs_act_forward_conv():
    from shinrl import build_obs_act_forward_conv

    batch = 10
    n_out = 5
    depth = 2
    hidden = 32
    net = build_obs_act_forward_conv(n_out, depth, hidden, jax.nn.relu)
    key = jax.random.PRNGKey(0)
    sample_obs = jnp.ones([batch, 28, 28, 1])
    sample_act = jnp.ones([batch, 2])
    params = net.init(key, sample_obs, sample_act)
    output = net.apply(params, sample_obs, sample_act)
    chex.assert_shape(output, (batch, n_out))
