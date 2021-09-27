import chex
import distrax
import jax
import jax.numpy as jnp


def test_forward_fc():
    from shinrl import build_forward_fc

    batch = 10
    n_out = 5
    depth = 2
    hidden = 32
    net = build_forward_fc(n_out, depth, hidden, jax.nn.relu)
    key = jax.random.PRNGKey(0)
    sample_init = jnp.ones([batch, 3])
    params = net.init(key, sample_init)
    output = net.apply(params, sample_init)
    chex.assert_shape(output, (batch, n_out))


def test_forward_conv():
    from shinrl import build_forward_conv

    batch = 10
    n_out = 5
    depth = 2
    hidden = 32
    net = build_forward_conv(n_out, depth, hidden, jax.nn.relu)
    key = jax.random.PRNGKey(0)
    sample_init = jnp.ones([batch, 28, 28, 1])
    params = net.init(key, sample_init)
    output = net.apply(params, sample_init)
    chex.assert_shape(output, (batch, n_out))


def test_net_act():
    from shinrl import build_forward_fc, build_net_act

    n_out = 5
    depth = 2
    hidden = 32
    net = build_forward_fc(n_out, depth, hidden, jax.nn.relu)
    key = jax.random.PRNGKey(0)
    sample_init = jnp.ones([1, 3])
    params = net.init(key, sample_init)
    Dist = distrax.Softmax
    act_fn = build_net_act(Dist, net)
    key, act, log_prob = act_fn(key, sample_init, params, temperature=1.0)
    chex.assert_rank([act, log_prob], 1)
