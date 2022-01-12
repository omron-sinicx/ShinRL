import shinrl as srl
import jax.numpy as jnp
import jax
import chex


def test_squashed_normal():
    loc = jnp.array([0.1, 0.5, 0.2])
    scale = jnp.array([1.0, 5.0, 10.0])
    dist = srl.SquashedNormal(loc, scale)
    sample = dist.sample(seed=jax.random.PRNGKey(0), sample_shape=10)
    log_prob = dist.log_prob(sample)
    chex.assert_shape(log_prob, (10, 3))
