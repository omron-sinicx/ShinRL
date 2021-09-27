import jax

import shinrl as srl


def test_l2_loss():
    pred = jax.random.normal(jax.random.PRNGKey(0), shape=(10, 2))
    targ = jax.random.normal(jax.random.PRNGKey(0), shape=(10, 2))
    srl.l2_loss(pred, targ)


def test_huber_loss():
    pred = jax.random.normal(jax.random.PRNGKey(0), shape=(10, 2))
    targ = jax.random.normal(jax.random.PRNGKey(0), shape=(10, 2))
    srl.huber_loss(pred, targ)


def test_cross_entropy_loss():
    logits = jax.random.normal(jax.random.PRNGKey(0), shape=(10, 2))
    targ_logits = jax.random.normal(jax.random.PRNGKey(0), shape=(10, 2))
    srl.cross_entropy_loss(logits, targ_logits)


def test_kl_loss():
    logits = jax.random.normal(jax.random.PRNGKey(0), shape=(10, 2))
    targ_logits = jax.random.normal(jax.random.PRNGKey(0), shape=(10, 2))
    srl.kl_loss(logits, targ_logits)
