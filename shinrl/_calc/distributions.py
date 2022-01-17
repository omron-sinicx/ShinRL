""" Distrax distributions. """


from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from distrax import Distribution, Normal
from distrax._src.distributions.distribution import PRNGKey

Numeric = chex.Numeric
Array = chex.Array
PRNGKey = chex.PRNGKey


class SquashedNormal(Distribution):
    def __init__(self, loc: Numeric, scale: Numeric):
        """
        A Squashed-Gaussian distribution. This assumes its space as [-1, 1].
        Often used in the SAC algorithm (See Appendix C in https://arxiv.org/abs/1801.01290).
        """
        super().__init__()
        self._unsquashed_dist = Normal(loc, scale)

    def log_prob(self, value: Array) -> Array:
        log_prob = self._unsquashed_dist.log_prob(value).sum(axis=-1, keepdims=True)
        log_prob -= 2 * (jnp.log(2) - value - jax.nn.softplus(-2 * value))
        return log_prob

    def _sample_n(self, key: PRNGKey, n: int) -> Array:
        unsquashed = self._unsquashed_dist._sample_n(key, n)
        return jnp.tanh(unsquashed)

    def event_shape(self) -> Tuple[int, ...]:
        return ()
