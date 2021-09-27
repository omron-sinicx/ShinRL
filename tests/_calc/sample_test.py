import chex
import distrax
import gym
import jax
import jax.numpy as jnp

import shinrl as srl


def test_samples():
    t1 = jnp.array([1, 2, 3])
    t2 = jnp.array([11, 22, 33])
    samples = srl.Sample(
        obs=t1, next_obs=t1, rew=t1, done=t1, log_prob=t1, act=t1, timeout=t1
    )
    chex.assert_shape(samples.rew, (3,))

    samples = [
        srl.Sample(
            obs=t1, next_obs=t1, rew=t1, done=t1, log_prob=t1, act=t1, timeout=t1
        ),
        srl.Sample(
            obs=t2, next_obs=t2, rew=t2, done=t2, log_prob=t2, act=t2, timeout=t2
        ),
    ]


@jax.jit
def tb_act(key, state, policy):
    key, new_key = jax.random.split(key)
    dist = distrax.Categorical(probs=policy[state])
    act = dist.sample(seed=key)
    log_prob = dist.log_prob(act)
    return new_key, act, log_prob


def test_collect_samples_shin():
    key = jax.random.PRNGKey(0)
    env = srl.Pendulum()
    env.obs = env.reset()
    pol = jax.random.uniform(key, shape=(env.mdp.dS, env.mdp.dA))
    pol /= pol.sum(axis=1, keepdims=True)

    _, samples = srl.collect_samples(
        key=key,
        env=env,
        act_fn=lambda key, state: tb_act(key, state, pol),
        num_samples=10,
        use_state=True,
    )
    chex.assert_shape(samples.obs, (10, *env.observation_space.shape))
    chex.assert_shape(samples.rew, (10, 1))

    _, samples = srl.collect_samples(
        key=key,
        env=env,
        act_fn=lambda key, state: tb_act(key, state, pol),
        num_episodes=2,
        use_state=True,
    )
    assert samples.done.sum() == 2


def test_collect_samples_gym():
    key = jax.random.PRNGKey(0)
    env = gym.make("CartPole-v0")
    env.obs = env.reset()

    @jax.jit
    def act_fn(key, obs):
        key, new_key = jax.random.split(key)
        dist = distrax.Categorical(logits=jnp.ones(env.action_space.n))
        act = dist.sample(seed=key)
        log_prob = dist.log_prob(act)
        return new_key, act, log_prob

    _, samples = srl.collect_samples(key, env, act_fn, 10, use_state=False)
    chex.assert_shape(samples.obs, (10, *env.observation_space.shape))
    chex.assert_shape(samples.rew, (10, 1))

    _, samples = srl.collect_samples(key, env, act_fn, num_episodes=2, use_state=False)
    assert samples.done.sum() == 2
