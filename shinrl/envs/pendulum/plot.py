import functools
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from chex import Array
from matplotlib.axes import Axes

from .calc import *
from .config import PendulumConfig


@jax.jit
def disc_th_vel(config: PendulumConfig, th: float, vel: float) -> Tuple[float, float]:
    th = normalize_angle(th)
    th_step = (2 * jnp.pi) / (config.theta_res - 1)
    vel_step = (2 * config.vel_max) / (config.theta_res - 1)
    th_round = jnp.floor((th + jnp.pi) / th_step + 1e-5).astype(jnp.uint32)
    vel_round = jnp.floor((vel + config.vel_max) / vel_step + 1e-5).astype(jnp.uint32)
    return th_round, vel_round


@functools.partial(jax.vmap, in_axes=(None, 1, 1), out_axes=0)
def undisc_th_vel(
    config: PendulumConfig, th_round: float, th_vel: float
) -> Tuple[float, float]:
    th_step = (2 * jnp.pi) / (config.theta_res - 1)
    vel_step = (2 * config.vel_max) / (config.theta_res - 1)
    th = th_round * th_step - jnp.pi
    th = normalize_angle(th)
    vel = th_vel * vel_step - config.vel_max
    return th, vel


@functools.partial(jax.vmap, in_axes=(None, 1), out_axes=0)
def disc(config, s):
    th, vel = state_to_th_vel(config, s)
    th, vel = disc_th_vel(config, th, vel)
    return th, vel


def plot_S(
    tb: Array,
    config: PendulumConfig,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    cbar_ax: Optional[Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    fontsize: Optional[int] = 10,
    **kwargs: Any,
) -> None:
    assert len(tb.shape) == 1
    reshaped_values = np.empty((config.theta_res, config.vel_res))
    reshaped_values[:] = np.nan
    ss = jnp.arange(tb.shape[0])[:, None]
    th, vel = disc(config, ss)
    reshaped_values[th, vel] = tb

    if ax is None:
        grid_kws = {"width_ratios": (0.95, 0.05)}
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6), gridspec_kw=grid_kws)
        ax, cbar_ax = axes[0], axes[1]
    vmin = tb.min() if vmin is None else vmin
    vmax = tb.max() if vmax is None else vmax

    th_ticks, vel_ticks = [], []
    ii = jnp.arange(config.theta_res)[:, None]
    th_ticks, vel_ticks = undisc_th_vel(config, ii, ii)
    th_ticks = th_ticks.reshape(-1).tolist()
    th_ticks = [round(th, 3) for th in th_ticks]
    vel_ticks = vel_ticks.reshape(-1).tolist()
    vel_ticks = [round(vel, 3) for vel in vel_ticks]

    data = pd.DataFrame(reshaped_values, index=th_ticks, columns=vel_ticks).T
    data = data.ffill(axis=0)
    data = data.ffill(axis=1)
    sns.heatmap(
        data,
        ax=ax,
        cbar=cbar_ax is not None,
        cbar_ax=cbar_ax,
        vmin=vmin,
        vmax=vmax,
        **kwargs,
    )
    ax.set_title(title, fontsize=fontsize)
    ax.set_ylabel("Velocity", fontsize=fontsize)
    ax.set_xlabel("Angle", fontsize=fontsize)
