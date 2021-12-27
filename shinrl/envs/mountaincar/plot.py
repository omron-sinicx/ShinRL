import functools
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from chex import Array
from matplotlib.axes import Axes

from .calc import *
from .config import MountainCarConfig


@jax.jit
def disc_pos_vel(
    config: MountainCarConfig, pos: float, vel: float
) -> Tuple[float, float]:
    pos_step = (config.pos_max - config.pos_min) / (config.pos_res - 1)
    vel_step = (config.vel_max - config.vel_min) / (config.pos_res - 1)
    pos_idx = jnp.floor((pos - config.pos_min) / pos_step + 1e-5).astype(jnp.uint32)
    pos_idx = jnp.clip(pos_idx, 0, config.pos_res - 1)
    vel_idx = jnp.floor((vel - config.vel_min) / vel_step + 1e-5).astype(jnp.uint32)
    vel_idx = jnp.clip(vel_idx, 0, config.vel_res - 1)
    return pos_idx, vel_idx


@functools.partial(jax.vmap, in_axes=(None, 1, 1), out_axes=0)
def undisc_pos_vel(
    config: MountainCarConfig, pos_round: float, pos_vel: float
) -> Tuple[float, float]:
    pos_step = (config.pos_max - config.pos_min) / (config.pos_res - 1)
    vel_step = (config.vel_max - config.vel_min) / (config.pos_res - 1)
    pos = pos_round * pos_step + config.pos_min
    pos = jnp.clip(pos, config.pos_min, config.pos_max)
    vel = pos_vel * vel_step + config.vel_min
    vel = jnp.clip(vel, config.vel_min, config.vel_max)
    return pos, vel


@functools.partial(jax.vmap, in_axes=(None, 1), out_axes=0)
def disc(config, s):
    pos, vel = state_to_pos_vel(config, s)
    pos, vel = disc_pos_vel(config, pos, vel)
    return pos, vel


def plot_S(
    tb: Array,
    config: MountainCarConfig,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    cbar_ax: Optional[Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    fontsize: Optional[int] = 10,
    **kwargs: Any,
) -> None:
    assert len(tb.shape) == 1
    reshaped_values = np.empty((config.pos_res, config.vel_res))
    reshaped_values[:] = np.nan
    ss = jnp.arange(tb.shape[0])[:, None]
    pos, vel = disc(config, ss)
    reshaped_values[pos, vel] = tb

    if ax is None:
        grid_kws = {"width_ratios": (0.95, 0.05)}
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6), gridspec_kw=grid_kws)
        ax, cbar_ax = axes[0], axes[1]
    vmin = tb.min() if vmin is None else vmin
    vmax = tb.max() if vmax is None else vmax

    pos_ticks, vel_ticks = [], []
    ii = jnp.arange(config.pos_res)[:, None]
    pos_ticks, vel_ticks = undisc_pos_vel(config, ii, ii)
    pos_ticks = pos_ticks.reshape(-1).tolist()
    pos_ticks = [round(pos, 3) for pos in pos_ticks]
    vel_ticks = vel_ticks.reshape(-1).tolist()
    vel_ticks = [round(vel, 3) for vel in vel_ticks]

    data = pd.DataFrame(reshaped_values, index=pos_ticks, columns=vel_ticks).T
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
    ax.set_xlabel("Position", fontsize=fontsize)
    ax.set_ylabel("Velocity", fontsize=fontsize)
