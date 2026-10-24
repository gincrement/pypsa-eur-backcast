# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Figures of the day-ahead price backcast errors over all areas, drawn by :mod:`plot_backcast_prices`.
"""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from scripts.lib.backcast_data import (
    decile_errors,
    hourly_errors,
    monthly_mae,
    top_errors,
)
from scripts.lib.backcast_plotting import (
    BAND_COLOR,
    DIVERGING,
    MODEL_COLOR,
    OBSERVED_COLOR,
    SEASONS,
    SEQUENTIAL,
    SHOWCASE_WIDTH,
    PriceInputs,
    area_heatmap,
    separate_groups,
    symmetric_limit,
)

MAP_CRS = "EPSG:3035"


def area_height(areas: int) -> float:
    """Return a figure height fitting one row per area."""
    return 0.17 * areas + 1.2


def price_scorecard(d: PriceInputs) -> Figure:
    """Plot annual mean prices, mean absolute error and Pearson correlation per area."""
    metrics = d.metrics.loc[d.buses.values].set_axis(d.buses.index)
    areas = list(metrics.index)
    y = np.arange(len(areas))
    fig, axes = plt.subplots(
        1, 3, figsize=(SHOWCASE_WIDTH, area_height(len(areas))), sharey=True
    )
    axes[0].hlines(y, metrics.mean_observed, metrics.mean_model, color=BAND_COLOR)
    axes[0].scatter(
        metrics.mean_observed, y, color=OBSERVED_COLOR, s=10, label="observed"
    )
    axes[0].scatter(metrics.mean_model, y, color=MODEL_COLOR, s=10, label="model")
    axes[0].set_xlabel("annual mean price (€/MWh)", fontsize=7)
    axes[0].legend(fontsize=6, loc="upper left", ncol=2)
    for ax, column, label in [
        (axes[1], "mae", "MAE (€/MWh)"),
        (axes[2], "pearson_r", "Pearson r"),
    ]:
        ax.barh(y, metrics[column], color=MODEL_COLOR, height=0.7)
        median = metrics[column].median()
        ax.axvline(median, color="k", ls="--", lw=0.8)
        ax.text(median, -1.2, f"median {median:.2f}", fontsize=6, ha="center")
        ax.set_xlabel(label, fontsize=7)
    axes[2].set_xlim(0, 1)
    axes[0].set_yticks(y, areas, fontsize=6)
    axes[0].set_ylim(len(areas) - 0.5, -1.8)
    for ax, title in zip(axes, "abc"):
        separate_groups(ax, areas)
        ax.tick_params(labelsize=6)
        ax.set_title(f"({title})", fontsize=8, loc="left")
    fig.tight_layout()
    return fig


def bias_heatmap(d: PriceInputs) -> Figure:
    """Plot the monthly mean bias and mean absolute error per area."""
    bias = d.monthly_bias[d.buses.values].set_axis(d.buses.index, axis=1).T
    mae = monthly_mae(d.area_error).T
    fig, axes = plt.subplots(1, 2, figsize=(12, area_height(len(bias))))
    limit = symmetric_limit(bias, 0.95)
    area_heatmap(axes[0], bias, DIVERGING, "mean bias (€/MWh)", (-limit, limit), True)
    area_heatmap(axes[1], mae, SEQUENTIAL, "MAE (€/MWh)", (0, np.nanmax(mae)), True)
    for ax, title in zip(axes, ["(a) mean bias", "(b) MAE"]):
        ax.set_xticks(
            range(len(bias.columns)), [m.strftime("%b") for m in bias.columns]
        )
        ax.tick_params(labelsize=6)
        ax.set_title(title, fontsize=8, loc="left")
    fig.tight_layout()
    return fig


def error_map(d: PriceInputs) -> Figure:
    """Map the annual mean bias and mean absolute error per bus."""
    regions = d.regions.to_crs(MAP_CRS)
    observed = regions.index.intersection(d.metrics.index)
    fig, axes = plt.subplots(1, 2, figsize=(SHOWCASE_WIDTH, 3.8))
    limit = symmetric_limit(d.metrics.mean_bias)
    for ax, column, cmap, vlim, label in [
        (axes[0], "mean_bias", DIVERGING, (-limit, limit), "mean bias (€/MWh)"),
        (axes[1], "mae", SEQUENTIAL, (0, d.metrics.mae.max()), "MAE (€/MWh)"),
    ]:
        regions.drop(observed).plot(
            ax=ax, color="0.85", hatch="////", edgecolor="0.6", lw=0.2
        )
        regions.loc[observed].assign(value=d.metrics.loc[observed, column]).plot(
            ax=ax,
            column="value",
            cmap=cmap,
            vmin=vlim[0],
            vmax=vlim[1],
            edgecolor="white",
            lw=0.3,
            legend=True,
            legend_kwds={
                "label": label,
                "orientation": "horizontal",
                "shrink": 0.8,
                "pad": 0.02,
            },
        )
        ax.set_axis_off()
        ax.set_title(f"annual {label}", fontsize=8)
    fig.tight_layout()
    return fig


def price_error_by_level(d: PriceInputs) -> Figure:
    """Plot the mean error per area by observed price decile and in the top 1 % hours."""
    deciles = decile_errors(d.area_error, d.area_observed)
    top = top_errors(d.area_error, d.area_observed, 0.01).to_frame("top 1 %")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8, area_height(len(deciles))),
        gridspec_kw={"width_ratios": [10, 2.2]},
    )
    limit = symmetric_limit(deciles, 0.95)
    area_heatmap(
        axes[0], deciles, DIVERGING, "mean error (€/MWh)", (-limit, limit), True
    )
    limit = symmetric_limit(top, 0.95)
    area_heatmap(axes[1], top, DIVERGING, "mean error (€/MWh)", (-limit, limit), True)
    axes[0].set_xlabel("observed price decile per area", fontsize=7)
    axes[1].set_yticks([])
    for ax in axes:
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


def price_error_by_hour(d: PriceInputs) -> Figure:
    """Plot the mean error per area and hour of day in winter and summer."""
    errors = {name: hourly_errors(d.area_error, m) for name, m in SEASONS.items()}
    limit = symmetric_limit(pd.concat(errors.values()), 0.95)
    fig, axes = plt.subplots(1, 2, figsize=(12, area_height(len(d.buses))), sharey=True)
    for ax, (name, error) in zip(axes, errors.items()):
        area_heatmap(ax, error, DIVERGING, "mean error (€/MWh)", (-limit, limit))
        ax.set_title(name, fontsize=8, loc="left")
        ax.set_xlabel("hour (CET)", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


ERROR_FIGURES: dict[str, Callable[[PriceInputs], Figure]] = {
    "price_scorecard": price_scorecard,
    "bias_heatmap": bias_heatmap,
    "error_map": error_map,
    "price_error_by_level": price_error_by_level,
    "price_error_by_hour": price_error_by_hour,
}
