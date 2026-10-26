# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Plots the benchmark figures of the modelled against the observed day-ahead prices.

The showcase figures compare hourly prices in two selected weeks, daily means
over the year, price duration curves, intraday profiles and hourly densities
for the showcase zones in ``SHOWCASE_BUSES``. The figures over all areas, see
:mod:`plot_backcast_errors`, show a scorecard of annual metrics, the monthly
bias, an error map and the price error by price level and hour of day. Each
area with observed prices is represented by its first observed bus. Every
figure is written as PDF and PNG.
"""

from collections.abc import Callable

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from scripts._helpers import configure_logging, set_scenario_config
from scripts.lib.backcast_data import (
    CLIP_PRICE_RANGE,
    ZERO_PRICE_TOLERANCE,
    align_prices,
    duration_curve,
    share,
)
from scripts.lib.backcast_plotting import (
    BAND_COLOR,
    DENSITY,
    MODEL_COLOR,
    OBSERVED_COLOR,
    SEASONS,
    SHOWCASE_BUSES,
    SHOWCASE_WIDTH,
    PriceInputs,
    cet_hour,
    representative_buses,
    save_figures,
)
from scripts.lib.entsoe import read_area_to_bus
from scripts.plot_backcast_errors import ERROR_FIGURES

HIGH_PRICE = 200.0
SEASON_STYLES = ["-", "--"]


def fit_text(model: pd.Series, observed: pd.Series) -> str:
    """Format the mean absolute error and Pearson correlation of two price series."""
    mae = (model - observed).abs().mean()
    return f"MAE {mae:.1f} €/MWh, r {model.corr(observed):.2f}"


def annotate(ax: Axes, text: str, loc: tuple[float, float] = (0.02, 0.95)) -> None:
    """Place a small text box in axes coordinates."""
    ax.text(
        *loc,
        text,
        transform=ax.transAxes,
        va="top",
        fontsize=6,
        bbox={"facecolor": "white", "alpha": 0.8, "lw": 0},
    )


def showcase_grid(
    rows: int, cols: int, height: float, **kwargs: bool | str
) -> tuple[Figure, np.ndarray]:
    """Create a grid of axes for the showcase zones."""
    return plt.subplots(
        rows, cols, figsize=(SHOWCASE_WIDTH, height), squeeze=False, **kwargs
    )


def series_legend(fig: Figure, band: bool = False) -> None:
    """Add the observed and model legend at the top of the figure."""
    handles = [
        plt.Line2D([], [], color=OBSERVED_COLOR, label="observed"),
        plt.Line2D([], [], color=MODEL_COLOR, label="model"),
    ]
    if band:
        handles.append(
            plt.Rectangle(
                (0, 0), 1, 1, color=BAND_COLOR, alpha=0.5, label="observed daily range"
            )
        )
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=7)


def price_weeks(d: PriceInputs) -> Figure:
    """Plot hourly prices of the showcase zones in the winter and summer week."""
    fig, axes = showcase_grid(len(SHOWCASE_BUSES), 2, 9.5, sharey="row")
    for row, bus in zip(axes, SHOWCASE_BUSES):
        for ax, (name, week) in zip(row, d.weeks.items()):
            model, observed = d.model.loc[week, bus], d.observed.loc[week, bus]
            ax.plot(week, observed, color=OBSERVED_COLOR, lw=0.8)
            ax.plot(week, model, color=MODEL_COLOR, lw=0.8)
            ax.set_title(fit_text(model, observed), fontsize=6, loc="right")
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
            ax.tick_params(labelsize=6)
        row[0].set_ylabel(f"{d.label(bus)}\n€/MWh", fontsize=7)
    for ax, (name, week) in zip(axes[0], d.weeks.items()):
        ax.set_title(f"{name} week from {week[0]:%d %b %Y}", fontsize=7, loc="left")
    for ax in axes[-1]:
        ax.set_xlabel("day of month (UTC)", fontsize=7)
    series_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def price_daily_year(d: PriceInputs) -> Figure:
    """Plot daily mean prices and the observed daily range over the year."""
    fig, axes = showcase_grid(len(SHOWCASE_BUSES), 1, 9.5, sharex=True)
    for ax, bus in zip(axes[:, 0], SHOWCASE_BUSES):
        daily = d.observed[bus].resample("D")
        ax.fill_between(
            daily.min().index,
            daily.min(),
            daily.max(),
            color=BAND_COLOR,
            alpha=0.5,
            lw=0,
        )
        ax.plot(daily.mean(), color=OBSERVED_COLOR, lw=0.8)
        ax.plot(d.model[bus].resample("D").mean(), color=MODEL_COLOR, lw=0.8)
        ax.set_title(fit_text(d.model[bus], d.observed[bus]), fontsize=6, loc="right")
        ax.set_ylabel(f"{d.label(bus)}\n€/MWh", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    series_legend(fig, band=True)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def price_duration(d: PriceInputs) -> Figure:
    """Plot price duration curves clipped to the plot range with the shares of non-positive and high price hours."""
    fig, axes = showcase_grid(2, 3, 5.0, sharex=True)
    for ax, bus in zip(axes.flat, SHOWCASE_BUSES):
        lines = []
        for name, prices, color in [
            ("obs", d.observed[bus], OBSERVED_COLOR),
            ("mod", d.model[bus], MODEL_COLOR),
        ]:
            curve = duration_curve(prices).clip(*CLIP_PRICE_RANGE)
            ax.plot(curve.index, curve.values, color=color, lw=1)
            low = share(prices <= ZERO_PRICE_TOLERANCE, prices)
            high = share(prices > HIGH_PRICE, prices)
            lines.append(f"{name}: ≤0 {low:.1%}, >200 {high:.1%}")
        annotate(ax, "\n".join(lines), (0.3, 0.95))
        ax.set_ylim(CLIP_PRICE_RANGE)
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_title(d.label(bus), fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes[:, 0]:
        ax.set_ylabel("€/MWh", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("share of hours", fontsize=7)
    series_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def price_profiles(d: PriceInputs) -> Figure:
    """Plot mean intraday price profiles in winter and summer."""
    fig, axes = showcase_grid(2, 3, 5.0, sharex=True)
    for ax, bus in zip(axes.flat, SHOWCASE_BUSES):
        for style, months in zip(SEASON_STYLES, SEASONS.values()):
            for prices, color in [(d.observed, OBSERVED_COLOR), (d.model, MODEL_COLOR)]:
                selected = prices.loc[prices.index.month.isin(months), bus]
                profile = selected.groupby(cet_hour(selected.index)).mean()
                ax.plot(profile.index, profile.values, color=color, ls=style, lw=1)
        ax.set_title(d.label(bus), fontsize=8)
        ax.set_xticks(range(0, 24, 6))
        ax.tick_params(labelsize=6)
    for ax in axes[:, 0]:
        ax.set_ylabel("€/MWh", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("hour (CET)", fontsize=7)
    handles = [
        plt.Line2D([], [], color=OBSERVED_COLOR, label="observed"),
        plt.Line2D([], [], color=MODEL_COLOR, label="model"),
        *[
            plt.Line2D([], [], color="0.4", ls=style, label=season)
            for style, season in zip(SEASON_STYLES, SEASONS)
        ],
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def price_density(d: PriceInputs) -> Figure:
    """Plot hexbin densities of modelled over observed hourly prices for the showcase zones and all areas pooled."""
    fig, axes = showcase_grid(2, 4, 4.4, sharex=True, sharey=True)
    pooled = (
        d.area_model.stack(future_stack=True),
        d.area_observed.stack(future_stack=True),
    )
    panels = {d.label(bus): (d.model[bus], d.observed[bus]) for bus in SHOWCASE_BUSES}
    panels["all areas pooled"] = pooled
    for ax, (title, (model, observed)) in zip(axes.flat, panels.items()):
        valid = observed.notna() & model.notna()
        model, observed = model[valid], observed[valid]
        image = ax.hexbin(
            observed.clip(*CLIP_PRICE_RANGE),
            model.clip(*CLIP_PRICE_RANGE),
            gridsize=40,
            extent=(*CLIP_PRICE_RANGE, *CLIP_PRICE_RANGE),
            bins="log",
            cmap=DENSITY,
            mincnt=1,
            linewidths=0,
        )
        ax.plot(CLIP_PRICE_RANGE, CLIP_PRICE_RANGE, color="k", lw=0.6)
        annotate(ax, f"{fit_text(model, observed)}\nn {valid.sum()}")
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes[:, 0]:
        ax.set_ylabel("model €/MWh", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("observed €/MWh", fontsize=7)
    axes.flat[-1].set_axis_off()
    fig.colorbar(image, ax=axes.flat[-1], label="hours (log)", fraction=0.5)
    fig.tight_layout()
    return fig


FIGURES: dict[str, Callable[[PriceInputs], Figure]] = {
    "price_weeks": price_weeks,
    "price_daily_year": price_daily_year,
    "price_duration": price_duration,
    "price_profiles": price_profiles,
    "price_density": price_density,
    **ERROR_FIGURES,
}


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("plot_backcast_prices")
    configure_logging(snakemake)
    set_scenario_config(snakemake)
    plt.style.use(["default", snakemake.input.rc])

    n = pypsa.Network(snakemake.input.network)
    model = n.buses_t.marginal_price[n.buses.query("carrier == 'AC'").index]
    observed = pd.read_csv(snakemake.input.prices, index_col=0, parse_dates=True)
    model, observed = align_prices(model, observed)
    area_to_bus = read_area_to_bus(snakemake.input.area_to_bus)
    monthly_bias = pd.read_csv(snakemake.input.monthly_bias, index_col=0)
    inputs = PriceInputs(
        model=model,
        observed=observed,
        area_to_bus=area_to_bus,
        buses=representative_buses(area_to_bus, observed.columns),
        metrics=pd.read_csv(snakemake.input.metrics, index_col=0),
        monthly_bias=monthly_bias.set_axis(
            pd.PeriodIndex(monthly_bias.index, freq="M")
        ),
        regions=gpd.read_file(snakemake.input.regions).set_index("name"),
    )
    save_figures(FIGURES, inputs, snakemake.output)
