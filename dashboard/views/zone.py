# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Zone detail: prices of one area in the period with duration, daily profile, density and month by hour error.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.data import aggregate, month_hour_errors, tail_shares
from dashboard.views.common import (
    CHART_HEIGHT,
    MONTHS,
    Context,
    event_markers,
    heatmap,
    price_traces,
    takeaway,
)
from scripts.evaluate_price_backcast import duration_curve
from scripts.lib.backcast_plotting import (
    DENSITY,
    MODEL_COLOR,
    OBSERVED_COLOR,
    SEASONS,
    SECOND_COLOR,
    cet_hour,
)

HISTOGRAM_BINS = 80


def hourly_line(ctx: Context) -> go.Figure:
    """Model, observed and comparison prices of the year zoomed to the period."""
    model = ctx.run.area_model[ctx.area]
    observed = ctx.run.area_observed[ctx.area]
    second = ctx.compare.area_model(ctx.run.buses)[ctx.area] if ctx.same_year else None
    series = [aggregate(s, ctx.aggregation) for s in (model, observed)]
    second = None if second is None else aggregate(second, ctx.aggregation)
    fig = go.Figure(price_traces(ctx, *series, second))
    if ctx.aggregation == "hourly":
        shedding = ctx.run.load["load_shedding"].reindex(
            columns=[ctx.bus], fill_value=0.0
        )
        shedding = shedding[ctx.bus]
        fig.add_traces(event_markers(model, shedding))
    start, end = ctx.hours[[0, -1]]
    fig.update_xaxes(range=[start, end])
    fig.update_layout(
        height=CHART_HEIGHT,
        hovermode="x unified",
        yaxis_title="€/MWh",
        margin=dict(t=30, b=30),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return fig


def period_series(ctx: Context) -> list[tuple[str, pd.Series, str]]:
    """Observed, model and same-year comparison prices of the area in the period."""
    series = [
        ("observed", ctx.run.area_observed.loc[ctx.hours, ctx.area], OBSERVED_COLOR),
        (ctx.run.name, ctx.run.area_model.loc[ctx.hours, ctx.area], MODEL_COLOR),
    ]
    if ctx.same_year:
        second = ctx.compare.area_model(ctx.run.buses).loc[ctx.hours, ctx.area]
        series.append((ctx.compare.name, second, SECOND_COLOR))
    return series


def duration(series: list[tuple[str, pd.Series, str]]) -> None:
    """Price duration curves with tail shares."""
    fig = go.Figure()
    for name, prices, color in series:
        curve = duration_curve(prices)
        fig.add_scatter(x=curve.index, y=curve, name=name, line=dict(color=color))
    fig.update_layout(
        xaxis_title="share of hours", yaxis_title="€/MWh", height=CHART_HEIGHT
    )
    left, right = st.columns([3, 1])
    left.plotly_chart(fig, width="stretch")
    shares = pd.DataFrame({name: tail_shares(prices) for name, prices, _ in series})
    right.markdown("**Share of hours by price (€/MWh)**")
    right.dataframe(shares.style.format("{:.1%}"))


def profiles(series: list[tuple[str, pd.Series, str]]) -> None:
    """Mean price by CET hour in winter and summer."""
    columns = st.columns(len(SEASONS))
    for column, (season, months) in zip(columns, SEASONS.items()):
        fig = go.Figure()
        for name, prices, color in series:
            selected = prices[prices.index.month.isin(months)]
            profile = selected.groupby(cet_hour(selected.index)).mean()
            fig.add_scatter(
                x=profile.index, y=profile, name=name, line=dict(color=color)
            )
        fig.update_layout(
            title=season,
            xaxis_title="hour (CET)",
            yaxis_title="€/MWh",
            height=CHART_HEIGHT,
        )
        column.plotly_chart(fig, width="stretch")


def density(model: pd.Series, observed: pd.Series) -> go.Figure:
    """2D histogram of model over observed prices with the identity line."""
    bounds = [np.nanmin([observed, model]), np.nanmax([observed, model])]
    fig = go.Figure(
        go.Histogram2d(
            x=observed,
            y=model,
            nbinsx=HISTOGRAM_BINS,
            nbinsy=HISTOGRAM_BINS,
            colorscale=DENSITY.capitalize(),
        )
    )
    fig.add_scatter(
        x=bounds, y=bounds, mode="lines", line=dict(color="black"), name="identity"
    )
    fig.update_layout(
        xaxis_title="observed (€/MWh)",
        yaxis_title="model (€/MWh)",
        height=CHART_HEIGHT + 80,
        width=CHART_HEIGHT + 180,
    )
    return fig


def lead(ctx: Context, series: list[tuple[str, pd.Series, str]]) -> str:
    """Summarise the price fit of the area in the period and its rank among all areas."""
    metrics = ctx.metrics()
    row = metrics.loc[ctx.area]
    rank = int(metrics.mae.rank()[ctx.area])
    (_, observed, _), (_, model, _) = series[:2]
    return (
        f"In {ctx.period}, **{ctx.area}** has MAE **{row.mae:.1f} €/MWh** "
        f"(rank {rank} of {len(metrics)}, 1 = best), Pearson r **{row.pearson_r:.2f}** "
        f"and bias **{row.mean_bias:+.1f} €/MWh**. Mean price: observed {observed.mean():.1f}, "
        f"model {model.mean():.1f} €/MWh. Hours at or below 0 €/MWh: observed "
        f"{(observed <= 0).mean():.1%}, model {(model <= 0).mean():.1%}."
    )


def render(ctx: Context) -> None:
    """Draw the zone detail page."""
    series = period_series(ctx)
    takeaway(lead(ctx, series))
    st.plotly_chart(hourly_line(ctx), width="stretch")
    if ctx.compare is not None and not ctx.same_year:
        st.caption(f"{ctx.compare.name} covers another year and is not overlaid.")
    tabs = st.tabs(
        [
            "Duration curve",
            "Daily profile",
            "Model vs observed",
            "Error by month and hour",
        ]
    )
    with tabs[0]:
        duration(series)
    with tabs[1]:
        profiles(series)
    with tabs[2]:
        st.plotly_chart(density(series[1][1], series[0][1]))
    with tabs[3]:
        st.caption("Mean error model minus observed, €/MWh, whole year.")
        error = ctx.run.area_model[ctx.area] - ctx.run.area_observed[ctx.area]
        table = month_hour_errors(error, cet_hour(error.index))
        table = table.set_axis([MONTHS[m - 1] for m in table.index])
        st.plotly_chart(heatmap(table, "€/MWh"), width="stretch")
