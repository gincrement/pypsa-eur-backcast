# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Bidding zone: prices and dispatch of one area in the period against ENTSO-E.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.data import (
    aggregate,
    carrier_errors,
    marginal_shares,
    month_hour_errors,
    tail_shares,
    worst_hours,
)
from dashboard.views.common import (
    CHART_HEIGHT,
    MONTHS,
    Context,
    Line,
    event_markers,
    heatmap,
    price_traces,
    takeaway,
)
from dashboard.views.dispatch import dispatch_lead, dispatch_sections
from scripts.lib.backcast_data import duration_curve
from scripts.lib.backcast_plotting import (
    DENSITY,
    MODEL_COLOR,
    OBSERVED_COLOR,
    SEASONS,
    cet_hour,
)

HISTOGRAM_BINS = 80
CONTEXT = pd.Timedelta(days=1)


def hourly_line(ctx: Context) -> go.Figure:
    """Model and observed prices of the period with one day of context on each side."""
    window = slice(ctx.hours[0] - CONTEXT, ctx.hours[-1] + CONTEXT)
    model = ctx.run.area_model.loc[window, ctx.area]
    observed = ctx.run.area_observed.loc[window, ctx.area]
    series = [aggregate(s, ctx.aggregation) for s in (model, observed)]
    fig = go.Figure(price_traces(ctx, *series))
    if ctx.aggregation == "hourly":
        shedding = ctx.run.load["load_shedding"].reindex(
            columns=[ctx.bus], fill_value=0.0
        )
        fig.add_traces(event_markers(model, shedding[ctx.bus]))
    fig.update_layout(
        height=CHART_HEIGHT,
        hovermode="x unified",
        yaxis_title="€/MWh",
        margin=dict(t=30, b=30),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return fig


def period_lines(ctx: Context) -> list[Line]:
    """Observed and model prices of the area in the period."""
    return [
        Line(
            "observed", ctx.run.area_observed.loc[ctx.hours, ctx.area], OBSERVED_COLOR
        ),
        Line(ctx.run.name, ctx.run.area_model.loc[ctx.hours, ctx.area], MODEL_COLOR),
    ]


def duration_figure(lines: list[Line]) -> go.Figure:
    """Price duration curves."""
    fig = go.Figure()
    for line in lines:
        curve = duration_curve(line.prices)
        fig.add_scatter(
            x=curve.index,
            y=curve,
            name=line.name,
            line=dict(color=line.color),
        )
    fig.update_layout(
        xaxis_title="share of hours", yaxis_title="€/MWh", height=CHART_HEIGHT
    )
    return fig


def duration(lines: list[Line]) -> None:
    """Price duration curves with tail shares."""
    left, right = st.columns([3, 1])
    left.plotly_chart(duration_figure(lines), width="stretch")
    shares = pd.DataFrame({line.name: tail_shares(line.prices) for line in lines})
    right.markdown("**Share of hours by price (€/MWh)**")
    right.dataframe(shares.style.format("{:.1%}"))


def profile_figures(lines: list[Line]) -> list[go.Figure]:
    """Mean price by CET hour in winter and summer."""
    figures = []
    for season, months in SEASONS.items():
        fig = go.Figure()
        for line in lines:
            selected = line.prices[line.prices.index.month.isin(months)]
            profile = selected.groupby(cet_hour(selected.index)).mean()
            fig.add_scatter(
                x=profile.index,
                y=profile,
                name=line.name,
                line=dict(color=line.color),
            )
        fig.update_layout(
            title=season,
            xaxis_title="hour (CET)",
            yaxis_title="€/MWh",
            height=CHART_HEIGHT,
        )
        figures.append(fig)
    return figures


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


def lead(ctx: Context, lines: list[Line]) -> str:
    """Summarise the price fit of the area in the period, its rank among all areas and the worst hours."""
    metrics = ctx.metrics()
    row = metrics.loc[ctx.area]
    rank = int(metrics.mae.rank()[ctx.area])
    observed, model = lines[0].prices, lines[1].prices
    error = model - observed
    first, last, peak = worst_hours(error.groupby(cet_hour(error.index)).mean())
    return (
        f"In {ctx.period}, **{ctx.area}** has MAE **{row.mae:.1f} €/MWh** "
        f"(rank {rank} of {len(metrics)}, 1 = best), Pearson r **{row.pearson_r:.2f}** "
        f"and bias **{row.mean_bias:+.1f} €/MWh**; worst hours (CET) **{first}:00 to "
        f"{last + 1}:00** ({peak:+.0f} €/MWh). Mean price: observed {observed.mean():.1f}, "
        f"model {model.mean():.1f} €/MWh. Hours at or below 0 €/MWh: observed "
        f"{(observed <= 0).mean():.1%}, model {(model <= 0).mean():.1%}."
    )


def render(ctx: Context) -> None:
    """Draw the bidding zone page."""
    lines = period_lines(ctx)
    table = carrier_errors(ctx.run.generation, ctx.area, ctx.hours)
    shares = marginal_shares(ctx.run.marginal[ctx.bus].loc[ctx.hours])
    takeaway(f"{lead(ctx, lines)}\n\n{dispatch_lead(table, shares)}")
    st.subheader("Price")
    st.plotly_chart(hourly_line(ctx), width="stretch")
    st.caption("One day of context on each side of the period.")
    tabs = st.tabs(
        [
            "Duration curve",
            "Error by month and hour",
            "Daily profile",
            "Model vs observed",
        ]
    )
    with tabs[0]:
        duration(lines)
    with tabs[1]:
        st.caption("Mean error model minus observed, €/MWh, whole year.")
        error = ctx.run.area_model[ctx.area] - ctx.run.area_observed[ctx.area]
        errors = month_hour_errors(error, cet_hour(error.index))
        errors = errors.set_axis([MONTHS[m - 1] for m in errors.index])
        st.plotly_chart(heatmap(errors, "€/MWh"), width="stretch")
    with tabs[2]:
        for column, figure in zip(st.columns(len(SEASONS)), profile_figures(lines)):
            column.plotly_chart(figure, width="stretch")
    with tabs[3]:
        st.plotly_chart(density(lines[1].prices, lines[0].prices))
    dispatch_sections(ctx, table, shares)
