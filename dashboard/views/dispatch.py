# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Dispatch of a bidding zone: generation stacks, carrier energy errors and price-setting carriers.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.data import (
    aggregate,
    source_generation,
)
from dashboard.views.common import CHART_HEIGHT, Context
from scripts.lib.backcast_data import stacked_generation
from scripts.lib.backcast_plotting import (
    BAND_COLOR,
    MODEL_COLOR,
    OBSERVED_COLOR,
    STACK,
)

SOURCES = {"model": MODEL_COLOR, "entsoe": OBSERVED_COLOR}
UNMATCHED = "none"
UNMATCHED_LABEL = "no local carrier"
CLIP_QUANTILES = (0.01, 0.99)


def nice_step(span: float, ticks: int = 5) -> float:
    """Round tick step that splits the span into at most about the given number of ticks."""
    raw = span / ticks
    magnitude = 10 ** np.floor(np.log10(raw))
    return magnitude * min(m for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)


def zero_aligned(
    top: float, low: float, high: float
) -> tuple[list[float], list[float]]:
    """Generation and price axis ranges sharing the zero line, padded by five percent."""
    low, high = 1.05 * min(low, 0.0), 1.05 * max(high, 0.0)
    share = -low / (high - low)
    return [-share / (1 - share) * top, top], [low, high]


def clipped_markers(
    price: pd.Series, low: float, high: float, color: str
) -> go.Scatter:
    """Markers at the axis edge for prices outside the price axis, hovering the true value."""
    outside = (price < low) | (price > high)
    return go.Scatter(
        x=price.index[outside],
        y=price.clip(low, high)[outside],
        mode="markers",
        marker=dict(
            color=color,
            size=7,
            symbol=np.where(price[outside] > high, "triangle-up", "triangle-down"),
        ),
        customdata=price[outside],
        hovertemplate="%{customdata:.1f} €/MWh (off axis)",
        showlegend=False,
    )


def generation_stacks(ctx: Context) -> go.Figure:
    """Stacked generation of model and ENTSO-E with the price on a secondary axis clipped to its 1st and 99th percentile."""
    fig = make_subplots(
        2,
        1,
        shared_xaxes=True,
        subplot_titles=[f"{ctx.run.name} model", "ENTSO-E"],
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
    )
    prices = {"model": ctx.run.area_model, "entsoe": ctx.run.area_observed}
    marginal = ctx.run.marginal[ctx.bus].reindex(ctx.hours).astype(str)
    both = pd.concat(
        [
            aggregate(p.loc[ctx.hours, ctx.area], ctx.aggregation)
            for p in prices.values()
        ]
    )
    low, high = both.quantile(CLIP_QUANTILES)
    low, high = min(low, 0.0), max(high, 0.0)
    for row, (source, color) in enumerate(SOURCES.items(), start=1):
        stack = stacked_generation(
            source_generation(ctx.run.generation, source, ctx.area), ctx.hours
        )
        stack = aggregate(stack, ctx.aggregation)
        for carrier in STACK:
            if stack[carrier].any():
                fig.add_scatter(
                    x=stack.index,
                    y=stack[carrier],
                    name=carrier,
                    stackgroup=source,
                    line=dict(width=0, color=ctx.colors[carrier]),
                    legendgroup=carrier,
                    showlegend=row == 1,
                    row=row,
                    col=1,
                )
        price = aggregate(prices[source].loc[ctx.hours, ctx.area], ctx.aggregation)
        hover = marginal if source == "model" and ctx.aggregation == "hourly" else None
        fig.add_scatter(
            x=price.index,
            y=price,
            name=f"price {source}",
            line=dict(color=color, width=1),
            customdata=hover,
            hovertemplate="%{y:.1f} €/MWh"
            + (" (%{customdata})" if hover is not None else ""),
            row=row,
            col=1,
            secondary_y=True,
        )
        fig.add_trace(
            clipped_markers(price, low, high, color), row=row, col=1, secondary_y=True
        )
        gw, price = zero_aligned(max(stack.sum(axis=1).max(), 1.0), low, high)
        fig.update_yaxes(
            title_text="GW",
            range=gw,
            tickvals=np.arange(0, gw[1], nice_step(gw[1])),
            row=row,
            col=1,
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text="€/MWh",
            range=price,
            tickmode="linear",
            tick0=0,
            dtick=nice_step(price[1] - price[0]),
            showgrid=False,
            row=row,
            col=1,
            secondary_y=True,
        )
    fig.update_layout(height=640, hovermode="x unified", margin=dict(t=40, b=30))
    return fig


def carrier_bars(table: pd.DataFrame) -> go.Figure:
    """Bars of the energy per carrier of model and ENTSO-E."""
    fig = go.Figure(
        [
            go.Bar(x=table.index, y=table[source], name=source, marker_color=color)
            for source, color in SOURCES.items()
        ]
    )
    fig.update_layout(
        barmode="group",
        yaxis_title="TWh",
        height=CHART_HEIGHT,
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return fig


def marginal_bars(shares: pd.Series, colors: dict[str, str]) -> go.Figure:
    """Horizontal bars of the share of hours per price-setting carrier."""
    fig = go.Figure(
        go.Bar(
            x=100 * shares,
            y=shares.index,
            orientation="h",
            marker_color=[colors.get(c, BAND_COLOR) for c in shares.index],
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(xaxis_title="share of hours (%)", height=CHART_HEIGHT)
    return fig


def dispatch_lead(table: pd.DataFrame, shares: pd.Series) -> str:
    """Summarise the price-setting carrier and the largest energy gap of the area."""
    setters = shares.drop(UNMATCHED, errors="ignore").head(2)
    named = " and ".join(f"**{c}** ({v:.0%})" for c, v in setters.items())
    text = f"The model price is set mostly by {named} of hours."
    if table.empty:
        return text
    gap = table.error.abs().idxmax()
    return (
        f"{text} Largest energy gap: **{gap}**, model {table.model[gap]:.1f} vs "
        f"ENTSO-E {table.entsoe[gap]:.1f} TWh."
    )


def dispatch_sections(ctx: Context, table: pd.DataFrame, shares: pd.Series) -> None:
    """Draw the generation stacks, energy per carrier and price-setting carrier."""
    st.subheader("Generation and price")
    st.caption(f"GW per carrier and price in €/MWh, {ctx.period}, {ctx.aggregation}.")
    st.plotly_chart(generation_stacks(ctx), width="stretch")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Energy per carrier")
        st.plotly_chart(carrier_bars(table), width="stretch")
    with right:
        st.subheader("Price-setting carrier (model)")
        st.plotly_chart(
            marginal_bars(shares.rename({UNMATCHED: UNMATCHED_LABEL}), ctx.colors),
            width="stretch",
        )
