# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Dispatch and flows: generation stacks, carrier energy errors, price-setting carriers, border positions and hydro.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.data import (
    aggregate,
    border_positions,
    carrier_errors,
    marginal_shares,
    source_generation,
)
from dashboard.views.common import CHART_HEIGHT, Context, takeaway
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


def generation_stacks(ctx: Context) -> go.Figure:
    """Stacked generation of model and ENTSO-E with the price on a secondary axis."""
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
        gw, price = zero_aligned(
            max(stack.sum(axis=1).max(), 1.0), both.min(), both.max()
        )
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


def net_exports(positions: pd.DataFrame) -> pd.DataFrame:
    """Net export in TWh per neighbour for model and scheduled."""
    energy = positions.sum().unstack("series") / 1e6
    return energy.reindex(columns=["model", "scheduled"])


def export_bars(energy: pd.DataFrame) -> go.Figure:
    """Bars of the net export per neighbour of model and scheduled."""
    fig = go.Figure(
        [
            go.Bar(x=energy.index, y=energy[source], name=source, marker_color=color)
            for source, color in [("model", MODEL_COLOR), ("scheduled", OBSERVED_COLOR)]
        ]
    )
    fig.update_layout(
        barmode="group",
        yaxis_title="net export (TWh)",
        height=CHART_HEIGHT,
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return fig


def border_figure(positions: pd.DataFrame, aggregation: str) -> go.Figure:
    """Export of the area per neighbour for model and scheduled, hovering the NTC."""
    neighbours = list(positions.columns.unique("neighbour"))
    fig = make_subplots(
        len(neighbours), 1, shared_xaxes=True, subplot_titles=neighbours
    )
    for row, neighbour in enumerate(neighbours, start=1):
        frame = aggregate(positions[neighbour], aggregation)
        ntc = frame.reindex(columns=["ntc out", "ntc in"])
        for source, color in [("model", MODEL_COLOR), ("scheduled", OBSERVED_COLOR)]:
            if source in frame:
                fig.add_scatter(
                    x=frame.index,
                    y=frame[source],
                    name=source,
                    line=dict(color=color, width=1),
                    customdata=ntc.to_numpy(),
                    hovertemplate="%{y:.0f} MW (NTC out %{customdata[0]:.0f}, in %{customdata[1]:.0f})",
                    legendgroup=source,
                    showlegend=row == 1,
                    row=row,
                    col=1,
                )
    fig.update_layout(
        height=160 * len(neighbours) + 80, hovermode="x unified", yaxis_title="MW"
    )
    return fig


def lead(
    ctx: Context, table: pd.DataFrame, shares: pd.Series, energy: pd.DataFrame
) -> str:
    """Summarise the price-setting carrier, the largest energy gap and the net export of the area."""
    setters = shares.drop(UNMATCHED, errors="ignore").head(2)
    named = " and ".join(f"**{c}** ({v:.0%})" for c, v in setters.items())
    text = f"In {ctx.period}, the model price of **{ctx.area}** is set mostly by {named} of hours."
    if not table.empty:
        gap = table.error.abs().idxmax()
        text += (
            f" Largest energy gap: **{gap}**, model {table.model[gap]:.1f} vs ENTSO-E "
            f"{table.entsoe[gap]:.1f} TWh."
        )
    if energy.empty:
        return text
    total = energy.sum()
    return (
        f"{text} Net export: model **{total.model:+.1f} TWh**, scheduled "
        f"{total.scheduled:+.1f} TWh."
    )


def render(ctx: Context) -> None:
    """Draw the dispatch and flows page."""
    table = carrier_errors(ctx.run.generation, ctx.area, ctx.hours)
    shares = marginal_shares(ctx.run.marginal[ctx.bus].loc[ctx.hours])
    positions = border_positions(ctx.run.flows, ctx.area, ctx.hours)
    energy = net_exports(positions) if not positions.empty else pd.DataFrame()
    takeaway(lead(ctx, table, shares, energy))
    st.subheader("Generation and price")
    st.caption(f"GW per carrier and price in €/MWh, {ctx.period}, {ctx.aggregation}.")
    st.plotly_chart(generation_stacks(ctx), width="stretch")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Energy per carrier")
        st.plotly_chart(carrier_bars(table), width="stretch")
        with st.expander("Energy table (TWh)"):
            st.dataframe(table.style.format("{:.2f}"))
    with right:
        st.subheader("Price-setting carrier (model)")
        st.plotly_chart(
            marginal_bars(shares.rename({UNMATCHED: UNMATCHED_LABEL}), ctx.colors),
            width="stretch",
        )
    st.subheader("Cross-border exchange")
    if positions.empty:
        st.info(f"No modelled borders of {ctx.area}")
    else:
        left, right = st.columns([2, 3])
        left.plotly_chart(export_bars(energy), width="stretch")
        gaps = (energy.model - energy.scheduled.fillna(0.0)).abs()
        neighbours = right.multiselect(
            "Hourly export per border (MW), largest gaps first",
            list(gaps.sort_values(ascending=False).index),
            default=list(gaps.nlargest(3).index),
            key=f"borders_{ctx.area}",
        )
        if neighbours:
            right.plotly_chart(
                border_figure(positions[neighbours], ctx.aggregation), width="stretch"
            )
    hydro = ctx.run.hydro["state_of_charge"]
    if ctx.area in hydro:
        st.subheader("Hydro reservoir state of charge (model)")
        level = aggregate(hydro[ctx.area].loc[ctx.hours] / 1e6, ctx.aggregation)
        figure = go.Figure(
            go.Scatter(x=level.index, y=level, line=dict(color=MODEL_COLOR))
        )
        figure.update_layout(height=320, yaxis_title="TWh")
        st.plotly_chart(figure, width="stretch")
