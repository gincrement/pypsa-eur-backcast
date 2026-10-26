# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Trade between countries: net and hourly exchange of one area per neighbour, model against ENTSO-E scheduled.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.data import aggregate, border_positions
from dashboard.views.common import CHART_HEIGHT, Context, takeaway
from scripts.lib.backcast_plotting import MODEL_COLOR, OBSERVED_COLOR

SOURCES = {"model": MODEL_COLOR, "scheduled": OBSERVED_COLOR}


def net_exports(positions: pd.DataFrame) -> pd.DataFrame:
    """Net export in TWh per neighbour for model and scheduled."""
    energy = positions.sum().unstack("series") / 1e6
    return energy.reindex(columns=["model", "scheduled"])


def export_bars(energy: pd.DataFrame) -> go.Figure:
    """Bars of the net export per neighbour of model and scheduled."""
    fig = go.Figure(
        [
            go.Bar(x=energy.index, y=energy[source], name=source, marker_color=color)
            for source, color in SOURCES.items()
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
        for source, color in SOURCES.items():
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


def lead(ctx: Context, energy: pd.DataFrame, gaps: pd.Series) -> str:
    """Summarise the net export of the area and its largest border gap."""
    total = energy.sum()
    gap = gaps.idxmax()
    return (
        f"In {ctx.period}, **{ctx.area}** net exports **{total.model:+.1f} TWh** in the "
        f"model against {total.scheduled:+.1f} TWh scheduled. Largest border gap: "
        f"**{gap}**, model {energy.model[gap]:+.1f} vs scheduled "
        f"{energy.scheduled[gap]:+.1f} TWh."
    )


def render(ctx: Context) -> None:
    """Draw the cross border flows page."""
    positions = border_positions(ctx.run.flows, ctx.area, ctx.hours)
    if positions.empty:
        st.info(f"No modelled borders of {ctx.area}")
        return
    energy = net_exports(positions)
    gaps = (energy.model - energy.scheduled.fillna(0.0)).abs()
    takeaway(lead(ctx, energy, gaps))
    left, right = st.columns([2, 3])
    left.plotly_chart(export_bars(energy), width="stretch")
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
