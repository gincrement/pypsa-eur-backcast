# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
P4 dispatch and exchange: generation stacks, carrier errors, border positions, marginal carriers and hydro.
"""

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
from dashboard.views.common import Context
from scripts.lib.backcast_data import stacked_generation
from scripts.lib.backcast_plotting import (
    MODEL_COLOR,
    OBSERVED_COLOR,
    STACK,
)

SOURCES = {"model": MODEL_COLOR, "entsoe": OBSERVED_COLOR}


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
    fig.update_yaxes(title_text="GW", secondary_y=False)
    both = pd.concat([p.loc[ctx.hours, ctx.area] for p in prices.values()])
    span = [both.min() - 10, both.max() + 10]
    fig.update_yaxes(title_text="€/MWh", range=span, secondary_y=True)
    fig.update_layout(height=700, hovermode="x unified")
    return fig


def carrier_bars(table: pd.DataFrame) -> go.Figure:
    """Bars of the energy per carrier of model and ENTSO-E."""
    fig = go.Figure(
        [
            go.Bar(x=table.index, y=table[source], name=source, marker_color=color)
            for source, color in SOURCES.items()
        ]
    )
    fig.update_layout(barmode="group", yaxis_title="TWh", height=380)
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
    fig.update_layout(height=160 * len(neighbours) + 80, hovermode="x unified")
    return fig


def render(ctx: Context) -> None:
    """Draw the dispatch and exchange page."""
    if ctx.aggregation == "hourly" and len(ctx.hours) > 24 * 62:
        st.info("Hourly stacks of long periods are slow, consider daily aggregation")
    st.plotly_chart(generation_stacks(ctx), width="stretch")
    table = carrier_errors(ctx.run.generation, ctx.area, ctx.hours)
    left, right = st.columns([3, 2])
    left.plotly_chart(carrier_bars(table), width="stretch")
    right.dataframe(table.style.format("{:.2f}"))
    st.subheader("net export per border (MW)")
    positions = border_positions(ctx.run.flows, ctx.area, ctx.hours)
    if positions.empty:
        st.info(f"No modelled borders of {ctx.area}")
    else:
        st.plotly_chart(border_figure(positions, ctx.aggregation), width="stretch")
    left, right = st.columns(2)
    left.subheader("price-setting carrier (model)")
    left.dataframe(
        marginal_shares(ctx.run.marginal[ctx.bus].loc[ctx.hours])
        .to_frame()
        .style.format("{:.1%}")
    )
    observed = ctx.run.area_observed.loc[ctx.hours, ctx.area]
    histogram = go.Figure(
        go.Histogram(x=observed, marker_color=OBSERVED_COLOR, nbinsx=80)
    )
    histogram.update_layout(
        xaxis_title="observed €/MWh", yaxis_title="hours", height=380
    )
    right.subheader("observed price distribution")
    right.plotly_chart(histogram, width="stretch")
    hydro = ctx.run.hydro["state_of_charge"]
    if ctx.area in hydro:
        st.subheader("hydro reservoir state of charge (model, TWh)")
        level = aggregate(hydro[ctx.area].loc[ctx.hours] / 1e6, ctx.aggregation)
        figure = go.Figure(
            go.Scatter(x=level.index, y=level, line=dict(color=MODEL_COLOR))
        )
        figure.update_layout(height=320, yaxis_title="TWh")
        st.plotly_chart(figure, width="stretch")
