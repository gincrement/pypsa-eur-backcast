# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Overview: headline metrics against the validation year, metric map and per-zone price, MAE and correlation.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.data import KPIS, kpis
from dashboard.state import follow_selection
from dashboard.views.common import (
    Context,
    show,
    takeaway,
    worst,
    zone_height,
    zone_layout,
)
from scripts.evaluate_price_backcast import price_metrics
from scripts.lib.backcast_plotting import (
    BAND_COLOR,
    DIVERGING,
    MODEL_COLOR,
    OBSERVED_COLOR,
    SECOND_COLOR,
    SEQUENTIAL,
    symmetric_limit,
)

LOWER_IS_BETTER = {
    "median MAE (€/MWh)",
    "median MAE / mean price",
    "median |bias| (€/MWh)",
}
PERCENT = {"median MAE / mean price"}
MAP_METRICS = {
    "MAE (€/MWh)": "mae",
    "bias (€/MWh)": "mean_bias",
    "Pearson r": "pearson_r",
    "std ratio": "std_ratio",
}
DEFAULT_MAP = "MAE (€/MWh)"
DIVERGING_METRICS = {"mean_bias"}
TABLE_COLUMNS = {
    "mae": "MAE",
    "mean_bias": "bias",
    "rmse": "RMSE",
    "pearson_r": "r",
    "std_ratio": "std ratio",
    "mean_observed": "mean observed",
    "mean_model": "mean model",
    "nonpositive_observed": "hours ≤ 0 observed",
    "nonpositive_model": "hours ≤ 0 model",
}


def formatted(name: str, value: float) -> str:
    """Format a headline number."""
    return f"{value:.0%}" if name in PERCENT else f"{value:.2f}"


def kpi_row(label: str, metrics: pd.DataFrame, reference: pd.DataFrame | None) -> None:
    """Show one row of headline numbers with the change against a same-year reference."""
    values = kpis(metrics)
    deltas = None if reference is None else kpis(reference)
    columns = st.columns([1.4, *[1] * len(KPIS)])
    columns[0].markdown(f"**{label}**")
    for column, name in zip(columns[1:], KPIS):
        delta = None if deltas is None else values[name] - deltas[name]
        column.metric(
            name,
            formatted(name, values[name]),
            None if delta is None else f"{formatted(name, delta)} vs B",
            delta_color="inverse" if name in LOWER_IS_BETTER else "normal",
        )


def summary(values: dict[str, float]) -> str:
    """Headline numbers as a sentence fragment."""
    return (
        f"MAE **{values['median MAE (€/MWh)']:.2f} €/MWh** "
        f"({values['median MAE / mean price']:.0%} of the mean price), "
        f"r **{values['median Pearson r']:.2f}**, |bias| {values['median |bias| (€/MWh)']:.2f} €/MWh"
    )


def lead(ctx: Context, buses: pd.DataFrame, metrics: pd.DataFrame) -> str:
    """Summarise the quality of run A, the validation runs over the comparable period and the worst zones."""
    text = (
        f"Median over **{len(buses)} model buses** ({len(metrics)} bidding zones), "
        f"{ctx.period}: {summary(kpis(buses))}."
    )
    for ref in ctx.references:
        text += f" Validation, {ref.period}: {summary(kpis(ref.bus_metrics()))}."
    return (
        f"{text}\n\nLargest MAE: {worst(metrics.mae)} €/MWh. "
        f"Largest bias: {worst(metrics.mean_bias, signed=True)} €/MWh. "
        f"Hours at or below 0 €/MWh: observed {metrics.nonpositive_observed.mean():.1%}, "
        f"model {metrics.nonpositive_model.mean():.1%} (mean over zones)."
    )


def zone_map(ctx: Context, column: str, height: int) -> go.Figure:
    """Choropleth of a bus metric over the bus regions, carrying the area of each bus."""
    values = price_metrics(
        ctx.run.model.loc[ctx.hours], ctx.run.observed.loc[ctx.hours]
    )[column]
    limit = symmetric_limit(values)
    scale = (
        dict(colorscale=DIVERGING, zmid=0, zmin=-limit, zmax=limit)
        if column in DIVERGING_METRICS
        else dict(colorscale=SEQUENTIAL.capitalize())
    )
    fig = go.Figure(
        go.Choropleth(
            geojson=ctx.run.regions,
            featureidkey="properties.name",
            locations=values.index,
            z=values.to_numpy(),
            customdata=ctx.run.area_to_bus[values.index].to_numpy(),
            hovertemplate="%{customdata} (%{location}): %{z:.2f}<extra></extra>",
            marker_line_width=0.3,
            colorbar=dict(orientation="h", thickness=10, len=0.7, y=0, yanchor="top"),
            **scale,
        )
    )
    fig.update_geos(
        visible=False,
        projection_type="mercator",
        lonaxis_range=[-10.5, 31.5],
        lataxis_range=[35.5, 71],
    )
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=40), clickmode="event+select"
    )
    return fig


def bars(metrics: pd.DataFrame, second: pd.DataFrame | None, label: str) -> go.Figure:
    """Dumbbell of mean prices and bars of MAE and Pearson r per area in zone order."""
    areas = list(metrics.index)
    fig = make_subplots(
        1,
        3,
        shared_yaxes=True,
        horizontal_spacing=0.03,
        subplot_titles=["mean price (€/MWh)", "MAE (€/MWh)", "Pearson r"],
    )
    for area, row in metrics.iterrows():
        fig.add_scatter(
            x=[row.mean_observed, row.mean_model],
            y=[area, area],
            mode="lines",
            line=dict(color=BAND_COLOR),
            showlegend=False,
            hoverinfo="skip",
            row=1,
            col=1,
        )
    for column, color, name in [
        ("mean_observed", OBSERVED_COLOR, "observed"),
        ("mean_model", MODEL_COLOR, label),
    ]:
        fig.add_scatter(
            x=metrics[column],
            y=areas,
            mode="markers",
            marker=dict(color=color),
            name=name,
            customdata=areas,
            row=1,
            col=1,
        )
    series = {label: (metrics, MODEL_COLOR)}
    if second is not None:
        series["B"] = (second.reindex(metrics.index), SECOND_COLOR)
    for col, column in [(2, "mae"), (3, "pearson_r")]:
        for name, (frame, color) in series.items():
            fig.add_bar(
                x=frame[column],
                y=areas,
                orientation="h",
                marker_color=color,
                name=name,
                customdata=areas,
                showlegend=col == 2 and name == "B",
                row=1,
                col=col,
            )
    fig.update_layout(
        barmode="group",
        clickmode="event+select",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0),
    )
    zone_layout(fig, areas)
    fig.update_layout(margin=dict(t=90))
    return fig


def render(ctx: Context) -> None:
    """Draw the overview page, a click on a zone selects it for the detail pages."""
    metrics = ctx.metrics()
    buses = ctx.bus_metrics()
    second = None if ctx.compare is None else ctx.compare_metrics()
    takeaway(lead(ctx, buses, metrics))
    st.caption("Headline numbers: median over model buses, as in price_metrics.csv.")
    kpi_row(
        f"{ctx.run.name}, {ctx.period}",
        buses,
        ctx.compare_bus_metrics() if ctx.same_year else None,
    )
    for ref in ctx.references:
        kpi_row(f"validation {ref.run.name}, {ref.period}", ref.bus_metrics(), None)
    left, right = st.columns(2)
    with left:
        name = st.segmented_control(
            "Map metric", list(MAP_METRICS), default=DEFAULT_MAP, key="map_metric"
        )
        figure = zone_map(
            ctx, MAP_METRICS[name or DEFAULT_MAP], zone_height(len(metrics))
        )
        points = show(figure, "zone_map", select=True)
    with right:
        points += show(
            bars(metrics, second, ctx.run.name), "overview_bars", select=True
        )
    if points:
        area = points[0]["customdata"]
        follow_selection("overview", (area,), area=area)
    table = metrics[list(TABLE_COLUMNS)].rename(columns=TABLE_COLUMNS)
    shares = [c for c in table if "≤" in c]
    table[shares] *= 100
    with st.expander("Metric table per zone (€/MWh, shares in %)"):
        st.dataframe(table.style.format("{:.2f}"))
