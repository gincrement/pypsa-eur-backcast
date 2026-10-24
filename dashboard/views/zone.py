# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
P2 zone explorer: metric map, hourly prices and duration, profile, density and error heatmap of one area.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.data import aggregate, month_hour_errors, tail_shares
from dashboard.state import follow_selection
from dashboard.views.common import (
    Context,
    event_markers,
    heatmap,
    price_traces,
    show,
)
from scripts.evaluate_price_backcast import duration_curve, price_metrics
from scripts.lib.backcast_plotting import (
    DENSITY,
    DIVERGING,
    MODEL_COLOR,
    OBSERVED_COLOR,
    SEASONS,
    SECOND_COLOR,
    cet_hour,
    symmetric_limit,
)

MAP_METRICS = {
    "bias": "mean_bias",
    "MAE": "mae",
    "r": "pearson_r",
    "std ratio": "std_ratio",
    "Δ MAE vs B": "delta_mae",
}
DIVERGING_METRICS = {"mean_bias", "delta_mae"}
HISTOGRAM_BINS = 80


def bus_metrics(ctx: Context) -> pd.DataFrame:
    """Price metrics per bus in the period with the MAE change against a same-year comparison."""
    observed = ctx.run.observed.loc[ctx.hours]
    metrics = price_metrics(ctx.run.model.loc[ctx.hours], observed)
    if ctx.same_year:
        second = price_metrics(ctx.compare.model.loc[ctx.hours], observed)
        metrics["delta_mae"] = metrics.mae - second.mae
    return metrics


def zone_map(ctx: Context, values: pd.Series, column: str) -> go.Figure:
    """Choropleth of a bus metric over the bus regions, carrying the area of each bus."""
    limit = symmetric_limit(values)
    scale = (
        dict(colorscale=DIVERGING, zmid=0, zmin=-limit, zmax=limit)
        if column in DIVERGING_METRICS
        else dict(colorscale="Viridis")
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
            **scale,
        )
    )
    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator")
    fig.update_layout(
        height=520, margin=dict(l=0, r=0, t=0, b=0), clickmode="event+select"
    )
    return fig


def hourly_line(ctx: Context) -> go.Figure:
    """Model, observed and comparison prices of the year with the period shaded and zoomed."""
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
    if len(ctx.hours) < len(model):
        fig.add_vrect(x0=start, x1=end, fillcolor="grey", opacity=0.1, line_width=0)
    fig.update_xaxes(range=[start, end], rangeslider=dict(visible=True))
    fig.update_layout(
        height=520, hovermode="x unified", yaxis_title="€/MWh", margin=dict(t=20)
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
    fig.update_layout(xaxis_title="share of hours", yaxis_title="€/MWh", height=420)
    st.plotly_chart(fig, width="stretch")
    shares = pd.DataFrame({name: tail_shares(prices) for name, prices, _ in series})
    st.dataframe(shares.style.format("{:.1%}"))


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
        fig.update_layout(title=season, xaxis_title="hour (CET)", yaxis_title="€/MWh")
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
        xaxis_title="observed €/MWh", yaxis_title="model €/MWh", height=500
    )
    return fig


def render(ctx: Context) -> None:
    """Draw the zone explorer page."""
    left, right = st.columns([2, 3])
    with left:
        options = [m for m in MAP_METRICS if ctx.same_year or m != "Δ MAE vs B"]
        name = st.selectbox("Map metric", options, key="map_metric")
        values = bus_metrics(ctx)[MAP_METRICS[name]]
        points = show(zone_map(ctx, values, MAP_METRICS[name]), "zone_map", select=True)
        if points:
            area = points[0]["customdata"]
            follow_selection("zone_map", (area,), area=area)
    with right:
        st.plotly_chart(hourly_line(ctx), width="stretch")
    series = period_series(ctx)
    tabs = st.tabs(["duration", "daily profile", "density", "month × hour error"])
    with tabs[0]:
        duration(series)
    with tabs[1]:
        profiles(series)
    with tabs[2]:
        st.plotly_chart(density(series[1][1], series[0][1]), width="stretch")
    with tabs[3]:
        error = ctx.run.area_model[ctx.area] - ctx.run.area_observed[ctx.area]
        table = month_hour_errors(error, cet_hour(error.index))
        st.plotly_chart(heatmap(table, "mean error (€/MWh)"), width="stretch")
    if ctx.compare is not None and not ctx.same_year:
        st.info(f"{ctx.compare.name} covers another year, hourly overlays disabled")
