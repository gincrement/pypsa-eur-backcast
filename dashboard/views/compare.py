# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
P5 compare: metric changes of run A against run B or a variant per area, month and hour.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.data import aggregate, month_errors
from dashboard.views.common import Context, group_breaks, heatmap, price_traces
from scripts.lib.backcast_plotting import MODEL_COLOR, SECOND_COLOR

DELTAS = {"Δ MAE": "mae", "Δ |bias|": "abs_bias", "Δ r": "pearson_r"}
SCATTER_METRICS = ["mae", "abs_bias", "pearson_r", "std_ratio", "mean_model"]


def delta_bars(delta: pd.DataFrame) -> go.Figure:
    """Bars of the metric changes A minus B per area in zone order."""
    areas = list(delta.index)
    fig = make_subplots(1, len(DELTAS), shared_yaxes=True, subplot_titles=list(DELTAS))
    for col, (name, column) in enumerate(DELTAS.items(), start=1):
        fig.add_bar(
            x=delta[column],
            y=areas,
            orientation="h",
            marker_color=MODEL_COLOR,
            name=name,
            row=1,
            col=col,
        )
    for y in group_breaks(areas):
        fig.add_hline(y=y, line_width=0.6, line_color="grey")
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_layout(height=18 * len(areas) + 120, showlegend=False)
    return fig


def metric_scatter(
    a: pd.DataFrame, b: pd.DataFrame, column: str, names: tuple[str, str]
) -> go.Figure:
    """Scatter of a per-area metric of A over B with the identity line."""
    bounds = [
        min(a[column].min(), b[column].min()),
        max(a[column].max(), b[column].max()),
    ]
    fig = go.Figure(
        go.Scatter(
            x=b[column],
            y=a[column],
            mode="markers+text",
            text=a.index,
            textposition="top center",
            marker=dict(color=MODEL_COLOR),
            name="areas",
        )
    )
    fig.add_scatter(
        x=bounds, y=bounds, mode="lines", line=dict(color="black"), name="identity"
    )
    fig.update_layout(
        xaxis_title=f"{names[1]} {column}",
        yaxis_title=f"{names[0]} {column}",
        height=500,
    )
    return fig


def skill_scatter(runs: dict[str, pd.DataFrame]) -> go.Figure:
    """Pearson r over MAE relative to the observed standard deviation per area for both runs."""
    fig = go.Figure()
    for (name, metrics), color in zip(runs.items(), [MODEL_COLOR, SECOND_COLOR]):
        fig.add_scatter(
            x=metrics.pearson_r,
            y=metrics.mae / metrics.std_observed,
            mode="markers+text",
            text=metrics.index,
            textposition="top center",
            marker=dict(color=color),
            name=name,
        )
    fig.update_layout(
        xaxis_title="Pearson r", yaxis_title="MAE / observed std", height=500
    )
    return fig


def render(ctx: Context) -> None:
    """Draw the comparison page."""
    if ctx.compare is None:
        st.info("Select run B or a variant in the sidebar")
        return
    a, b = ctx.metrics(), ctx.compare_metrics()
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    names = (ctx.run.name, ctx.compare.name)
    st.plotly_chart(delta_bars(a - b), width="stretch")
    left, right = st.columns(2)
    column = left.selectbox("Scatter metric", SCATTER_METRICS)
    left.plotly_chart(metric_scatter(a, b, column, names), width="stretch")
    right.plotly_chart(skill_scatter(dict(zip(names, [a, b]))), width="stretch")
    if not ctx.same_year:
        st.info(
            f"{ctx.compare.name} covers another year, monthly and hourly comparisons disabled"
        )
        return
    observed = ctx.run.area_observed
    second = ctx.compare.area_model(ctx.run.buses)
    delta = (
        month_errors(ctx.run.area_model - observed)["MAE"]
        - month_errors(second - observed)["MAE"]
    )
    st.plotly_chart(
        heatmap(
            delta.set_axis(delta.columns.month, axis=1), "Δ MAE (€/MWh)", text=True
        ),
        width="stretch",
    )
    series = [
        aggregate(frame.loc[ctx.hours, ctx.area], ctx.aggregation)
        for frame in (ctx.run.area_model, observed, second)
    ]
    figure = go.Figure(price_traces(ctx, *series))
    figure.update_layout(height=450, hovermode="x unified", yaxis_title="€/MWh")
    st.plotly_chart(figure, width="stretch")
