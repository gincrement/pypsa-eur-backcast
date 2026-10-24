# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
P1 scorecard: headline metrics, metric table and per-area price, MAE and correlation bars.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.data import KPIS, kpis
from dashboard.state import follow_selection
from dashboard.views.common import Context, group_breaks, show
from scripts.lib.backcast_plotting import (
    BAND_COLOR,
    MODEL_COLOR,
    OBSERVED_COLOR,
    SECOND_COLOR,
)

LOWER_IS_BETTER = {"median MAE", "median |bias|"}
TABLE_COLUMNS = [
    "mean_bias",
    "mae",
    "rmse",
    "pearson_r",
    "std_ratio",
    "mean_observed",
    "mean_model",
    "zero_share_model",
    "zero_share_observed",
]


def kpi_tiles(ctx: Context, metrics: pd.DataFrame, second: pd.DataFrame | None) -> None:
    """Show the headline numbers with the change against the comparison."""
    values = kpis(metrics)
    reference = None if second is None else kpis(second)
    for column, name in zip(st.columns(len(KPIS)), KPIS):
        delta = None if reference is None else values[name] - reference[name]
        column.metric(
            name,
            f"{values[name]:.2f}",
            None if delta is None else f"{delta:+.2f} vs B",
            delta_color="inverse" if name in LOWER_IS_BETTER else "normal",
        )


def bars(metrics: pd.DataFrame, second: pd.DataFrame | None, label: str) -> go.Figure:
    """Dumbbell of mean prices and bars of MAE and Pearson r per area in zone order."""
    areas = list(metrics.index)
    fig = make_subplots(
        1, 3, shared_yaxes=True, subplot_titles=["mean price", "MAE", "Pearson r"]
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
    for y in group_breaks(areas):
        fig.add_hline(y=y, line_width=0.6, line_color="grey")
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_layout(
        height=18 * len(areas) + 120, barmode="group", clickmode="event+select"
    )
    return fig


def render(ctx: Context) -> None:
    """Draw the scorecard page."""
    metrics = ctx.metrics()
    second = None if ctx.compare is None else ctx.compare_metrics()
    kpi_tiles(ctx, metrics, second)
    left, right = st.columns([2, 3])
    table = metrics[TABLE_COLUMNS].round(3)
    event = left.dataframe(
        table, on_select="rerun", selection_mode="single-row", key="scorecard_table"
    )
    rows = event["selection"]["rows"]
    with right:
        points = show(
            bars(metrics, second, ctx.run.name), "scorecard_bars", select=True
        )
    if rows:
        follow_selection(
            "scorecard_table", (table.index[rows[0]],), area=table.index[rows[0]]
        )
    if points:
        area = points[0]["customdata"]
        follow_selection("scorecard_bars", (area,), area=area)
