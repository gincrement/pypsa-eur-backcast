# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
P3 error structure: error heatmaps by area and month, observed price level and hour of day.
"""

import streamlit as st

from dashboard.data import level_errors, month_errors
from dashboard.state import follow_selection
from dashboard.views.common import Context, heatmap, show
from scripts.lib.backcast_data import hourly_errors
from scripts.lib.backcast_plotting import SEASONS


def render(ctx: Context) -> None:
    """Draw the error structure page, a click on a month cell opens the zone explorer for that area and month."""
    error = ctx.run.area_model - ctx.run.area_observed
    by_month = month_errors(error)
    columns = st.columns(len(by_month))
    for column, (name, table) in zip(columns, by_month.items()):
        table = table.set_axis(table.columns.month, axis=1)
        with column:
            figure = heatmap(
                table, f"{name} (€/MWh)", diverging=name == "bias", text=True
            )
            points = show(figure, f"month_{name}", select=True)
        if points:
            area, month = points[0]["customdata"]
            follow_selection(
                f"month_{name}",
                (area, month),
                area=area,
                month=int(month),
                period="month",
                page="Zone explorer",
            )
    hours = error.loc[ctx.hours]
    observed = ctx.run.area_observed.loc[ctx.hours]
    st.subheader("by observed price level (period)")
    figure = heatmap(level_errors(hours, observed), "mean error (€/MWh)", text=True)
    points = show(figure, "level", select=True)
    if points:
        area = points[0]["customdata"][0]
        follow_selection("level", (area,), area=area)
    st.subheader("by hour of day (CET)")
    for column, (season, months) in zip(st.columns(len(SEASONS)), SEASONS.items()):
        column.plotly_chart(
            heatmap(hourly_errors(error, months), f"{season} error (€/MWh)"),
            width="stretch",
        )
