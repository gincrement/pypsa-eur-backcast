# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Error patterns: error heatmaps by area and month, observed price level and hour of day.
"""

import pandas as pd
import streamlit as st

from dashboard.data import TOP_SHARE, level_errors, month_errors
from dashboard.state import follow_selection
from dashboard.views.common import MONTHS, Context, heatmap, show, takeaway
from scripts.lib.backcast_data import hourly_errors
from scripts.lib.backcast_plotting import SEASONS

TOP = f"top {TOP_SHARE:.0%}"


def lead(
    by_month: dict[str, pd.DataFrame],
    levels: pd.DataFrame,
    hours: dict[str, pd.DataFrame],
) -> str:
    """Summarise the worst month, the error at the price extremes and the worst hour per season."""
    mae = by_month["MAE"].median()
    peaks = ", ".join(
        f"{season.split()[0]} at {table.median().idxmax()}:00 ({table.median().max():+.0f})"
        for season, table in hours.items()
    )
    return (
        f"Median MAE across zones peaks in **{MONTHS[mae.idxmax() - 1]}** "
        f"({mae.max():.1f} €/MWh) and is lowest in {MONTHS[mae.idxmin() - 1]} ({mae.min():.1f}). "
        f"In the cheapest decile of hours the model is too high by a median "
        f"**{levels['D1'].median():+.0f} €/MWh**; in the {TOP} most expensive hours it is off by "
        f"**{levels[TOP].median():+.0f} €/MWh**. Largest median hourly bias: {peaks} €/MWh."
    )


def render(ctx: Context) -> None:
    """Draw the error patterns page, a click on a month cell opens the zone detail for that area and month."""
    error = ctx.run.area_model - ctx.run.area_observed
    by_month = {
        name: table.set_axis(table.columns.month, axis=1)
        for name, table in month_errors(error).items()
    }
    period = error.loc[ctx.hours]
    levels = level_errors(period, ctx.run.area_observed.loc[ctx.hours])
    levels = levels.set_axis([*[f"D{d}" for d in range(1, 11)], TOP], axis=1)
    hours = {season: hourly_errors(error, months) for season, months in SEASONS.items()}
    takeaway(lead(by_month, levels, hours))
    left, right = st.columns([3, 2])
    with left:
        st.subheader("By month")
        name = (
            st.segmented_control(
                "Metric", ["bias", "MAE"], default="bias", key="month_metric"
            )
            or "bias"
        )
        st.caption(
            "Model minus observed, €/MWh, whole year. Click a cell to open that zone and month."
        )
        table = by_month[name].set_axis(
            [MONTHS[m - 1] for m in by_month[name].columns], axis=1
        )
        points = show(
            heatmap(table, "€/MWh", diverging=name == "bias", text=True),
            f"month_{name}",
            select=True,
        )
        if points:
            area, month = points[0]["customdata"]
            follow_selection(
                f"month_{name}",
                (area, month),
                area=area,
                month=MONTHS.index(month) + 1,
                period="month",
                page="Zone detail",
            )
    with right:
        st.subheader("By observed price level")
        st.caption(
            f"Mean error per decile of observed prices (D1 cheapest) and the {TOP} hours, €/MWh, {ctx.period}."
        )
        points = show(heatmap(levels, "€/MWh", text=True), "level", select=True)
        if points:
            area = points[0]["customdata"][0]
            follow_selection("level", (area,), area=area)
    st.subheader("By hour of day (CET)")
    st.caption("Mean error, €/MWh, whole year.")
    for column, (season, table) in zip(st.columns(len(hours)), hours.items()):
        column.markdown(f"**{season}**")
        column.plotly_chart(
            heatmap(table, "€/MWh"), width="stretch", key=f"hour_{season}"
        )
