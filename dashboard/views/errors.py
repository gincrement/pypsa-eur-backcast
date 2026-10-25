# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Error patterns: error heatmaps by area and month, observed price level and hour of day.
"""

import pandas as pd
import streamlit as st

from dashboard.data import MONTH_NAMES, TOP_SHARE, level_errors, month_errors
from dashboard.state import follow_selection
from dashboard.views.common import (
    MONTHS,
    Context,
    heatmap,
    show,
    takeaway,
    zone_link,
)
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


def links(by_month: dict[str, pd.DataFrame], levels: pd.DataFrame) -> list:
    """Zone detail of the worst zone in the worst month and of the largest error in the cheapest hours."""
    mae = by_month["MAE"]
    month = mae.median().idxmax()
    return [
        zone_link(mae[month].idxmax(), period=MONTH_NAMES[month - 1]),
        zone_link(levels["D1"].abs().idxmax()),
    ]


def render(ctx: Context) -> None:
    """Draw the error patterns page, a click on a cell opens the zone detail of its area."""
    error = ctx.run.area_model - ctx.run.area_observed
    by_month = {
        name: table.set_axis(table.columns.month, axis=1)
        for name, table in month_errors(error).items()
    }
    period = error.loc[ctx.hours]
    observed = ctx.run.area_observed.loc[ctx.hours]
    columns = [*[f"D{d}" for d in range(1, 11)], TOP]
    levels = {
        "bias": level_errors(period, observed).set_axis(columns, axis=1),
        "MAE": level_errors(period.abs(), observed).set_axis(columns, axis=1),
    }
    hours = {season: hourly_errors(error, months) for season, months in SEASONS.items()}
    takeaway(lead(by_month, levels["bias"], hours), links(by_month, levels["bias"]))
    name = (
        st.segmented_control(
            "Metric of both maps", ["bias", "MAE"], default="bias", key="month_metric"
        )
        or "bias"
    )
    diverging = name == "bias"
    left, right = st.columns([3, 2])
    with left:
        st.subheader("By month")
        table = by_month[name].set_axis(
            [MONTHS[m - 1] for m in by_month[name].columns], axis=1
        )
        points = show(
            heatmap(table, "€/MWh", diverging=diverging, text=True),
            f"month_{name}",
            select=True,
        )
        st.caption(
            "Model minus observed, €/MWh, whole year. Click a cell to open that zone and month."
        )
        cell = tuple(points[0]["customdata"]) if points else None
        follow_selection(
            f"month_{name}",
            cell,
            area=cell and cell[0],
            period=cell and MONTH_NAMES[MONTHS.index(cell[1])],
            page="Zone detail",
        )
    with right:
        st.subheader("By observed price level")
        figure = heatmap(levels[name], "€/MWh", diverging=diverging)
        figure.update_traces(xgap=1, selector=dict(type="heatmap"))
        figure.add_vline(x=len(columns) - 1.5, line_width=2, line_color="grey")
        points = show(figure, f"level_{name}", select=True)
        st.caption(
            f"Per decile of observed prices (D1 cheapest) and the {TOP} hours, €/MWh, {ctx.period}. "
            "Click a row to open the zone."
        )
        follow_zone(f"level_{name}", points)
    st.subheader("By hour of day (CET)")
    st.caption("Mean error, €/MWh, whole year. Click a row to open the zone.")
    for column, (season, table) in zip(st.columns(len(hours)), hours.items()):
        with column:
            st.markdown(f"**{season}**")
            follow_zone(
                f"hour_{season}",
                show(heatmap(table, "€/MWh"), f"hour_{season}", select=True),
            )


def follow_zone(key: str, points: list[dict]) -> None:
    """Open the zone detail of the area of the selected heatmap cell."""
    area = points[0]["customdata"][0] if points else None
    follow_selection(key, area, area=area, page="Zone detail")
