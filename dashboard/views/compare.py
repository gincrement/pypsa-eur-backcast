# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Compare runs: metric changes of run A against run B or a variant per area, month and hour.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.data import aggregate, month_errors
from dashboard.views.common import (
    CHART_HEIGHT,
    MONTHS,
    WORSE_COLOR,
    Context,
    full_year,
    heatmap,
    price_traces,
    takeaway,
    zone_layout,
    zone_link,
)
from dashboard.views.zone import duration_figure, profile_figures, run_lines
from scripts.lib.backcast_plotting import MODEL_COLOR, SECOND_COLOR

SAME_YEAR = {
    "Δ MAE (€/MWh)": "mae",
    "Δ |bias| (€/MWh)": "abs_bias",
    "Δ Pearson r": "pearson_r",
}
OTHER_YEAR = {"Δ MAE / mean price": "rel_mae", "Δ Pearson r": "pearson_r"}
HIGHER_IS_BETTER = {"pearson_r"}


def delta_bars(delta: pd.DataFrame, deltas: dict[str, str]) -> go.Figure:
    """Bars of the metric changes A minus B per area in zone order, coloured by improvement."""
    areas = list(delta.index)
    fig = make_subplots(
        1,
        len(deltas),
        shared_yaxes=True,
        horizontal_spacing=0.03,
        subplot_titles=list(deltas),
    )
    for col, column in enumerate(deltas.values(), start=1):
        better = delta[column] > 0 if column in HIGHER_IS_BETTER else delta[column] < 0
        fig.add_bar(
            x=delta[column],
            y=areas,
            orientation="h",
            marker_color=better.map({True: MODEL_COLOR, False: WORSE_COLOR}).tolist(),
            row=1,
            col=col,
        )
    fig.update_layout(showlegend=False)
    return zone_layout(fig, areas)


def skill_scatter(runs: dict[str, pd.DataFrame]) -> go.Figure:
    """Pearson r over MAE relative to the observed standard deviation per area for both runs."""
    fig = go.Figure()
    for (name, metrics), color in zip(runs.items(), [MODEL_COLOR, SECOND_COLOR]):
        fig.add_scatter(
            x=metrics.pearson_r,
            y=metrics.mae / metrics.std_observed,
            mode="markers",
            text=metrics.index,
            hovertemplate="%{text}: r %{x:.2f}, MAE/std %{y:.2f}",
            marker=dict(color=color, size=8),
            name=name,
        )
    fig.update_layout(
        xaxis_title="Pearson r (better →)",
        yaxis_title="MAE / observed std (better ↓)",
        height=CHART_HEIGHT + 120,
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return fig


def lead(ctx: Context, a: pd.DataFrame, b: pd.DataFrame, column: str) -> str:
    """Summarise the median change and the share of improved zones."""
    label = "MAE" if column == "mae" else "MAE / mean price"
    template = "{:.2f} €/MWh" if column == "mae" else "{:.0%}"
    better = (a[column] < b[column]).sum()
    text = (
        f"**{ctx.run.name}** vs **{ctx.compare.name}**: median {label} "
        f"**{template.format(a[column].median())}** vs {template.format(b[column].median())}, "
        f"median r {a.pearson_r.median():.2f} vs {b.pearson_r.median():.2f}. "
        f"{ctx.run.name} has the lower {label} in **{better} of {len(a)} zones**."
    )
    if ctx.same_year:
        return text
    return f"{text} The runs cover different years, so they are compared on relative MAE and r over each whole year."


def other_year(ctx: Context) -> None:
    """Duration curves and daily profiles of the selected zone over both whole years, run B dashed."""
    run, other = ctx.run, ctx.compare.run
    st.subheader(f"{ctx.area}: {run.name} vs {other.name}, whole years")
    if ctx.area not in other.buses:
        st.info(f"{other.name} has no prices for {ctx.area}")
        return
    lines = [
        *run_lines(run, ctx.area, run.observed.index),
        *run_lines(other, ctx.area, other.observed.index, "dash"),
    ]
    figures = [duration_figure(lines), *profile_figures(lines)]
    figures[0].update_layout(title="duration curve")
    for column, figure in zip(st.columns(len(figures)), figures):
        figure.update_layout(legend=dict(orientation="h", y=-0.25, x=0))
        column.plotly_chart(figure, width="stretch")


def render(ctx: Context) -> None:
    """Draw the comparison page."""
    if ctx.compare is None:
        st.info("Select run B or a variant in the sidebar to compare it with run A.")
        return
    a, b = ctx.metrics(), ctx.compare_metrics()
    if not ctx.same_year:
        a = full_year(ctx.run)
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    deltas = SAME_YEAR if ctx.same_year else OTHER_YEAR
    takeaway(lead(ctx, a, b, next(iter(deltas.values()))), [zone_link(ctx.area)])
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Change per zone (A minus B)")
        st.caption("Blue: A better, orange: A worse.")
        st.plotly_chart(delta_bars(a - b, deltas), width="stretch")
    with right:
        st.subheader("Skill per zone")
        st.plotly_chart(
            skill_scatter({ctx.run.name: a, ctx.compare.name: b}), width="stretch"
        )
    if not ctx.same_year:
        other_year(ctx)
        return
    observed = ctx.run.area_observed
    second = ctx.compare.area_model(ctx.run.buses)
    delta = (
        month_errors(ctx.run.area_model - observed)["MAE"]
        - month_errors(second - observed)["MAE"]
    )
    st.subheader("Δ MAE per month (A minus B, €/MWh, whole year)")
    st.plotly_chart(
        heatmap(
            delta.set_axis([MONTHS[m.month - 1] for m in delta.columns], axis=1),
            "€/MWh",
            text=True,
        ),
        width="stretch",
    )
    st.subheader(f"Prices in {ctx.area} ({ctx.period})")
    series = [
        aggregate(frame.loc[ctx.hours, ctx.area], ctx.aggregation)
        for frame in (ctx.run.area_model, observed, second)
    ]
    figure = go.Figure(price_traces(ctx, *series))
    figure.update_layout(
        height=CHART_HEIGHT, hovermode="x unified", yaxis_title="€/MWh"
    )
    st.plotly_chart(figure, width="stretch")
