# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Price replay: market and model prices per model bus over one week as two animated maps.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard import narrative
from dashboard.data import RunData
from dashboard.views.common import Context, map_layout, region_choropleth
from scripts.lib.backcast_plotting import SEQUENTIAL

FRAME_MS = 150
COLOR_QUANTILES = (0.02, 0.98)


def replay_figure(run: RunData, week: pd.DatetimeIndex) -> go.Figure:
    """Side-by-side maps of market and model prices with one frame per hour sharing a play button and slider, frames update only the colours."""
    observed = run.observed.loc[week]
    model = run.model.loc[week, observed.columns]
    low, high = pd.concat([observed, model]).stack().quantile(COLOR_QUANTILES)
    fig = make_subplots(
        1,
        2,
        specs=[[{"type": "geo"}, {"type": "geo"}]],
        subplot_titles=["market", "model"],
        horizontal_spacing=0.01,
    )
    for col, prices in enumerate([observed, model], start=1):
        trace = region_choropleth(
            run, prices.iloc[0], coloraxis="coloraxis", marker_line_width=0.3
        )
        fig.add_trace(trace, row=1, col=col)
    labels = [f"{hour:%a %d %b %H:00}" for hour in week]
    fig.frames = [
        go.Frame(
            name=label,
            data=[
                go.Choropleth(z=observed.loc[hour]),
                go.Choropleth(z=model.loc[hour]),
            ],
            traces=[0, 1],
        )
        for hour, label in zip(week, labels)
    ]
    play = dict(frame=dict(duration=FRAME_MS, redraw=True), fromcurrent=True)
    jump = dict(mode="immediate", frame=dict(duration=0, redraw=True))
    fig.update_layout(
        coloraxis=dict(
            colorscale=SEQUENTIAL.capitalize(),
            cmin=low,
            cmax=high,
            colorbar=dict(title="€/MWh", thickness=12, len=0.8),
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0,
                y=0,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(label="▶ play", method="animate", args=[None, play]),
                    dict(label="❚❚ pause", method="animate", args=[[None], jump]),
                ],
            )
        ],
        sliders=[
            dict(
                x=0.15,
                len=0.85,
                y=0,
                yanchor="top",
                currentvalue=dict(prefix="hour: "),
                steps=[
                    dict(label=label, method="animate", args=[[label], jump])
                    for label in labels
                ],
            )
        ],
    )
    map_layout(fig)
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=90))
    return fig


def replay(ctx: Context, week: pd.DatetimeIndex) -> None:
    """Draw the price replay of the week in an expander that only builds the maps when opened."""
    box = st.expander(narrative.REPLAY, key="replay", on_change="rerun")
    if box.open:
        box.markdown(narrative.REPLAY_LEAD)
        box.plotly_chart(replay_figure(ctx.run, week), width="stretch")
