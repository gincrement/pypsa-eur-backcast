# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Page context and Plotly helpers shared by the dashboard pages.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.data import Comparison, RunData, area_metrics
from scripts.lib.backcast_plotting import (
    DIVERGING,
    MODEL_COLOR,
    OBSERVED_COLOR,
    SECOND_COLOR,
    ZONE_GROUPS,
    ZONE_ORDER,
    symmetric_limit,
)

HEATMAP_HEIGHT = 18
MARKER_COLORS = {"negative price": "#CC79A7", "load shedding": "#D55E00"}


@dataclass(frozen=True)
class Context:
    """Run A, the optional comparison and the sidebar selection."""

    run: RunData
    compare: Comparison | None
    area: str
    hours: pd.DatetimeIndex
    aggregation: str
    colors: dict[str, str]

    @property
    def bus(self) -> str:
        """Representative bus of the selected area."""
        return self.run.buses[self.area]

    @property
    def same_year(self) -> bool:
        """Whether the comparison covers the year of run A."""
        return self.compare is not None and (
            self.compare.run is None or self.compare.run.year == self.run.year
        )

    def metrics(self) -> pd.DataFrame:
        """Area metrics of run A in the period."""
        return area_metrics(
            self.run.area_model.loc[self.hours], self.run.area_observed.loc[self.hours]
        )

    def compare_metrics(self) -> pd.DataFrame:
        """Area metrics of the comparison in the period, or over its own year if it covers another year."""
        assert self.compare is not None
        if not self.same_year:
            run = self.compare.run
            return area_metrics(run.area_model, run.area_observed)
        model = self.compare.area_model(self.run.buses)
        return area_metrics(
            model.loc[self.hours], self.run.area_observed.loc[self.hours]
        )


def selected_points(event: dict | None) -> list[dict]:
    """Selected points of a Plotly selection event."""
    return [] if event is None else list(event["selection"]["points"])


def group_breaks(areas: list[str]) -> list[float]:
    """Positions between consecutive areas of different zone groups."""
    group = pd.Series({a: g for g, members in ZONE_GROUPS.items() for a in members})
    values = group[areas].to_numpy()
    return list(np.flatnonzero(values[1:] != values[:-1]) + 0.5)


def heatmap(
    data: pd.DataFrame,
    title: str,
    diverging: bool = True,
    text: bool = False,
) -> go.Figure:
    """Draw an area heatmap with group separators and a transparent scatter layer to select cells."""
    rows, columns = [str(i) for i in data.index], [str(c) for c in data.columns]
    limit = symmetric_limit(data, 0.95)
    scale = (
        dict(colorscale=DIVERGING, zmid=0, zmin=-limit, zmax=limit)
        if diverging
        else dict(colorscale="Viridis")
    )
    fig = go.Figure(
        go.Heatmap(
            z=data.to_numpy(),
            x=columns,
            y=rows,
            text=data.round(0).to_numpy() if text else None,
            texttemplate="%{text}" if text else None,
            hovertemplate="%{y} %{x}: %{z:.1f}<extra></extra>",
            colorbar=dict(title=title),
            **scale,
        )
    )
    grid_y, grid_x = np.meshgrid(rows, columns, indexing="ij")
    fig.add_scatter(
        x=grid_x.ravel(),
        y=grid_y.ravel(),
        mode="markers",
        marker=dict(size=12, opacity=0.01, symbol="square"),
        customdata=np.stack([grid_y.ravel(), grid_x.ravel()], axis=1),
        hoverinfo="skip",
        showlegend=False,
    )
    if data.index.isin(ZONE_ORDER).all():
        for y in group_breaks(rows):
            fig.add_hline(y=y, line_width=1, line_color="grey")
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_xaxes(type="category")
    fig.update_layout(
        title=title,
        height=HEATMAP_HEIGHT * len(rows) + 120,
        clickmode="event+select",
        margin=dict(t=40, b=30),
    )
    return fig


def price_traces(
    ctx: Context, model: pd.Series, observed: pd.Series, second: pd.Series | None
) -> list[go.Scatter]:
    """Model, observed and second price lines, hovering the error."""
    error = (model - observed).to_numpy()
    traces = [
        go.Scatter(
            x=observed.index,
            y=observed,
            name="observed",
            line=dict(color=OBSERVED_COLOR, width=1),
        ),
        go.Scatter(
            x=model.index,
            y=model,
            name=ctx.run.name,
            line=dict(color=MODEL_COLOR, width=1),
            customdata=error,
            hovertemplate="%{y:.1f} (error %{customdata:.1f})",
        ),
    ]
    if second is not None:
        name = ctx.compare.name
        traces.append(
            go.Scatter(
                x=second.index,
                y=second,
                name=name,
                line=dict(color=SECOND_COLOR, width=1),
            )
        )
    return traces


def event_markers(model: pd.Series, shedding: pd.Series) -> list[go.Scatter]:
    """Markers on the model price line at negative-price and load-shedding hours."""
    masks = {
        "negative price": model < 0,
        "load shedding": shedding.reindex(model.index) > 0,
    }
    return [
        go.Scatter(
            x=model.index[mask],
            y=model[mask],
            mode="markers",
            name=name,
            marker=dict(color=MARKER_COLORS[name], size=5),
        )
        for name, mask in masks.items()
        if mask.any()
    ]


def show(fig: go.Figure, key: str, select: bool = False) -> list[dict]:
    """Render a figure and return the selected points if selection is enabled."""
    if not select:
        st.plotly_chart(fig, key=key, width="stretch")
        return []
    event = st.plotly_chart(
        fig, key=key, on_select="rerun", selection_mode="points", width="stretch"
    )
    return selected_points(event)
