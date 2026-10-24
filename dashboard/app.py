# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Streamlit app to explore the day-ahead price backcast: ``pixi run -e dashboard backcast-dashboard``.
"""

from collections.abc import Callable
from datetime import date

import pandas as pd
import streamlit as st
import yaml

from dashboard import data
from dashboard.state import DEFAULTS, init_state, sync_query_params
from dashboard.views import PAGES, compare, dispatch, errors, scorecard, zone
from dashboard.views.common import Context
from scripts.lib.backcast_plotting import STACK, group_colors

RENDER: dict[str, Callable[[Context], None]] = dict(
    zip(
        PAGES,
        [scorecard.render, zone.render, errors.render, dispatch.render, compare.render],
    )
)


@st.cache_data
def cached_run(name: str) -> data.RunData:
    return data.load_run(data.CACHE / name)


@st.cache_data
def cached_variant(variant: str, run: str) -> pd.DataFrame:
    return data.load_variant(
        data.discover_variants(data.VARIANTS)[variant], cached_run(run).observed
    )


@st.cache_data
def carrier_colors() -> dict[str, str]:
    plotting = yaml.safe_load((data.ROOT / "config/plotting.default.yaml").read_text())
    return group_colors(plotting["plotting"]["tech_colors"], STACK)


def comparison(name: str, run: data.RunData, runs: list[str]) -> data.Comparison | None:
    """Load the comparison run or the variant aligned to the cached run of its year."""
    if not name:
        return None
    if name in runs:
        other = cached_run(name)
        return data.Comparison(name, other.model, other)
    year = data.variant_year(data.discover_variants(data.VARIANTS)[name])
    same = [r for r in runs if cached_run(r).year == year]
    if not same:
        st.sidebar.error(f"No extracted run of {year} to align {name}")
        return None
    return data.Comparison(name, cached_variant(name, same[0]), None)


def period_dates(run: data.RunData) -> tuple[date, date]:
    """Custom date range from the state, defaulting to the run's span."""
    first, last = run.observed.index[[0, -1]]
    start = st.session_state.start or str(first.date())
    end = st.session_state.end or str(last.date())
    return date.fromisoformat(start), date.fromisoformat(end)


def sidebar(runs: list[str]) -> Context:
    """Draw the global sidebar and return the page context."""
    bar = st.sidebar
    bar.radio("Page", PAGES, key="page")
    if st.session_state.run_a not in runs:
        st.session_state.run_a = runs[-1]
    bar.selectbox("Run A", runs, key="run_a")
    run = cached_run(st.session_state.run_a)
    choices = [
        "",
        *[r for r in runs if r != run.name],
        *data.discover_variants(data.VARIANTS),
    ]
    if st.session_state.run_b not in choices:
        st.session_state.run_b = ""
    bar.selectbox(
        "Run B or variant", choices, key="run_b", format_func=lambda c: c or "none"
    )
    areas = list(run.buses.index)
    if st.session_state.area not in areas:
        st.session_state.area = (
            DEFAULTS["area"] if DEFAULTS["area"] in areas else areas[0]
        )
    bar.selectbox("Area", areas, key="area")
    bar.radio("Period", data.PRESETS, key="period", horizontal=True)
    dates = None
    if st.session_state.period == "month":
        bar.slider("Month", 1, 12, key="month")
    if st.session_state.period == "custom":
        picked = bar.date_input("Date range", period_dates(run))
        if len(picked) == 2:
            st.session_state.start, st.session_state.end = map(str, picked)
        dates = tuple(pd.Timestamp(d) for d in period_dates(run))
    bar.radio("Aggregation", list(data.AGGREGATIONS), key="agg", horizontal=True)
    stale = data.stale_sources(run.manifest)
    if stale:
        bar.warning(f"Extract older than {len(stale)} sources, rerun backcast-extract")
    hours = data.period_index(
        run.observed.index,
        st.session_state.period,
        run.weeks,
        st.session_state.month,
        dates,
    )
    return Context(
        run=run,
        compare=comparison(st.session_state.run_b, run, runs),
        area=st.session_state.area,
        hours=hours,
        aggregation=st.session_state.agg,
        colors=carrier_colors(),
    )


st.set_page_config(page_title="Backcast dashboard", layout="wide")
init_state()
runs = data.discover_runs(data.CACHE)
if not runs:
    st.error(
        f"No extracts in {data.CACHE}, run `pixi run -e dashboard backcast-extract results/<run>`"
    )
    st.stop()
context = sidebar(runs)
st.title(f"{st.session_state.page}: {context.run.name}, {context.area}")
RENDER[st.session_state.page](context)
sync_query_params()
