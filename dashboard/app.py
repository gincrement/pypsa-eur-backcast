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
from dashboard.views import PAGES, QUESTIONS, compare, dispatch, errors, overview, zone
from dashboard.views.common import Context, Reference
from scripts.lib.backcast_plotting import STACK, group_colors

AUTO_DAILY_HOURS = 24 * 62
RENDER: dict[str, Callable[[Context], None]] = dict(
    zip(
        PAGES,
        [overview.render, errors.render, zone.render, dispatch.render, compare.render],
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


def period_label(period: str, hours: pd.DatetimeIndex) -> str:
    """Readable name of the selected period."""
    if period == "month":
        return hours[0].strftime("%B %Y")
    if period == "year":
        return f"the year {hours[0].year}"
    return f"{period} {hours[0]:%d %b} to {hours[-1]:%d %b %Y}"


def resolve_aggregation(choice: str, hours: pd.DatetimeIndex) -> str:
    """Daily means for periods longer than two months under auto, else the choice."""
    if choice != "auto":
        return choice
    return "daily" if len(hours) > AUTO_DAILY_HOURS else "hourly"


def period_hours(
    run: data.RunData, dates: tuple[pd.Timestamp, pd.Timestamp] | None
) -> pd.DatetimeIndex:
    """Hours of the selected period preset in the year of a run."""
    return data.period_index(
        run.observed.index,
        st.session_state.period,
        run.weeks,
        st.session_state.month,
        dates,
    )


def reference(
    run: data.RunData, dates: tuple[pd.Timestamp, pd.Timestamp] | None
) -> Reference:
    """Validation run over the same period preset, custom dates moved to its year."""
    moved = None if dates is None else tuple(d.replace(year=run.year) for d in dates)
    hours = period_hours(run, moved)
    return Reference(run, hours, period_label(st.session_state.period, hours))


def sidebar(runs: list[str]) -> Context:
    """Draw the global sidebar and return the page context."""
    bar = st.sidebar
    if st.session_state.page not in PAGES:
        st.session_state.page = PAGES[0]
    bar.radio("Page", PAGES, key="page")
    if st.session_state.run_a not in runs:
        st.session_state.run_a = runs[-1]
    bar.divider()
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
    bar.selectbox("Bidding zone", areas, key="area")
    bar.divider()
    bar.selectbox("Period", data.PRESETS, key="period")
    dates = None
    if st.session_state.period == "month":
        bar.slider("Month", 1, 12, key="month")
    if st.session_state.period == "custom":
        picked = bar.date_input("Date range", period_dates(run))
        if len(picked) == 2:
            st.session_state.start, st.session_state.end = map(str, picked)
        dates = tuple(pd.Timestamp(d) for d in period_dates(run))
    if st.session_state.agg not in ["auto", *data.AGGREGATIONS]:
        st.session_state.agg = "auto"
    bar.segmented_control(
        "Time series resolution", ["auto", *data.AGGREGATIONS], key="agg"
    )
    stale = data.stale_sources(run.manifest)
    if stale:
        bar.warning(
            f"Extract is older than {len(stale)} of its sources, rerun backcast-extract"
        )
    bar.caption(f"Extract {run.manifest['created']} at commit {run.manifest['commit']}")
    hours = period_hours(run, dates)
    return Context(
        run=run,
        compare=comparison(st.session_state.run_b, run, runs),
        area=st.session_state.area,
        hours=hours,
        period=period_label(st.session_state.period, hours),
        aggregation=resolve_aggregation(st.session_state.agg or "auto", hours),
        colors=carrier_colors(),
        references=[
            reference(cached_run(r), dates)
            for r in runs
            if cached_run(r).year != run.year
        ],
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
page = st.session_state.page
compare_name = context.compare.name if context.compare is not None else "run B"
st.title(page)
st.markdown(
    QUESTIONS[page].format(
        area=context.area, run=context.run.name, compare=compare_name
    )
)
RENDER[st.session_state.page](context)
sync_query_params()
