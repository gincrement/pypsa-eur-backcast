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
from dashboard.views import (
    CONTROLS,
    PAGES,
    QUESTIONS,
    compare,
    dispatch,
    errors,
    overview,
    report,
    zone,
)
from dashboard.views.common import Context, Reference
from scripts.lib.backcast_plotting import STACK, group_colors

AUTO_DAILY_HOURS = 24 * 62
RENDER: dict[str, Callable[[Context], None]] = dict(
    zip(
        PAGES,
        [
            report.render,
            overview.render,
            errors.render,
            zone.render,
            dispatch.render,
            compare.render,
        ],
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
    if period in data.MONTH_NAMES:
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
        run.observed.index, st.session_state.period, run.weeks, dates
    )


def reference(
    run: data.RunData, dates: tuple[pd.Timestamp, pd.Timestamp] | None
) -> Reference:
    """Validation run over the same period preset, custom dates moved to its year."""
    moved = None if dates is None else tuple(d.replace(year=run.year) for d in dates)
    hours = period_hours(run, moved)
    return Reference(run, hours, period_label(st.session_state.period, hours))


def normalise(runs: list[str]) -> tuple[data.RunData, list[str]]:
    """Reset state values that the selected run does not offer and return the run and the comparison choices."""
    state = st.session_state
    if state.page not in PAGES:
        state.page = PAGES[0]
    if state.run_a not in runs:
        state.run_a = runs[-1]
    run = cached_run(state.run_a)
    choices = [
        "",
        *[r for r in runs if r != run.name],
        *data.discover_variants(data.VARIANTS),
    ]
    if state.run_b not in choices:
        state.run_b = ""
    areas = list(run.buses.index)
    if state.area not in areas:
        state.area = DEFAULTS["area"] if DEFAULTS["area"] in areas else areas[0]
    if state.period not in data.PERIODS:
        state.period = DEFAULTS["period"]
    if state.agg not in ["auto", *data.AGGREGATIONS]:
        state.agg = DEFAULTS["agg"]
    return run, choices


def sidebar(runs: list[str]) -> Context:
    """Draw the page list and the controls the page uses, return the page context."""
    run, choices = normalise(runs)
    bar = st.sidebar
    bar.radio("Page", PAGES, key="page")
    controls = CONTROLS[st.session_state.page]
    if controls:
        bar.divider()
    if "run_a" in controls:
        bar.selectbox("Run A", runs, key="run_a")
    if "run_b" in controls:
        bar.selectbox(
            "Run B or variant", choices, key="run_b", format_func=lambda c: c or "none"
        )
    if "area" in controls:
        bar.selectbox("Bidding zone", list(run.buses.index), key="area")
    if "period" in controls:
        bar.selectbox("Period", data.PERIODS, key="period")
    custom = st.session_state.period == "custom"
    if custom and "period" in controls:
        picked = bar.date_input("Date range", period_dates(run))
        if len(picked) == 2:
            st.session_state.start, st.session_state.end = map(str, picked)
    dates = tuple(pd.Timestamp(d) for d in period_dates(run)) if custom else None
    if "agg" in controls:
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


def selection(ctx: Context, controls: list[str]) -> str:
    """The sidebar state a page uses as one line."""
    run = ctx.run.name
    if "run_b" in controls and ctx.compare is not None:
        run += f" vs {ctx.compare.name}"
    parts = [run, ctx.area if "area" in controls else "", ctx.period]
    return " | ".join(p for p in parts if p) if controls else ""


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
title, line = st.columns([3, 2], vertical_alignment="bottom")
title.title(page)
line.caption(selection(context, CONTROLS[page]), text_alignment="right")
st.markdown(
    QUESTIONS[page].format(
        area=context.area, run=context.run.name, compare=compare_name
    )
)
RENDER[st.session_state.page](context)
sync_query_params()
