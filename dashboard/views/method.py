# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
How it works: method, headline results, scoring, limitations and refinement history, and the run facts filling the prose.
"""

import pandas as pd
import streamlit as st

from dashboard import narrative
from dashboard.data import TOP_SHARE, RunData, area_metrics, net_export
from dashboard.views.common import Context, bus_metrics, export_direction, worst
from scripts.lib.backcast_data import top_errors

EXPORTER = "DE_LU"
INTEGER_ROWS = ["year", "scored buses"]


def summary(run: RunData) -> dict[str, float]:
    """Headline numbers of a run over its whole year, medians over model buses."""
    buses = bus_metrics(run.model, run.observed, run.observed.index)
    error = run.model - run.observed
    export = net_export(run.flows, EXPORTER, run.observed.index)
    return {
        "year": run.year,
        "scored buses": len(buses),
        "mean observed price (€/MWh)": buses.mean_observed.mean(),
        "median MAE (€/MWh)": buses.mae.median(),
        "median Pearson r": buses.pearson_r.median(),
        "median |bias| (€/MWh)": buses.abs_bias.median(),
        "hours ≤ 0 observed (%)": 100 * buses.nonpositive_observed.mean(),
        "hours ≤ 0 model (%)": 100 * buses.nonpositive_model.mean(),
        f"error in top {TOP_SHARE:.0%} hours (€/MWh)": top_errors(
            error, run.observed, TOP_SHARE
        ).median(),
        f"{EXPORTER} net export model (TWh)": export["model"],
        f"{EXPORTER} net export scheduled (TWh)": export["scheduled"],
    }


def facts(run: RunData, numbers: dict[str, float]) -> dict[str, str]:
    """Values filling the placeholders of the prose for a run."""
    metrics = area_metrics(run.area_model, run.area_observed)
    model = numbers[f"{EXPORTER} net export model (TWh)"]
    scheduled = numbers[f"{EXPORTER} net export scheduled (TWh)"]
    return {
        "run": run.name,
        "r": f"{numbers['median Pearson r']:.2f}",
        "top": f"{TOP_SHARE:.0%}",
        "top_error": f"{numbers[f'error in top {TOP_SHARE:.0%} hours (€/MWh)']:+.0f}",
        "nonpositive_observed": f"{numbers['hours ≤ 0 observed (%)']:.1f} %",
        "nonpositive_model": f"{numbers['hours ≤ 0 model (%)']:.1f} %",
        "exporter": narrative.NAMES[EXPORTER],
        "direction": export_direction(model, scheduled),
        "export_model": f"{model:+.0f}",
        "export_scheduled": f"{scheduled:+.0f}",
        "high_bias": worst(metrics.mean_bias),
        "low_bias": worst(-metrics.mean_bias),
        "worst_mae": worst(metrics.mae),
    }


def pipeline() -> None:
    """Draw inputs, model and output as three boxes."""
    for column, (title, items) in zip(st.columns(3), narrative.PIPELINE.items()):
        box = column.container(border=True)
        box.markdown(f"**{title}**")
        box.markdown("\n".join(f"- {item}" for item in items))


def method() -> None:
    """Draw what the backcast is and how it works."""
    st.header("What the model does")
    st.markdown(narrative.WHAT)
    st.markdown(narrative.HOW)
    pipeline()


def headline_results(table: pd.DataFrame, filled: dict[str, str]) -> None:
    """Draw what works and the whole-year results per run in an expander."""
    st.header("What works")
    st.markdown(narrative.WORKS.format(**filled))
    with st.expander("Headline results per run"):
        st.caption(
            "Whole year per run, independent of the period in the sidebar. "
            f"Top {TOP_SHARE:.0%} hours are the most expensive observed hours of each bus."
        )
        shown = table.map("{:.2f}".format)
        shown.loc[INTEGER_ROWS] = table.loc[INTEGER_ROWS].map("{:.0f}".format)
        st.table(shown)


def scoring() -> None:
    """Draw how a run is scored."""
    st.header("How it is scored")
    st.markdown(narrative.SCORING)


def limitations(filled: dict[str, str]) -> None:
    """Draw the limitations table filled from the run."""
    st.header("What it leaves out")
    st.markdown(narrative.LIMITATIONS.format(**filled))


def history() -> None:
    """Draw the refinement history."""
    st.header("How the model got better")
    st.markdown(narrative.HISTORY)


def render(ctx: Context) -> None:
    """Draw the how-it-works page from method to refinement history, with whole-year results of the run and the cached runs of other years."""
    runs = [ctx.run, *[ref.run for ref in ctx.references]]
    table = pd.DataFrame({run.name: summary(run) for run in runs})
    filled = facts(ctx.run, table[ctx.run.name].to_dict())
    method()
    headline_results(table, filled)
    scoring()
    limitations(filled)
    history()
