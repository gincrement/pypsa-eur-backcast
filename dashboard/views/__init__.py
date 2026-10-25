# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Pages of the backcast dashboard in story order with the question each answers."""

QUESTIONS = {
    "Report": "How does the backcast work, how good is it and where does it fall short?",
    "Overview": "How good is the backcast, and in which bidding zones does it fail?",
    "Error patterns": "When does it fail: in which months, hours and price levels?",
    "Zone detail": "How do modelled and observed prices compare in {area}?",
    "Dispatch and flows": "Why: what sets the price in {area} and how does it trade?",
    "Compare runs": "What changed between {run} and {compare}?",
}
PAGES = list(QUESTIONS)
RUN = ["run_a", "run_b", "period"]
CONTROLS = {
    "Report": [],
    "Overview": RUN,
    "Error patterns": ["run_a", "period"],
    "Zone detail": [*RUN, "area", "agg"],
    "Dispatch and flows": ["run_a", "area", "period", "agg"],
    "Compare runs": [*RUN, "area", "agg"],
}
