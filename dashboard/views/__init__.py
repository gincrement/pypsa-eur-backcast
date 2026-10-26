# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Pages of the backcast dashboard in story order with the question each answers."""

QUESTIONS = {
    "Overview": "How does the backcast work, how good is it, and where and when does it fail?",
    "Bidding zone": "How do modelled prices and dispatch in {area} compare with ENTSO-E?",
    "Cross border flows": "How do the modelled exchanges of {area} compare with the ENTSO-E scheduled exchanges?",
    "Limitations": "Where does the backcast fall short, and which refinements were tested?",
}
PAGES = list(QUESTIONS)
ZONE = ["run_a", "area", "period", "agg"]
CONTROLS = {
    "Overview": ["run_a", "period"],
    "Bidding zone": ZONE,
    "Cross border flows": ZONE,
    "Limitations": ["run_a"],
}
