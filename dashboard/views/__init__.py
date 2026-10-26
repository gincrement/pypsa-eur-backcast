# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Pages of the backcast dashboard in story order with their title and the plain question each answers."""

from dashboard.narrative import CLAIM

STORY_PAGE = "The story"
ZONE_PAGE = "Explore a country"
FLOWS_PAGE = "Trade between countries"
METHOD_PAGE = "How it works"
QUESTIONS = {
    STORY_PAGE: "If every power plant simply asked for its fuel and CO2 cost, how close would we get to the real prices?",
    ZONE_PAGE: "How close does the model get to the real prices and power plants of {area}?",
    FLOWS_PAGE: "Does {area} buy and sell power across its borders like the real market did?",
    METHOD_PAGE: "How is the model built, how is it scored, and what does it leave out?",
}
PAGES = list(QUESTIONS)
TITLES = {page: page for page in PAGES} | {STORY_PAGE: CLAIM}
ZONE_CONTROLS = ["run_a", "area", "period", "agg"]
CONTROLS = {
    STORY_PAGE: ["run_a", "period"],
    ZONE_PAGE: ZONE_CONTROLS,
    FLOWS_PAGE: ZONE_CONTROLS,
    METHOD_PAGE: ["run_a"],
}
