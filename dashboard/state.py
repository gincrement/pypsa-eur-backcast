# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Sidebar state of the dashboard, persisted in the session and the URL query parameters.
"""

from typing import Any

import streamlit as st

from dashboard.views import PAGES

DEFAULTS: dict[str, Any] = {
    "page": PAGES[0],
    "run_a": "",
    "run_b": "",
    "area": "DE_LU",
    "period": "year",
    "month": 1,
    "start": "",
    "end": "",
    "agg": "auto",
}
PENDING = "_pending"


def parse(key: str, value: str) -> Any:
    """Convert a query parameter to the type of its default."""
    return type(DEFAULTS[key])(value)


def init_state() -> None:
    """Fill the session state from the query parameters or defaults once and apply requested updates."""
    for key, default in DEFAULTS.items():
        if key not in st.session_state:
            params = st.query_params
            st.session_state[key] = (
                parse(key, params[key]) if key in params else default
            )
    st.session_state.update(st.session_state.pop(PENDING, {}))


def request(**updates: Any) -> None:
    """Set state keys bound to widgets on the next run and rerun."""
    st.session_state[PENDING] = updates
    st.rerun()


def sync_query_params() -> None:
    """Write the non-default state to the query parameters."""
    state = {k: st.session_state[k] for k in DEFAULTS}
    st.query_params.from_dict({k: str(v) for k, v in state.items() if v != DEFAULTS[k]})


def follow_selection(key: str, selection: tuple | None, **updates: Any) -> None:
    """Apply the updates once when the selection of a chart or table changes."""
    last = f"_selection_{key}"
    if selection == st.session_state.get(last):
        return
    st.session_state[last] = selection
    if selection is not None:
        request(**updates)
