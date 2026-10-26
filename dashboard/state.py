# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Sidebar state of the dashboard, persisted in the session and the URL query parameters.
"""

from typing import Any

import streamlit as st

from dashboard.data import FOCUS_AREA
from dashboard.views import PAGES

DEFAULTS: dict[str, Any] = {
    "page": PAGES[0],
    "run_a": "",
    "area": FOCUS_AREA,
    "period": "year",
    "start": "",
    "end": "",
    "agg": "auto",
}
PENDING = "_pending"


def parse(key: str, value: str) -> Any:
    """Convert a query parameter to the type of its default."""
    return type(DEFAULTS[key])(value)


def init_state() -> None:
    """Fill the session state from the query parameters or defaults once, keep the state of hidden widgets and apply requested updates."""
    params = st.query_params
    for key, default in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = (
                parse(key, params[key]) if key in params else default
            )
        st.session_state[key] = st.session_state[key]
    st.session_state.update(st.session_state.pop(PENDING, {}))


def request(**updates: Any) -> None:
    """Set state keys bound to widgets on the next run and rerun."""
    st.session_state[PENDING] = updates
    st.rerun()


def sync_query_params() -> None:
    """Write the non-default state to the query parameters."""
    state = {k: st.session_state[k] for k in DEFAULTS}
    st.query_params.from_dict({k: str(v) for k, v in state.items() if v != DEFAULTS[k]})


def open_page(page: str, **updates: Any) -> None:
    """Switch to a page with updated state, as a widget callback."""
    st.session_state.update(page=page, **updates)


def follow_selection(key: str, selection: Any, **updates: Any) -> None:
    """Apply the updates once when the selection of a chart changes to a new value, None clears it."""
    last = f"_selection_{key}"
    if selection == st.session_state.get(last):
        return
    st.session_state[last] = selection
    if selection is not None:
        request(**updates)
