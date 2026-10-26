# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Limitations: where the backcast falls short with evidence from the selected run, and the refinement history.
"""

import streamlit as st

from dashboard import narrative
from dashboard.views.common import Context
from dashboard.views.method import facts, summary


def render(ctx: Context) -> None:
    """Draw the limitations table filled from the run and the refinement history."""
    st.subheader("Limitations")
    st.markdown(narrative.LIMITATIONS.format(**facts(ctx.run, summary(ctx.run))))
    st.subheader("Refinement history")
    st.markdown(narrative.HISTORY)
