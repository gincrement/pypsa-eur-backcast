# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Shared helpers for the transmission capacities between ENTSO-E bidding zone areas.
"""

import logging

import numpy as np
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)


def border_name(bus0: str, bus1: str) -> str:
    """Return the column name of the direction from `bus0` to `bus1`."""
    return f"{bus0}->{bus1}"


def area_border(area0: str, area1: str) -> str:
    """Return the undirected name of the border between two areas."""
    return "-".join(sorted([area0, area1]))


def inter_zone_branches(n: pypsa.Network, area_to_bus: pd.Series) -> pd.DataFrame:
    """Return lines and DC links between buses of different areas with their nominal and thermal capacity, length and underwater fraction, ordered by bus name."""
    lines = n.lines.assign(
        capacity=n.lines.s_nom,
        thermal=n.lines.s_nom * n.lines.s_max_pu,
        underwater_fraction=0.0,
    )
    links = n.links.query("carrier == 'DC'").assign(
        capacity=lambda df: df.p_nom, thermal=lambda df: df.p_nom * df.p_max_pu
    )
    columns = ["bus0", "bus1", "capacity", "thermal", "length", "underwater_fraction"]
    branches = pd.concat(
        {"Line": lines[columns], "Link": links[columns]}, names=["component", "name"]
    )
    swap = branches.bus0 > branches.bus1
    branches.loc[swap, ["bus0", "bus1"]] = branches.loc[swap, ["bus1", "bus0"]].values
    branches["area0"] = branches.bus0.map(area_to_bus)
    branches["area1"] = branches.bus1.map(area_to_bus)
    unmapped = branches[["area0", "area1"]].isna().any(axis=1)
    assert not unmapped.any(), (
        f"Branches with buses without ENTSO-E area: {branches[unmapped].index.tolist()}"
    )
    return branches.query("area0 != area1")


def raise_to_physical(
    capacities: pd.DataFrame,
    sources: pd.DataFrame,
    thermal: pd.Series,
    physical_factor: float,
    proxy_sources: list[str],
    areas: list[str],
) -> pd.DataFrame:
    """Raise the capacities of directions with a source in `proxy_sources` touching `areas` to `physical_factor` times the thermal capacity of their bus pair."""
    selected = sources.source.isin(proxy_sources) & (
        sources.area_from.isin(areas) | sources.area_to.isin(areas)
    )
    names = sources.index[selected].intersection(capacities.columns)
    floor = pd.Series(
        [physical_factor * thermal[tuple(sorted(name.split("->")))] for name in names],
        index=names,
    )
    raised = capacities.copy()
    raised[names] = capacities[names].clip(lower=floor, axis=1)
    logger.info(
        f"Raised {len(names)} cross-border directions to {physical_factor} x thermal "
        f"capacity, mean capacity {capacities[names].mean().mean():.0f} -> "
        f"{raised[names].mean().mean():.0f} MW."
    )
    return raised


def apply_cross_border_capacities(
    n: pypsa.Network,
    capacities: pd.DataFrame,
    area_to_bus: pd.Series,
    sources: pd.DataFrame | None = None,
    hybrid: dict | None = None,
) -> None:
    """
    Replace lines and DC links between bidding zones by one bidirectional link per border.

    Each link gets `p_nom` as the maximum capacity over both directions and
    hourly `p_max_pu` and `p_min_pu` from the capacities of the forward and
    reverse direction. Copperplate connections and connections within one
    bidding zone are kept.

    Parameters
    ----------
    n : pypsa.Network
        Network with hourly snapshots.
    capacities : pd.DataFrame
        Hourly capacities in MW with columns ``<bus0>-><bus1>`` for both directions of each border.
    area_to_bus : pd.Series
        ENTSO-E bidding zone area per bus.
    sources : pd.DataFrame, optional
        Source and areas per direction from ``build_cross_border_capacities``, required with `hybrid`.
    hybrid : dict, optional
        ``backcast: hybrid_capacities`` settings. If enabled, the capacities of
        directions with a source in ``sources`` touching ``areas`` are raised to
        ``physical_factor`` times the thermal capacity of the branches between
        their buses.
    """
    branches = inter_zone_branches(n, area_to_bus)
    if hybrid and hybrid["enable"]:
        capacities = raise_to_physical(
            capacities,
            sources,
            branches.groupby(["bus0", "bus1"]).thermal.sum(),
            hybrid["physical_factor"],
            hybrid["sources"],
            hybrid["areas"],
        )
    borders = (
        branches.query("capacity > 0")
        .groupby(["bus0", "bus1"])
        .agg(
            length=("length", "mean"),
            underwater_fraction=("underwater_fraction", "mean"),
        )
    )
    forward = [border_name(b0, b1) for b0, b1 in borders.index]
    reverse = [border_name(b1, b0) for b0, b1 in borders.index]
    missing = pd.Index(forward + reverse).difference(capacities.columns)
    assert missing.empty, f"No cross-border capacities for {missing.tolist()}."
    capacities = capacities.reindex(n.snapshots)[forward + reverse]
    assert capacities.notna().all().all(), "Cross-border capacities miss snapshots."

    p_nom = pd.Series(
        np.maximum(capacities[forward].max().values, capacities[reverse].max().values),
        index=forward,
    )
    for c, names in branches.index.to_frame().groupby(level="component").name:
        n.remove(c, names)
    logger.info(
        f"Replacing {len(branches)} lines and links by {len(borders)} cross-border links."
    )
    n.add(
        "Link",
        forward,
        bus0=borders.index.get_level_values("bus0"),
        bus1=borders.index.get_level_values("bus1"),
        carrier="DC",
        p_nom=p_nom,
        p_max_pu=capacities[forward].div(p_nom).fillna(0.0),
        p_min_pu=-capacities[reverse].set_axis(forward, axis=1).div(p_nom).fillna(0.0),
        length=borders.length.values,
        underwater_fraction=borders.underwater_fraction.values,
    )
