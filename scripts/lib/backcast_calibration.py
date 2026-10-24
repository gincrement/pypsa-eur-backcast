# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Calibration of generator availabilities to the actual generation reported by the ENTSO-E Transparency Platform.
"""

import logging

import pandas as pd
import pypsa

logger = logging.getLogger(__name__)

WIND_CARRIERS = ["onwind", "offwind"]


def read_generation(path: str) -> pd.DataFrame:
    """Read the monthly generation in MWh and hours with data indexed by area, carrier and month."""
    generation = pd.read_csv(path, parse_dates=["month"])
    return generation.set_index(["area", "carrier", "month"])


def snapshot_months(n: pypsa.Network) -> pd.DatetimeIndex:
    """Return the first day of the month of each snapshot."""
    return n.snapshots.to_period("M").to_timestamp()


def area_generators(
    n: pypsa.Network, area_to_bus: pd.Series, area: str, carrier: str
) -> pd.Index:
    """Return the generators of an area whose carrier starts with `carrier`."""
    g = n.generators
    return g.index[g.bus.map(area_to_bus).eq(area) & g.carrier.str.startswith(carrier)]


def carrier_curtailment(
    curtailment: float | dict[str, float], energy: pd.Series
) -> pd.Series:
    """Return the curtailment in MWh per wind carrier, splitting a total by the actual generation."""
    if isinstance(curtailment, dict):
        return (
            pd.Series(curtailment, dtype=float).reindex(energy.index, fill_value=0.0)
            * 1e6
        )
    if not curtailment:
        return energy * 0.0
    return curtailment * 1e6 * energy / energy.sum()


def calibrate_wind(
    n: pypsa.Network,
    generation: pd.DataFrame,
    area_to_bus: pd.Series,
    curtailment_twh: dict[str, float | dict[str, float]],
) -> None:
    """
    Scale the onshore and offshore wind availability per area to the actual generation.

    Only the months with actual generation of a carrier are compared. Their
    actual energy is the generation divided by the hours with data times the
    hours of the month. The target energy adds the curtailment of the area,
    given per carrier or as a total split by the actual generation. The
    available energy of the model is taken over the same months. All generators
    of a carrier in the area get the same factor for all snapshots,
    ``p_max_pu`` is clipped at 1. Generators of carriers without actual
    generation in their area keep their availability.

    Parameters
    ----------
    n : pypsa.Network
        Network with time-varying ``p_max_pu`` of the wind generators.
    generation : pd.DataFrame
        Monthly generation in MWh and hours with data indexed by area, carrier and month.
    area_to_bus : pd.Series
        ENTSO-E area per bus.
    curtailment_twh : dict[str, float | dict[str, float]]
        Curtailed wind energy in TWh per area added to the actual generation,
        in total or per wind carrier.
    """
    weights = n.snapshot_weightings.generators
    months = snapshot_months(n)
    month_hours = weights.groupby(months).sum()
    p_max_pu = n.generators_t.p_max_pu
    observed = generation[
        generation.index.get_level_values("carrier").isin(WIND_CARRIERS)
    ]
    energy = (
        observed.generation
        / observed.hours
        * month_hours.reindex(observed.index.get_level_values("month")).values
    )
    for area in area_to_bus.unique():
        gens = {
            carrier: area_generators(n, area_to_bus, area, carrier).intersection(
                p_max_pu.columns
            )
            for carrier in WIND_CARRIERS
        }
        area_energy = (
            energy[area].groupby(level="carrier").sum()
            if area in energy.index.unique("area")
            else pd.Series(dtype=float)
        )
        value = curtailment_twh.get(area, 0.0)
        covered = (
            set(value) <= set(area_energy.index)
            if isinstance(value, dict)
            else not value or not area_energy.empty
        )
        assert covered, (
            f"Wind curtailment given for {area} without actual generation of its wind carriers."
        )
        curtailment = carrier_curtailment(value, area_energy)
        for carrier, carrier_gens in gens.items():
            if carrier not in area_energy.index:
                if not carrier_gens.empty:
                    logger.info(
                        f"No actual {carrier} generation for {area}, keeping its availability."
                    )
                continue
            mask = months.isin(energy[area][carrier].index)
            available = (
                (p_max_pu.loc[mask, carrier_gens] * n.generators.p_nom[carrier_gens])
                .mul(weights[mask], axis=0)
                .sum()
                .sum()
            )
            target = area_energy[carrier] + curtailment[carrier]
            assert available > 0, (
                f"Actual {carrier} generation in {area} without {carrier} generators."
            )
            factor = target / available
            p_max_pu[carrier_gens] = (p_max_pu[carrier_gens] * factor).clip(upper=1)
            logger.info(
                f"Calibrated {carrier} in {area} over {mask.sum()} snapshots: "
                f"{available / 1e6:.1f} -> {target / 1e6:.1f} TWh (x{factor:.2f})."
            )


def apply_monthly_availability(
    n: pypsa.Network,
    generation: pd.DataFrame,
    area_to_bus: pd.Series,
    carrier: str,
    area_map: dict[str, str],
) -> None:
    """
    Set the availability of a carrier per area and month to its actual capacity factor.

    The capacity factor is the monthly generation divided by the capacity of the
    carrier in the area and the hours of the month, clipped to [0, 1]. Months
    and hours without data count as zero generation, e.g. after a phase-out.
    Generators of areas without any published generation keep their
    availability, areas that publish generation but none of the carrier raise
    an error. Generators of the keys of `area_map` are pooled with the area
    they map to.

    Parameters
    ----------
    n : pypsa.Network
        Network with snapshots.
    generation : pd.DataFrame
        Monthly generation in MWh and hours with data indexed by area, carrier and month.
    area_to_bus : pd.Series
        ENTSO-E area per bus.
    carrier : str
        Generator carrier, e.g. ``nuclear``.
    area_map : dict[str, str]
        Area whose generators are calibrated with the actual generation of another area.
    """
    g = n.generators.query("carrier == @carrier")
    areas = g.bus.map(area_to_bus).replace(area_map)
    published = areas.isin(generation.index.unique("area"))
    if not published.all():
        logger.info(
            f"No actual generation published for {sorted(areas[~published].unique())}, "
            f"keeping their {carrier} availability."
        )
    areas = areas[published]
    observed = generation.generation[
        generation.index.get_level_values("carrier") == carrier
    ]
    observed = observed.droplevel("carrier").unstack("area")
    missing = pd.Index(areas.unique()).difference(observed.columns)
    assert missing.empty, (
        f"No actual {carrier} generation for areas with {carrier} generators: {missing.tolist()}"
    )
    weights = n.snapshot_weightings.generators
    months = snapshot_months(n)
    hours = weights.groupby(months).sum()
    capacity = g.p_nom[areas.index].groupby(areas).sum()
    observed = observed.reindex(index=hours.index, columns=capacity.index).fillna(0.0)
    capacity_factor = observed.div(hours, axis=0).div(capacity).clip(0, 1)
    n.generators_t.p_max_pu[areas.index] = (
        capacity_factor.reindex(months)[areas]
        .set_axis(n.snapshots)
        .set_axis(areas.index, axis=1)
    )
    for area, cf in capacity_factor.mean().items():
        logger.info(f"Monthly {carrier} availability in {area}: mean {cf:.2f}.")
