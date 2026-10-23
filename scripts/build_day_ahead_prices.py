# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Builds hourly observed day-ahead electricity prices per bus of the bidding zone clustered network.

The ENTSO-E prices per bidding zone are converted to UTC, resampled to hourly
means and restricted to the snapshots. Gaps of up to
``MAX_INTERPOLATE_PRICE_GAP_HOURS`` are interpolated, gaps up to
``backcast: max_price_gap_hours`` (or the area specific value in
``backcast: max_price_gap_hours_by_area``) are kept as missing values and longer
gaps raise an error. The bidding zones are mapped to the
buses with ``data/entsoe_area_to_bus.csv``, where several buses may share one
bidding zone. Buses of areas listed in ``backcast: exclude_price_areas`` are
omitted.
"""

import logging

import pandas as pd
import pypsa

from scripts._helpers import configure_logging, get_snapshots, set_scenario_config
from scripts.lib.entsoe import read_area_to_bus
from scripts.lib.timeseries import fill_short_gaps
from scripts.lib.validation.config.backcast import MAX_INTERPOLATE_PRICE_GAP_HOURS

logger = logging.getLogger(__name__)


def to_hourly_utc(prices: pd.Series) -> pd.Series:
    """Convert timezone-aware prices to hourly means with naive UTC timestamps."""
    return prices.tz_convert("UTC").resample("h").mean().tz_localize(None)


def map_areas_to_buses(
    prices: pd.DataFrame,
    area_to_bus: pd.Series,
    buses: pd.Index,
    exclude: list[str],
) -> pd.DataFrame:
    """Assign bidding zone prices to buses, omitting buses of excluded areas."""
    missing = buses.difference(area_to_bus.index)
    assert missing.empty, f"Buses without ENTSO-E area: {missing.tolist()}"
    unknown = prices.columns.difference(area_to_bus)
    assert unknown.empty, f"Unknown ENTSO-E areas: {unknown.tolist()}"
    bus_areas = area_to_bus.loc[buses].loc[lambda s: ~s.isin(exclude)]
    return prices[bus_areas.to_numpy()].set_axis(bus_areas.index, axis=1)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_day_ahead_prices")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    snapshots = get_snapshots(
        snakemake.params.snapshots, snakemake.params.drop_leap_day
    )
    area_to_bus = read_area_to_bus(snakemake.input.area_to_bus)
    buses = pypsa.Network(snakemake.input.network).buses.query("carrier == 'AC'").index

    raw = [
        pd.read_csv(fn, index_col=0, parse_dates=True).squeeze("columns")
        for fn in snakemake.input.prices
    ]
    hourly = pd.concat(
        [to_hourly_utc(s) for s in raw], keys=[s.name for s in raw]
    ).unstack(level=0)

    prices = fill_short_gaps(
        hourly.reindex(snapshots),
        MAX_INTERPOLATE_PRICE_GAP_HOURS,
        snakemake.params.max_price_gap_hours,
        snakemake.params.max_price_gap_hours_by_area,
    )
    prices = map_areas_to_buses(
        prices, area_to_bus, buses, snakemake.params.exclude_price_areas
    )

    prices.to_csv(snakemake.output.csv)
