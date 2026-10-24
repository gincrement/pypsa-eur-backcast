# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Data transforms of the day-ahead price backcast shared by the static figures and the dashboard.
"""

import numpy as np
import pandas as pd
import pypsa

from scripts.lib.backcast_plotting import CARRIER_GROUPS, STACK, cet_hour

MIN_COVERAGE = 0.95
MARGINAL_TOLERANCE = 1.0
NO_MARGINAL = "none"


def monthly_mae(error: pd.DataFrame) -> pd.DataFrame:
    """Compute the monthly mean absolute error per column."""
    return error.abs().groupby(error.index.to_period("M")).mean()


def decile_errors(error: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """Compute the mean error per area and observed price decile of that area."""
    deciles = observed.rank(pct=True).mul(10).apply(np.ceil).clip(1, 10)
    return pd.DataFrame(
        {area: error[area].groupby(deciles[area]).mean() for area in error}
    ).T.set_axis(range(1, 11), axis=1)


def top_errors(error: pd.DataFrame, observed: pd.DataFrame, top: float) -> pd.Series:
    """Compute the mean error per area in the hours with the highest observed prices."""
    return error.where(observed.ge(observed.quantile(1 - top))).mean()


def hourly_errors(error: pd.DataFrame, months: list[int]) -> pd.DataFrame:
    """Compute the mean error per area and hour of day (CET) in the months."""
    selected = error[error.index.month.isin(months)]
    return selected.groupby(cet_hour(selected.index)).mean().T


def stacked_generation(
    generation: pd.DataFrame, index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Select the hourly generation of one area in GW with the carrier stack as columns."""
    return generation.reindex(index=index, columns=STACK).fillna(0.0) / 1e3


def net_exchanges(exchanges: pd.DataFrame) -> pd.DataFrame:
    """Net scheduled exchanges per alphabetically oriented area border with sufficient coverage of both directions."""
    pairs = {tuple(sorted(c.split("->"))) for c in exchanges.columns}
    net = {}
    for a, b in sorted(pairs):
        forward = exchanges.get(f"{a}->{b}", pd.Series(np.nan, exchanges.index))
        backward = exchanges.get(f"{b}->{a}", pd.Series(np.nan, exchanges.index))
        flow = forward - backward
        if flow.notna().mean() >= MIN_COVERAGE:
            net[f"{a}->{b}"] = flow
    return pd.DataFrame(net, index=exchanges.index)


def area_capacities(capacities: pd.DataFrame, area_to_bus: pd.Series) -> pd.DataFrame:
    """Sum directed bus-to-bus transfer capacities to directed area borders ``A->B``, dropping intra-area pairs."""
    pairs = capacities.columns.str.split("->", expand=True)
    area0 = pairs.get_level_values(0).map(area_to_bus)
    area1 = pairs.get_level_values(1).map(area_to_bus)
    keep = (area0 != area1) & area0.notna() & area1.notna()
    names = [f"{a}->{b}" for a, b in zip(area0[keep], area1[keep])]
    return capacities.loc[:, keep].T.groupby(names).sum().T


def closest_group(
    cost: np.ndarray, groups: np.ndarray, price: np.ndarray, tolerance: float
) -> np.ndarray:
    """Return per hour the group of the column with cost closest to the price within the tolerance, NaN cost marking units not dispatching."""
    distance = np.nan_to_num(np.abs(cost - price[:, None]), nan=np.inf)
    best = distance.argmin(axis=1)
    found = distance[np.arange(len(price)), best] <= tolerance
    return np.where(found, groups[best], NO_MARGINAL)


def marginal_carriers(
    n: pypsa.Network, tolerance: float = MARGINAL_TOLERANCE
) -> pd.DataFrame:
    """Assign each AC bus and hour the carrier group of the dispatching generator with marginal cost closest to the bus price within the tolerance, preferring generators at the bus over all others."""
    buses = n.buses.query("carrier == 'AC'").index
    prices = n.buses_t.marginal_price[buses]
    generators = n.generators
    cost = n.get_switchable_as_dense("Generator", "marginal_cost")[generators.index]
    cost = cost.where(n.generators_t.p[generators.index] > 1e-3).to_numpy()
    groups = generators.carrier.replace(CARRIER_GROUPS).to_numpy()
    carriers = {}
    for bus in buses:
        price = prices[bus].to_numpy()
        local = (generators.bus == bus).to_numpy()
        found = closest_group(cost[:, local], groups[local], price, tolerance)
        anywhere = closest_group(cost, groups, price, tolerance)
        carriers[bus] = np.where(found == NO_MARGINAL, anywhere, found)
    return pd.DataFrame(carriers, index=prices.index)
