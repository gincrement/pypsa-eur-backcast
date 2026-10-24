# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Operational limits and costs of generators applied once the network is composed.
"""

import logging

import pypsa

logger = logging.getLogger(__name__)


def clip_p_min_pu(n: pypsa.Network) -> None:
    """Clip the minimum to the maximum availability of generators wherever ``p_min_pu`` exceeds ``p_max_pu``."""
    g = n.generators
    gens = g.index[g.p_min_pu > 0].union(n.generators_t.p_min_pu.columns)
    p_min_pu = n.get_switchable_as_dense("Generator", "p_min_pu", inds=gens)
    p_max_pu = n.get_switchable_as_dense("Generator", "p_max_pu", inds=gens)
    clipped = gens[(p_min_pu > p_max_pu).any()]
    if clipped.empty:
        return
    n.generators_t.p_min_pu[clipped] = p_min_pu[clipped].clip(upper=p_max_pu[clipped])
    logger.info(f"Clipped p_min_pu to p_max_pu for {len(clipped)} generators.")


def spread_marginal_cost(
    n: pypsa.Network, carriers: list[str], spread: float, utilisation: float
) -> None:
    """
    Let the marginal cost of the generators of `carriers` rise linearly with their output.

    With the mean marginal cost ``mc`` of a generator over all snapshots, the
    marginal cost rises from ``mc * (1 - 2 * utilisation * spread)`` at zero
    output to ``mc * (1 + 2 * (1 - utilisation) * spread)`` at ``p_nom`` by
    shifting the linear and setting the quadratic marginal cost.
    """
    g = n.generators
    missing = sorted(set(carriers) - set(g.carrier))
    assert not missing, f"No generators to spread the marginal cost of: {missing}"
    gens = g.index[g.carrier.isin(carriers)]
    invalid = gens[g.p_nom_extendable[gens] | (g.p_nom[gens] <= 0)]
    assert invalid.empty, (
        f"Marginal cost spread requires fixed positive p_nom: {invalid.tolist()}"
    )
    mean = n.get_switchable_as_dense("Generator", "marginal_cost", inds=gens).mean()
    offset = 2 * utilisation * spread * mean
    dynamic = gens.intersection(n.generators_t.marginal_cost.columns)
    g.loc[gens, "marginal_cost"] -= offset
    n.generators_t.marginal_cost[dynamic] -= offset[dynamic]
    g.loc[gens, "marginal_cost_quadratic"] = spread * mean / g.p_nom[gens]
    logger.info(
        f"Spread the marginal cost of {len(gens)} generators by {spread} around "
        f"utilisation {utilisation}."
    )
