# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for the generator limits and costs in scripts/lib/conventional.py."""

import numpy as np
import pandas as pd
import pypsa
import pytest
from pydantic import ValidationError

from scripts.lib.conventional import clip_p_min_pu, spread_marginal_cost
from scripts.lib.validation.config.conventional import ConventionalConfig

SNAPSHOTS = pd.date_range("2023-01-01", periods=3, freq="h")


@pytest.fixture
def network() -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(SNAPSHOTS)
    n.add("Bus", "DE00")
    n.add(
        "Generator",
        ["nuclear", "lignite"],
        bus="DE00",
        carrier=["nuclear", "lignite"],
        p_nom=[100.0, 200.0],
        p_min_pu=0.3,
        p_max_pu=pd.DataFrame(
            {"nuclear": [0.2, 0.5, 1.0], "lignite": [0.4, 0.5, 1.0]}, index=SNAPSHOTS
        ),
    )
    n.add(
        "Generator",
        "CCGT",
        bus="DE00",
        carrier="CCGT",
        p_nom=50.0,
        marginal_cost=pd.Series([80.0, 100.0, 120.0], index=SNAPSHOTS),
    )
    n.add(
        "Generator", "OCGT", bus="DE00", carrier="OCGT", p_nom=10.0, marginal_cost=150.0
    )
    return n


def test_clip_p_min_pu(network):
    clip_p_min_pu(network)

    p_min_pu = network.get_switchable_as_dense("Generator", "p_min_pu")
    np.testing.assert_allclose(p_min_pu.nuclear, [0.2, 0.3, 0.3])
    np.testing.assert_allclose(p_min_pu.lignite, 0.3)
    assert network.generators_t.p_min_pu.columns.tolist() == ["nuclear"]


@pytest.mark.parametrize(
    "spread, utilisation",
    [(0.3, 0.35), (0.5, 0.5), (0.0, 0.35)],
)
def test_spread_marginal_cost(network, spread, utilisation):
    spread_marginal_cost(network, ["CCGT", "OCGT"], spread, utilisation)

    g = network.generators.loc[["CCGT", "OCGT"]]
    linear = network.get_switchable_as_dense("Generator", "marginal_cost")[g.index]
    full = linear + 2 * g.marginal_cost_quadratic * g.p_nom
    mean = pd.Series({"CCGT": 100.0, "OCGT": 150.0})
    np.testing.assert_allclose(linear.mean(), mean * (1 - 2 * utilisation * spread))
    np.testing.assert_allclose(full.mean(), mean * (1 + 2 * (1 - utilisation) * spread))
    np.testing.assert_allclose(linear.CCGT.diff().dropna(), 20.0)
    assert network.generators.marginal_cost_quadratic.nuclear == 0.0


@pytest.mark.parametrize(
    "carriers, extendable, match",
    [
        (["CCGT", "oil"], False, "No generators"),
        (["CCGT"], True, "fixed positive p_nom"),
    ],
    ids=["missing-carrier", "extendable"],
)
def test_spread_marginal_cost_fails(network, carriers, extendable, match):
    network.generators.loc["CCGT", "p_nom_extendable"] = extendable
    with pytest.raises(AssertionError, match=match):
        spread_marginal_cost(network, carriers, 0.3, 0.35)


@pytest.mark.parametrize(
    "spread", [{"spread": -0.1}, {"utilisation": 1.1}, {"utilisation": -0.1}]
)
def test_conventional_config_rejects_invalid_spread(spread):
    with pytest.raises(ValidationError):
        ConventionalConfig(marginal_cost_spread=spread)
