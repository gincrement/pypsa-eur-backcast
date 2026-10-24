# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for the ENTSO-E actual generation retrieval and build and the calibration in scripts/lib/backcast_calibration.py."""

import numpy as np
import pandas as pd
import pypsa
import pytest
from pydantic import ValidationError

from scripts.build_entsoe_generation import (
    hourly_carrier_generation,
    monthly_generation,
)
from scripts.lib.backcast_calibration import apply_monthly_availability, calibrate_wind
from scripts.lib.validation.config import validate_config
from scripts.lib.validation.config.backcast import BackcastConfig
from scripts.retrieve_actual_generation_entsoe import aggregated_generation

SNAPSHOTS = pd.date_range("2023-01-31 22:00", periods=4, freq="h")
AREA_TO_BUS = pd.Series({"ES00": "ES", "ESCE": "ES", "IE00": "IE_SEM"})


@pytest.fixture
def network() -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(SNAPSHOTS)
    n.add("Bus", AREA_TO_BUS.index, carrier="AC")
    wind = ["ES00 onwind", "ESCE onwind", "IE00 onwind", "IE00 offwind-ac"]
    n.add(
        "Generator",
        wind,
        bus=["ES00", "ESCE", "IE00", "IE00"],
        carrier=["onwind", "onwind", "onwind", "offwind-ac"],
        p_nom=100.0,
        p_max_pu=pd.DataFrame(0.5, index=SNAPSHOTS, columns=wind),
    )
    n.add(
        "Generator",
        ["ES00 nuclear", "ESCE nuclear"],
        bus=["ES00", "ESCE"],
        carrier="nuclear",
        p_nom=[100.0, 300.0],
        p_max_pu=0.9,
    )
    return n


def frame(entries: dict[tuple[str, str, str], tuple[float, int]]) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [(a, c, pd.Timestamp(m)) for (a, c, m) in entries],
        names=["area", "carrier", "month"],
    )
    return pd.DataFrame(list(entries.values()), index, ["generation", "hours"])


def test_aggregated_generation_drops_consumption_and_converts_to_utc():
    index = pd.date_range("2023-12-31 23:00", periods=3, freq="h", tz="Europe/Berlin")
    columns = pd.MultiIndex.from_tuples(
        [
            ("Solar", "Actual Aggregated"),
            ("Solar", "Actual Consumption"),
            ("Nuclear", "Actual Aggregated"),
        ]
    )
    raw = pd.DataFrame(np.arange(9.0).reshape(3, 3), index=index, columns=columns)
    result = aggregated_generation(raw, pd.Timestamp("2024-01-01", tz="UTC"))
    assert result.columns.tolist() == ["Solar", "Nuclear"]
    assert result.index.equals(index[:2].tz_convert("UTC"))


def test_hourly_carrier_generation_maps_production_types():
    index = pd.date_range("2023-01-01", periods=8, freq="15min", tz="UTC")
    raw = pd.DataFrame(
        {
            "Fossil Brown coal/Lignite": [4.0] * 8,
            "Fossil Oil shale": [np.nan] * 4 + [2.0] * 4,
            "Wind Offshore": [1.0] * 8,
            "Other": [9.0] * 8,
        },
        index=index,
    )
    result = hourly_carrier_generation(raw)
    expected = pd.DataFrame(
        {"lignite": [4.0, 6.0], "offwind": [1.0, 1.0]},
        index=pd.date_range("2023-01-01", periods=2, freq="h"),
    )
    pd.testing.assert_frame_equal(result, expected, check_freq=False)


def test_monthly_generation_sums_within_snapshots():
    hourly = pd.DataFrame(
        {
            "nuclear": [1.0, 2.0, np.nan, 4.0, 8.0],
            "onwind": [5.0, np.nan, np.nan, 3.0, np.nan],
        },
        index=pd.date_range("2023-01-31 21:00", periods=5, freq="h"),
    )
    result = monthly_generation({"ES": hourly}, SNAPSHOTS)
    expected = frame(
        {
            ("ES", "nuclear", "2023-01-01"): (2.0, 1),
            ("ES", "nuclear", "2023-02-01"): (12.0, 2),
            ("ES", "onwind", "2023-02-01"): (3.0, 1),
        }
    )
    pd.testing.assert_frame_equal(result, expected, check_index_type=False)


ES_FULL = {
    ("ES", "onwind", "2023-01-01"): (40.0, 2),
    ("ES", "onwind", "2023-02-01"): (40.0, 2),
}
ES_JAN = {("ES", "onwind", "2023-01-01"): (40.0, 2)}
ES_JAN_PARTIAL = {("ES", "onwind", "2023-01-01"): (20.0, 1)}
IE_JAN = {("IE_SEM", "onwind", "2023-01-01"): (100.0, 2)}


@pytest.mark.parametrize(
    "observed, curtailment, expected",
    [
        (ES_FULL | IE_JAN, {}, {"ES00 onwind": 0.1, "IE00 onwind": 0.5}),
        (ES_JAN | IE_JAN, {}, {"ES00 onwind": 0.1}),
        (ES_JAN_PARTIAL, {}, {"ES00 onwind": 0.1, "IE00 onwind": 0.5}),
        (ES_FULL, {"ES": 0.00008}, {"ES00 onwind": 0.2}),
        (ES_FULL | IE_JAN, {"IE_SEM": 0.001}, {"IE00 onwind": 1.0}),
        (ES_FULL | IE_JAN, {"IE_SEM": {"onwind": 0.00005}}, {"IE00 onwind": 0.75}),
    ],
    ids=[
        "all",
        "observed-months",
        "partial-month",
        "curtailment",
        "clip",
        "per-carrier",
    ],
)
def test_calibrate_wind(network, observed, curtailment, expected):
    calibrate_wind(network, frame(observed), AREA_TO_BUS, curtailment)
    p_max_pu = network.generators_t.p_max_pu
    for gen, value in expected.items():
        assert np.allclose(p_max_pu[gen], value)
    assert np.allclose(p_max_pu["ESCE onwind"], p_max_pu["ES00 onwind"])
    assert np.allclose(p_max_pu["IE00 offwind-ac"], 0.5)


@pytest.mark.parametrize(
    "observed, curtailment, match",
    [
        ({("ES", "nuclear", "2023-01-01"): (1.0, 2)}, {"ES": 1.0}, "without actual"),
        (ES_FULL, {"ES": {"offwind": 1.0}}, "without actual"),
        ({("ES", "offwind", "2023-01-01"): (1.0, 2)}, {}, "offwind generation in ES"),
    ],
    ids=["curtailment-no-wind", "curtailment-no-carrier", "no-generators"],
)
def test_calibrate_wind_raises(network, observed, curtailment, match):
    with pytest.raises(AssertionError, match=match):
        calibrate_wind(network, frame(observed), AREA_TO_BUS, curtailment)


@pytest.mark.parametrize(
    "observed, expected",
    [
        (
            {
                ("ES", "nuclear", "2023-01-01"): (400.0, 2),
                ("ES", "nuclear", "2023-02-01"): (2000.0, 2),
            },
            [0.5, 0.5, 1.0, 1.0],
        ),
        ({("ES", "nuclear", "2023-01-01"): (400.0, 1)}, [0.5, 0.5, 0.0, 0.0]),
        ({("IE_SEM", "onwind", "2023-01-01"): (1.0, 2)}, [0.9] * 4),
    ],
    ids=["clipped", "missing-month", "unpublished-area"],
)
def test_apply_monthly_availability(network, observed, expected):
    apply_monthly_availability(network, frame(observed), AREA_TO_BUS, "nuclear", {})
    p_max_pu = network.get_switchable_as_dense("Generator", "p_max_pu")
    for gen in ["ES00 nuclear", "ESCE nuclear"]:
        np.testing.assert_allclose(p_max_pu[gen], expected)


def test_apply_monthly_availability_fails_without_data(network):
    observed = frame({("ES", "onwind", "2023-01-01"): (1.0, 2)})
    with pytest.raises(AssertionError, match="No actual nuclear generation"):
        apply_monthly_availability(network, observed, AREA_TO_BUS, "nuclear", {})


@pytest.mark.parametrize(
    "config",
    [
        {"wind_curtailment_twh": {"DE_LU": -1.0}},
        {"wind_curtailment_twh": {"DE_LU": {"solar": 1.0}}},
        {"hybrid_capacities": {"sources": ["offered_capacity"]}},
        {"hybrid_capacities": {"physical_factor": 0.0}},
    ],
)
def test_backcast_config_rejects_invalid_calibration(config):
    with pytest.raises(ValidationError):
        BackcastConfig(**config)


def test_backcast_config_accepts_per_carrier_curtailment():
    curtailment = {"DE_LU": {"onwind": 6.0, "offwind": 4.0}, "DK_1": 1.0}
    config = BackcastConfig(wind_curtailment_twh=curtailment)
    assert config.wind_curtailment_twh == curtailment


@pytest.mark.parametrize("foresight", ["myopic", "perfect"])
def test_backcast_calibration_requires_overnight_foresight(foresight):
    backcast = {"calibrate_wind": True}
    validate_config({"backcast": backcast})
    with pytest.raises(ValueError, match="require foresight: overnight"):
        validate_config({"foresight": foresight, "backcast": backcast})


def test_apply_monthly_availability_pools_mapped_areas(network):
    network.add("Generator", "IE00 nuclear", bus="IE00", carrier="nuclear", p_nom=400.0)
    observed = frame({("ES", "nuclear", "2023-01-01"): (800.0, 2)})
    apply_monthly_availability(
        network, observed, AREA_TO_BUS, "nuclear", {"IE_SEM": "ES"}
    )
    p_max_pu = network.generators_t.p_max_pu
    for gen in ["ES00 nuclear", "ESCE nuclear", "IE00 nuclear"]:
        np.testing.assert_allclose(p_max_pu[gen], [0.5, 0.5, 0.0, 0.0])
