# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for the backcast year configs in config/backcast."""

import re
from pathlib import Path

import pandas as pd
import pytest
import yaml
from entsoe.mappings import Area

from scripts.lib.entsoe import read_area_to_bus
from scripts.lib.validation.config import validate_config

ROOT = Path(__file__).parent.parent
YEAR_FILES = sorted(
    (ROOT / "config" / "backcast").glob("config.backcast-[0-9][0-9][0-9][0-9].yaml")
)


@pytest.fixture(params=YEAR_FILES, ids=lambda p: p.stem.split("-")[-1])
def year_config(request) -> tuple[int, dict]:
    return int(request.param.stem.split("-")[-1]), yaml.safe_load(
        request.param.read_text()
    )


def test_year_files_found():
    assert YEAR_FILES, "No backcast year configs in config/backcast"


def test_validates(year_config):
    _, config = year_config
    validate_config(config)


def test_year_settings_consistent(year_config):
    year, config = year_config
    electricity = config["electricity"]
    assert config["run"]["name"] == f"backcast-{year}"
    assert config["planning_horizons"] == [year]
    assert config["snapshots"] == {"start": f"{year}-01-01", "end": f"{year + 1}-01-01"}
    assert config["atlite"]["default_cutout"] == f"europe-{year}-sarah3-era5"
    assert electricity["estimate_renewable_capacities"]["year"] == year
    assert config["conventional"]["estimate_efficiencies"]["reference_year"] == year
    for key in ["powerplants_filter", "custom_powerplants"]:
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", electricity[key]))
        assert years == {str(year)}, f"electricity: {key} names years {years}"


def test_areas_known(year_config):
    _, config = year_config
    backcast = config["backcast"]
    area_to_bus = read_area_to_bus(
        ROOT / backcast.get("area_to_bus", "data/entsoe_area_to_bus.csv")
    )
    borders = pd.read_csv(ROOT / backcast.get("borders", "data/entsoe_borders.csv"))
    known = set(area_to_bus)
    assert known <= set(Area.__members__), "Areas unknown to entsoe-py"
    assert set(borders.area0) | set(borders.area1) <= known

    fixed = [
        a
        for border in backcast.get("fixed_border_capacities", {})
        for a in border.split("-")
    ]
    configured = {
        "exclude_price_areas": backcast.get("exclude_price_areas", []),
        "exclude_generation_areas": backcast.get("exclude_generation_areas", []),
        "fixed_border_capacities": fixed,
        "wind_calibration_exclude": backcast.get("wind_calibration_exclude", {}),
        "wind_target_twh": backcast.get("wind_target_twh", {}),
        "wind_curtailment_twh": backcast.get("wind_curtailment_twh", {}),
        "nuclear_phase_out": backcast.get("nuclear_phase_out", {}),
    }
    for key, areas in configured.items():
        unknown = sorted(set(areas) - known)
        assert not unknown, (
            f"backcast: {key} names areas outside the area map: {unknown}"
        )
