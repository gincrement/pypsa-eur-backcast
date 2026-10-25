# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for the backcast dashboard data layer, extract and app."""

import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import pytest
from shapely.geometry import box

from dashboard import data
from dashboard.extract import extract
from dashboard.views import PAGES
from scripts.evaluate_price_backcast import monthly_bias, price_metrics
from scripts.lib.backcast_data import (
    NO_MARGINAL,
    area_capacities,
    marginal_carriers,
    monthly_mae,
    stacked_generation,
    top_errors,
)

HOURS = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="h")
AREA_TO_BUS = pd.Series({"DE00": "DE_LU", "FR00": "FR"}, name="entsoe_area")
FILES = {
    "prices",
    "load",
    "generation",
    "flows",
    "hydro",
    "marginal",
    "metrics",
    "monthly_bias",
}


def solved_network(hours: pd.DatetimeIndex = HOURS) -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(hours)
    n.add("Bus", AREA_TO_BUS.index, carrier="AC")
    n.add("Link", "DE00->FR00", bus0="DE00", bus1="FR00", carrier="DC", p_nom=1.0)
    n.add(
        "Generator",
        ["DE00 CCGT", "DE00 solar", "FR00 nuclear", "FR00 load"],
        bus=["DE00", "DE00", "FR00", "FR00"],
        carrier=["CCGT", "solar", "nuclear", "load"],
        marginal_cost=[50.0, 0.0, 10.0, 1000.0],
    )
    n.add("Load", ["DE00 load", "FR00 load"], bus=["DE00", "FR00"])
    n.add("StorageUnit", "FR00 hydro", bus="FR00", carrier="hydro")
    day = np.tile(np.r_[np.full(12, 50.0), np.full(12, 10.5)], len(hours) // 24)
    n.buses_t.marginal_price = pd.DataFrame({"DE00": day, "FR00": 10.0}, hours)
    n.generators_t.p = pd.DataFrame(
        {"DE00 CCGT": 1.0, "DE00 solar": 2.0, "FR00 nuclear": 3.0, "FR00 load": 0.0},
        hours,
    )
    n.links_t.p0 = pd.DataFrame({"DE00->FR00": 1.0}, hours)
    n.loads_t.p = pd.DataFrame({"DE00 load": 3.0, "FR00 load": 2.0}, hours)
    n.storage_units_t.p = pd.DataFrame({"FR00 hydro": 1.0}, hours)
    n.storage_units_t.state_of_charge = pd.DataFrame({"FR00 hydro": 5.0}, hours)
    n.storage_units_t.inflow = pd.DataFrame({"FR00 hydro": 0.5}, hours)
    return n


def write_run(tmp: Path, name: str, hours: pd.DatetimeIndex) -> Path:
    n = solved_network(hours)
    observed = n.buses_t.marginal_price + np.sin(np.arange(len(hours)))[:, None]
    model = n.buses_t.marginal_price
    entsoe = pd.DataFrame(
        {("DE_LU", "gas"): 1.0, ("FR", "nuclear"): 3.0}, hours
    ).rename_axis(columns=["area", "carrier"])
    paths = {
        "network": tmp / "network.nc",
        "prices": tmp / "prices.csv",
        "generation": tmp / "generation.csv",
        "exchanges": tmp / "exchanges.csv",
        "capacities": tmp / "capacities.csv",
        "regions": tmp / "regions.geojson",
        "metrics": tmp / "metrics.csv",
        "monthly_bias": tmp / "monthly_bias.csv",
        "area_to_bus": tmp / "area_to_bus.csv",
    }
    n.export_to_netcdf(paths["network"])
    observed.to_csv(paths["prices"])
    entsoe.to_csv(paths["generation"])
    pd.DataFrame({"DE_LU->FR": 2.0, "FR->DE_LU": 1.0}, hours).to_csv(paths["exchanges"])
    pd.DataFrame({"DE00->FR00": 5.0, "FR00->DE00": 4.0}, hours).to_csv(
        paths["capacities"]
    )
    regions = gpd.GeoDataFrame(
        {"name": AREA_TO_BUS.index}, geometry=[box(6, 48, 14, 54), box(-4, 43, 6, 50)]
    )
    regions.set_crs(4326).to_file(paths["regions"], driver="GeoJSON")
    price_metrics(model, observed).to_csv(paths["metrics"])
    monthly_bias(model, observed).to_csv(paths["monthly_bias"])
    AREA_TO_BUS.rename_axis("bus").to_csv(paths["area_to_bus"])
    out = tmp / "cache" / name
    extract(paths, out, name)
    return out


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    return write_run(tmp_path_factory.mktemp("run"), "test-run", HOURS)


@pytest.fixture(scope="module")
def validation_cache(tmp_path_factory, run_dir) -> Path:
    tmp = tmp_path_factory.mktemp("validation")
    write_run(
        tmp, "test-run-2019", pd.date_range("2019-01-01", periods=len(HOURS), freq="h")
    )
    shutil.copytree(run_dir, tmp / "cache" / run_dir.name)
    return tmp / "cache"


def test_extract_writes_files_and_marginal_carriers(run_dir):
    written = {p.stem for p in run_dir.glob("*.parquet")}
    run = data.load_run(run_dir)

    assert written == FILES
    assert run.model.shape == (len(HOURS), 2)
    assert run.buses.to_dict() == {"DE_LU": "DE00", "FR": "FR00"}
    assert run.flows["ntc"].columns.tolist() == ["DE_LU->FR", "FR->DE_LU"]
    assert run.marginal["DE00"].iloc[:13].tolist() == [*["gas"] * 12, "nuclear"]
    assert set(run.marginal["FR00"]) == {"nuclear"}
    assert run.year == 2023 and data.stale_sources(run.manifest) == []


@pytest.mark.parametrize(
    "price, expected",
    [(50.4, "gas"), (10.0, "nuclear"), (0.0, "solar"), (30.0, NO_MARGINAL)],
)
def test_marginal_carriers_tolerance(price, expected):
    n = solved_network()
    n.buses_t.marginal_price["DE00"] = price

    assert set(marginal_carriers(n)["DE00"]) == {expected}


def test_area_capacities_sum_directed_area_pairs():
    capacities = pd.DataFrame(
        {"DE00->FR00": [1.0], "DE01->FR00": [2.0], "DE00->DE01": [9.0]}
    )
    area_to_bus = pd.Series({"DE00": "DE_LU", "DE01": "DE_LU", "FR00": "FR"})

    result = area_capacities(capacities, area_to_bus)

    assert result.to_dict("list") == {"DE_LU->FR": [3.0]}


def test_monthly_mae_and_top_errors():
    index = pd.date_range("2023-01-31", periods=4, freq="12h")
    error = pd.DataFrame({"A": [1.0, -3, 2, -2]}, index)
    observed = pd.DataFrame({"A": [1.0, 2, 3, 4]}, index)

    assert monthly_mae(error)["A"].tolist() == [2.0, 2.0]
    assert top_errors(error, observed, 0.25)["A"] == -2.0


def test_stacked_generation_fills_carriers_in_gw():
    generation = pd.DataFrame({"gas": [1000.0, 2000.0]}, HOURS[:2])

    stack = stacked_generation(generation, HOURS[:3])

    assert stack.shape == (3, len(stack.columns)) and stack.gas.tolist() == [1, 2, 0]


@pytest.mark.parametrize(
    "preset, kwargs, hours",
    [
        ("year", {}, len(HOURS)),
        ("February", {}, 28 * 24),
        ("winter week", {}, 168),
        ("custom", {"dates": (pd.Timestamp("2023-03-01"),) * 2}, 24),
    ],
)
@pytest.mark.parametrize("aggregation, factor", [("hourly", 1), ("daily", 24)])
def test_period_index_and_aggregate(preset, kwargs, hours, aggregation, factor):
    weeks = {"winter": HOURS[:168], "summer": HOURS[-168:]}
    index = data.period_index(HOURS, preset, weeks, **kwargs)
    prices = pd.Series(1.0, index)

    assert len(index) == hours
    assert len(data.aggregate(prices, aggregation)) == hours // factor


@pytest.mark.parametrize(
    "errors, expected",
    [
        ([0, 1, 5, 6, 4, 2, 0], (2, 4, 5.0)),
        ([-1, -8, -2, 3, 0, 0, 0], (1, 1, -8.0)),
    ],
)
def test_worst_hours_window(errors, expected):
    assert data.worst_hours(pd.Series(errors, dtype=float)) == expected


def test_discover_and_load_variants(tmp_path, run_dir):
    observed = data.load_run(run_dir).observed
    for name in ["prices_base.csv", "prices_wind_2019.csv"]:
        (tmp_path / "exp").mkdir(exist_ok=True)
        observed.to_csv(tmp_path / "exp" / name)

    variants = data.discover_variants(tmp_path)

    assert list(variants) == ["exp/base", "exp/wind_2019"]
    assert data.variant_year(variants["exp/wind_2019"]) == 2019
    pd.testing.assert_frame_equal(
        data.load_variant(variants["exp/base"], observed),
        observed,
        check_freq=False,
        check_index_type=False,
    )
    with pytest.raises(AssertionError, match="covers"):
        data.load_variant(variants["exp/wind_2019"], observed)


def test_level_and_border_tables(run_dir):
    run = data.load_run(run_dir)
    error = run.area_model - run.area_observed

    levels = data.level_errors(error, run.area_observed)
    positions = data.border_positions(run.flows, "FR", HOURS)

    assert levels.columns.tolist() == [*range(1, 11), "top 1%"]
    assert positions.loc[HOURS[0]].to_dict() == {
        ("DE_LU", "model"): -1.0,
        ("DE_LU", "scheduled"): -1.0,
        ("DE_LU", "ntc out"): 4.0,
        ("DE_LU", "ntc in"): 5.0,
    }


def app_test(cache: Path, monkeypatch, **params: str):
    streamlit = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(data, "CACHE", cache)
    monkeypatch.setattr(data, "VARIANTS", cache / "none")
    streamlit.cache_data.clear()
    app = AppTest.from_file(
        str(Path(data.__file__).parent / "app.py"), default_timeout=60
    )
    app.query_params.update(params)
    return app.run()


@pytest.mark.slow
@pytest.mark.parametrize("page", PAGES)
def test_app_renders_pages_and_keeps_state(run_dir, monkeypatch, page):
    app = app_test(run_dir.parent, monkeypatch, page=page, area="FR", period="March")

    assert not app.exception
    assert app.session_state["area"] == "FR" and app.session_state["period"] == "March"
    assert app.query_params["area"] == "FR"


@pytest.mark.slow
def test_overview_scores_validation_run_over_same_period(validation_cache, monkeypatch):
    app = app_test(
        validation_cache, monkeypatch, page="Overview", run_a="test-run", period="March"
    )

    text = " ".join(m.value for m in app.markdown)
    assert not app.exception
    assert "Validation, March 2019" in text
    assert "validation test-run-2019, March 2019" in text


@pytest.mark.slow
def test_zone_link_opens_dispatch_and_hidden_state_survives(run_dir, monkeypatch):
    app = app_test(
        run_dir.parent, monkeypatch, page="Zone detail", area="FR", agg="daily"
    )
    app.button(key="link_Dispatch and flows for FR").click().run()
    shown = [s.label for s in app.selectbox]
    app.radio(key="page").set_value("Report").run()
    hidden = len(app.selectbox)
    app.radio(key="page").set_value("Zone detail").run()

    assert not app.exception
    assert shown == ["Run A", "Bidding zone", "Period"] and hidden == 0
    assert app.session_state["area"] == "FR" and app.session_state["agg"] == "daily"


@pytest.mark.slow
def test_report_page_fills_results_and_limitations(run_dir, monkeypatch):
    app = app_test(run_dir.parent, monkeypatch, page="Report")

    text = " ".join(m.value for m in app.markdown)
    assert not app.exception
    assert [h.value for h in app.subheader][-1] == "Refinement history"
    assert app.table[0].value.columns.tolist() == ["test-run"]
    assert "{" not in text and "test-run" in text
