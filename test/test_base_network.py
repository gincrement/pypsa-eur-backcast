# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Tests the functionalities of scripts/base_network.py.
"""

import pathlib
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import pytest
from shapely.geometry import Point, box

sys.path.append("./scripts")

from scripts.base_network import (
    _get_country,
    _get_linetype_by_voltage,
    _get_oid,
    _load_buses,
    _load_converters_from_eg,
    _load_converters_from_raw,
    _load_lines,
    _load_links_from_eg,
    _load_links_from_raw,
    _load_transformers,
    _reconnect_crimea,
    _set_electrical_parameters_converters,
    _set_electrical_parameters_lines_eg,
    _set_electrical_parameters_lines_raw,
    _set_electrical_parameters_links_raw,
    bidding_zone_shapes,
    build_admin_shapes,
)

path_cwd = pathlib.Path.cwd()


@pytest.mark.parametrize(
    "column_name, expected",
    [
        ("tags", pd.Series(["NG", "CH", "AU"])),
        ("other", pd.Series([np.nan, np.nan, np.nan])),
    ],
)
def test_get_country(column_name, expected):
    """
    Verify what returned by _get_country()
    """
    data_list = [['"country"=>"NG"'], ['"country"=>"CH"'], ['"country"=>"AU"']]
    df_exercise = pd.DataFrame(data_list, columns=[column_name])
    output_series = _get_country(df_exercise)
    comparison_series = output_series.compare(expected)
    assert comparison_series.size == 0


def test_get_linetype_by_voltage(config):
    """
    Verify what returned by _get_linetype_by_voltage.
    """

    reference_list = [
        "Al/St 240/40 2-bundle 220.0",
        "Al/St 240/40 3-bundle 300.0",
        "Al/St 240/40 3-bundle 300.0",
        "Al/St 240/40 4-bundle 380.0",
        "Al/St 240/40 4-bundle 380.0",
        "Al/St 240/40 4-bundle 380.0",
        "Al/St 560/50 4-bundle 750.0",
    ]

    v_nom_list = [
        220.0,
        300.0,
        330.0,
        380.0,
        400.0,
        500.0,
        750.0,
    ]

    line_type_list = []

    for v_nom in v_nom_list:
        line_type_list.append(_get_linetype_by_voltage(v_nom, config["lines"]["types"]))
    assert len(line_type_list) == len(reference_list)
    assert all([x == y for x, y in zip(line_type_list, reference_list)])


@pytest.mark.parametrize(
    "column_name, expected",
    [
        ("tags", pd.Series(["12", "12345", "9876654"])),
        ("other", pd.Series([np.nan, np.nan, np.nan])),
    ],
)
def test_get_oid(column_name, expected):
    """
    Verify what returned by _get_oid()
    """
    data_list = [['"oid"=>"12"'], ['"oid"=>"12345"'], ['"oid"=>"9876654"']]
    df_exercise = pd.DataFrame(data_list, columns=[column_name])
    output_series = _get_oid(df_exercise)
    comparison_series = output_series.compare(expected)
    assert comparison_series.size == 0


def test_load_buses(buses_dataframe, config, italy_shape, tmpdir):
    """
    Verify what returned by _load_buses.
    """
    df_buses_reference = pd.DataFrame(
        {
            "bus_id": ["5231", "5232"],
            "v_nom": [380.0, 380.0],
            "symbol": ["Substation", "Substation"],
            "under_construction": [False, False],
            "x": [6.8884, 6.8894],
            "y": [45.6783, 45.6793],
            "country": ["IT", "IT"],
            "geometry": ["POINT (6.8884 45.6783)", "POINT (6.8894 45.6793)"],
            "carrier": ["AC", "AC"],
        },
    )
    buses_path = pathlib.Path(tmpdir, "buses.csv")
    buses_dataframe.to_csv(buses_path, index=False)
    countries = config["countries"]
    df_buses_output = _load_buses(
        buses_path, italy_shape, countries, config
    ).reset_index()
    pathlib.Path(buses_path).unlink(missing_ok=True)
    df_comparison = df_buses_output.compare(df_buses_reference)
    assert df_comparison.empty


def test_load_converters_from_eg(
    buses_dataframe, config, converters_dataframe, italy_shape, tmpdir
):
    """
    Verify what returned by _load_converters_from_eg.
    """
    df_converters_eg_reference = pd.DataFrame(
        {
            "converter_id": "convert_5231_5232",
            "bus0": "5231",
            "bus1": "5232",
            "voltage": 380.0,
            "geometry": "LINESTRING(6.8884 45.6783 ,6.8894 45.6793)",
            "carrier": "B2B",
        },
        index=[0],
    )
    buses_path = pathlib.Path(tmpdir, "buses.csv")
    buses_dataframe.to_csv(buses_path, index=False)
    countries = config["countries"]
    df_buses = _load_buses(buses_path, italy_shape, countries, config)
    converters_path = pathlib.Path(tmpdir, "converters_exercise.csv")
    converters_dataframe.to_csv(converters_path, index=False)
    df_converters_output = (
        _load_converters_from_eg(df_buses, converters_path)
        .reset_index()
        .loc[:, ("converter_id", "bus0", "bus1", "voltage", "geometry", "carrier")]
    )
    df_converters_comparison = df_converters_output.compare(df_converters_eg_reference)
    pathlib.Path(buses_path).unlink(missing_ok=True)
    pathlib.Path(converters_path).unlink(missing_ok=True)
    assert df_converters_comparison.empty


def test_load_converters_from_raw(
    buses_dataframe, config, converters_dataframe, italy_shape, tmpdir
):
    """
    Verify what returned by _load_converters_from_raw.
    """
    df_converters_raw_reference = pd.DataFrame(
        {
            "converter_id": "convert_5231_5232",
            "bus0": "5231",
            "bus1": "5232",
            "voltage": 380.0,
            "geometry": "LINESTRING(6.8884 45.6783 ,6.8894 45.6793)",
            "carrier": "",
        },
        index=[0],
    )
    buses_path = pathlib.Path(tmpdir, "buses.csv")
    buses_dataframe.to_csv(buses_path, index=False)
    countries = config["countries"]
    df_buses = _load_buses(buses_path, italy_shape, countries, config)
    converters_path = pathlib.Path(tmpdir, "converters_exercise.csv")
    converters_dataframe.to_csv(converters_path, index=False)
    df_converters_output = (
        _load_converters_from_raw(df_buses, converters_path)
        .reset_index()
        .loc[:, ("converter_id", "bus0", "bus1", "voltage", "geometry", "carrier")]
    )
    df_converters_comparison = df_converters_output.compare(df_converters_raw_reference)
    pathlib.Path(buses_path).unlink(missing_ok=True)
    pathlib.Path(converters_path).unlink(missing_ok=True)
    assert df_converters_comparison.empty


def test_load_lines(buses_dataframe, config, italy_shape, lines_dataframe, tmpdir):
    """
    Verify what returned by _load_lines.
    """
    df_lines_reference = pd.DataFrame(
        {
            "line_id": "line_5231_5232",
            "bus0": "5231",
            "bus1": "5232",
            "v_nom": 380.0,
            "num_parallel": 1.0,
            "length": 1.0,
            "underground": True,
            "under_construction": False,
            "geometry": "LINESTRING(6.8884 45.6783 ,6.8894 45.6793)",
            "carrier": "AC",
        },
        index=[0],
    )
    buses_path = pathlib.Path(tmpdir, "buses.csv")
    buses_dataframe.to_csv(buses_path, index=False)
    countries = config["countries"]
    df_buses = _load_buses(buses_path, italy_shape, countries, config)
    lines_path = pathlib.Path(tmpdir, "lines_exercise.csv")
    lines_dataframe.to_csv(lines_path, index=False)
    df_lines_output = (
        _load_lines(df_buses, lines_path)
        .reset_index()
        .loc[
            :,
            (
                "line_id",
                "bus0",
                "bus1",
                "v_nom",
                "num_parallel",
                "length",
                "underground",
                "under_construction",
                "geometry",
                "carrier",
            ),
        ]
    )
    df_lines_comparison = df_lines_output.compare(df_lines_reference)
    pathlib.Path(buses_path).unlink(missing_ok=True)
    pathlib.Path(lines_path).unlink(missing_ok=True)
    assert df_lines_comparison.empty


def test_load_links_from_eg(
    buses_dataframe, config, italy_shape, links_dataframe, tmpdir
):
    """
    Verify what returned by _load_links_from_eg.
    """
    df_links_eg_reference = pd.DataFrame(
        {
            "link_id": "link_5231_5232",
            "bus0": "5231",
            "bus1": "5232",
            "voltage": 380.0,
            "p_nom": 600.0,
            "length": 1.0,
            "underground": True,
            "under_construction": False,
            "geometry": "LINESTRING(6.8884 45.6783 ,6.8894 45.6793)",
            "carrier": "DC",
        },
        index=[0],
    )
    buses_path = pathlib.Path(tmpdir, "buses.csv")
    buses_dataframe.to_csv(buses_path, index=False)
    countries = config["countries"]
    df_buses = _load_buses(buses_path, italy_shape, countries, config)
    links_path = pathlib.Path(tmpdir, "links_exercise.csv")
    links_dataframe.to_csv(links_path, index=False)
    df_links_output = (
        _load_links_from_eg(df_buses, links_path)
        .reset_index()
        .loc[
            :,
            (
                "link_id",
                "bus0",
                "bus1",
                "voltage",
                "p_nom",
                "length",
                "underground",
                "under_construction",
                "geometry",
                "carrier",
            ),
        ]
    )
    df_links_comparison = df_links_output.compare(df_links_eg_reference)
    pathlib.Path(buses_path).unlink(missing_ok=True)
    pathlib.Path(links_path).unlink(missing_ok=True)
    assert df_links_comparison.empty


def test_load_links_from_raw(
    buses_dataframe, config, italy_shape, links_dataframe, tmpdir
):
    """
    Verify what returned by _load_links_from_raw.
    """
    df_links_raw_reference = pd.DataFrame(
        {
            "link_id": "link_5231_5232",
            "bus0": "5231",
            "bus1": "5232",
            "voltage": 380.0,
            "p_nom": 600.0,
            "length": 1.0,
            "underground": True,
            "under_construction": False,
            "geometry": "LINESTRING(6.8884 45.6783 ,6.8894 45.6793)",
            "carrier": "DC",
        },
        index=[0],
    )
    buses_path = pathlib.Path(tmpdir, "buses.csv")
    buses_dataframe.to_csv(buses_path, index=False)
    countries = config["countries"]
    df_buses = _load_buses(buses_path, italy_shape, countries, config)
    links_path = pathlib.Path(tmpdir, "links_exercise.csv")
    links_dataframe.to_csv(links_path, index=False)
    df_links_output = (
        _load_links_from_raw(df_buses, links_path)
        .reset_index()
        .loc[
            :,
            (
                "link_id",
                "bus0",
                "bus1",
                "voltage",
                "p_nom",
                "length",
                "underground",
                "under_construction",
                "geometry",
                "carrier",
            ),
        ]
    )
    df_links_comparison = df_links_output.compare(df_links_raw_reference)
    pathlib.Path(buses_path).unlink(missing_ok=True)
    pathlib.Path(links_path).unlink(missing_ok=True)
    assert df_links_comparison.empty


def test_load_transformers(
    buses_dataframe, config, italy_shape, tmpdir, transformers_dataframe
):
    """
    Verify what returned by _load_transformers.
    """
    df_transformers_reference = pd.DataFrame(
        {
            "transformer_id": "transf_5231_5232",
            "bus0": "5231",
            "bus1": "5232",
            "voltage": 380.0,
            "geometry": "LINESTRING(6.8884 45.6783 ,6.8894 45.6793)",
        },
        index=[0],
    )
    buses_path = pathlib.Path(tmpdir, "buses.csv")
    buses_dataframe.to_csv(buses_path, index=False)
    countries = config["countries"]
    df_buses = _load_buses(buses_path, italy_shape, countries, config)
    transformers_path = pathlib.Path(tmpdir, "transformers_exercise.csv")
    transformers_dataframe.to_csv(transformers_path, index=False)
    df_transformers_output = (
        _load_transformers(df_buses, transformers_path)
        .reset_index()
        .loc[:, ("transformer_id", "bus0", "bus1", "voltage", "geometry")]
    )
    df_transformers_comparison = df_transformers_output.compare(
        df_transformers_reference
    )
    pathlib.Path(buses_path).unlink(missing_ok=True)
    pathlib.Path(transformers_path).unlink(missing_ok=True)
    assert df_transformers_comparison.empty


def test_reconnect_crimea(
    buses_dataframe, config, italy_shape, lines_dataframe, tmpdir
):
    """
    Verify what returned by _reconnect_crimea.
    """
    df_lines_crimea_reference = pd.DataFrame(
        {
            "index": [
                "line_5231_5232",
                "Melitopol",
                "Liubymivka left",
                "Luibymivka right",
            ],
            "bus0": ["5231", "3065", "3181", "3181"],
            "bus1": ["5232", "3057", "3055", "3057"],
            "v_nom": [380.0, 300.0, 300.0, 300.0],
            "num_parallel": [1.0, 1.0, 1.0, 1.0],
            "length": [1.0, 140.0, 120.0, 140.0],
            "underground": [True, False, False, False],
            "under_construction": [False, False, False, False],
            "geometry": [
                "LINESTRING(6.8884 45.6783 ,6.8894 45.6793)",
                np.nan,
                np.nan,
                np.nan,
            ],
            "carrier": ["AC", "AC", "AC", "AC"],
        },
        index=[0, 1, 2, 3],
    )
    buses_path = pathlib.Path(tmpdir, "buses.csv")
    buses_dataframe.to_csv(buses_path, index=False)
    countries = config["countries"]
    df_buses = _load_buses(buses_path, italy_shape, countries, config)
    lines_path = pathlib.Path(tmpdir, "lines_exercise.csv")
    lines_dataframe.to_csv(lines_path, index=False)
    df_lines = _load_lines(df_buses, lines_path).loc[
        :,
        (
            "bus0",
            "bus1",
            "v_nom",
            "num_parallel",
            "length",
            "underground",
            "under_construction",
            "geometry",
            "carrier",
        ),
    ]
    df_lines_crimea_output = _reconnect_crimea(df_lines).reset_index()
    df_lines_crimea_comparison = df_lines_crimea_output.compare(
        df_lines_crimea_reference
    )
    pathlib.Path(buses_path).unlink(missing_ok=True)
    pathlib.Path(lines_path).unlink(missing_ok=True)
    assert df_lines_crimea_comparison.empty


def test_set_electrical_parameters_lines_eg(
    buses_dataframe, config, italy_shape, lines_dataframe, tmpdir
):
    """
    Verify what returned by _set_electrical_parameters_lines_eg.
    """
    df_lines_parameters_reference = pd.DataFrame(
        {
            "line_id": "line_5231_5232",
            "bus0": "5231",
            "bus1": "5232",
            "v_nom": 380.0,
            "num_parallel": 1.0,
            "length": 1.0,
            "underground": True,
            "under_construction": False,
            "geometry": "LINESTRING(6.8884 45.6783 ,6.8894 45.6793)",
            "carrier": "AC",
            "type": "Al/St 240/40 4-bundle 380.0",
            "s_max_pu": 0.7,
        },
        index=[0],
    )
    buses_path = pathlib.Path(tmpdir, "buses.csv")
    buses_dataframe.to_csv(buses_path, index=False)
    countries = config["countries"]
    df_buses = _load_buses(buses_path, italy_shape, countries, config)
    lines_path = pathlib.Path(tmpdir, "lines_exercise.csv")
    lines_dataframe.to_csv(lines_path, index=False)
    df_lines = (
        _load_lines(df_buses, lines_path)
        .reset_index()
        .loc[
            :,
            (
                "line_id",
                "bus0",
                "bus1",
                "v_nom",
                "num_parallel",
                "length",
                "underground",
                "under_construction",
                "geometry",
                "carrier",
            ),
        ]
    )
    df_lines_output = _set_electrical_parameters_lines_eg(df_lines, config)
    pathlib.Path(buses_path).unlink(missing_ok=True)
    pathlib.Path(lines_path).unlink(missing_ok=True)
    df_lines_comparison = df_lines_output.compare(df_lines_parameters_reference)
    assert df_lines_comparison.empty


def test_set_electrical_parameters_lines_raw(
    buses_dataframe, config, italy_shape, lines_dataframe, tmpdir
):
    """
    Verify what returned by _set_electrical_parameters_lines_raw.
    """
    df_lines_parameters_reference = pd.DataFrame(
        {
            "line_id": "line_5231_5232",
            "bus0": "5231",
            "bus1": "5232",
            "v_nom": 380.0,
            "num_parallel": 1.0,
            "length": 1.0,
            "underground": True,
            "under_construction": False,
            "geometry": "LINESTRING(6.8884 45.6783 ,6.8894 45.6793)",
            "carrier": "AC",
            "dc": False,
            "type": "Al/St 240/40 4-bundle 380.0",
            "s_max_pu": 0.7,
        },
        index=[0],
    )
    buses_path = pathlib.Path(tmpdir, "buses.csv")
    buses_dataframe.to_csv(buses_path, index=False)
    countries = config["countries"]
    df_buses = _load_buses(buses_path, italy_shape, countries, config)
    lines_path = pathlib.Path(tmpdir, "lines_exercise.csv")
    lines_dataframe.to_csv(lines_path, index=False)
    df_lines = (
        _load_lines(df_buses, lines_path)
        .reset_index()
        .loc[
            :,
            (
                "line_id",
                "bus0",
                "bus1",
                "v_nom",
                "num_parallel",
                "length",
                "underground",
                "under_construction",
                "geometry",
                "carrier",
            ),
        ]
    )
    df_lines_output = _set_electrical_parameters_lines_raw(df_lines, config)
    pathlib.Path(buses_path).unlink(missing_ok=True)
    pathlib.Path(lines_path).unlink(missing_ok=True)
    df_lines_comparison = df_lines_output.compare(df_lines_parameters_reference)
    assert df_lines_comparison.empty


def test_set_electrical_parameters_links_raw(
    buses_dataframe, config, italy_shape, links_dataframe, tmpdir
):
    """
    Verify what returned by _set_electrical_parameters_links_raw.
    """
    df_links_parameters_reference = pd.DataFrame(
        {
            "link_id": "link_5231_5232",
            "bus0": "5231",
            "bus1": "5232",
            "voltage": 380.0,
            "p_nom": 600.0,
            "length": 1.0,
            "underground": True,
            "under_construction": False,
            "geometry": "LINESTRING(6.8884 45.6783 ,6.8894 45.6793)",
            "carrier": "DC",
            "p_max_pu": 1.0,
            "p_min_pu": -1.0,
            "dc": True,
        },
        index=[0],
    )
    buses_path = pathlib.Path(tmpdir, "buses.csv")
    buses_dataframe.to_csv(buses_path, index=False)
    countries = config["countries"]
    df_buses = _load_buses(buses_path, italy_shape, countries, config)
    links_path = pathlib.Path(tmpdir, "links_exercise.csv")
    links_dataframe.to_csv(links_path, index=False)
    df_links = (
        _load_links_from_eg(df_buses, links_path)
        .reset_index()
        .loc[
            :,
            (
                "link_id",
                "bus0",
                "bus1",
                "voltage",
                "p_nom",
                "length",
                "underground",
                "under_construction",
                "geometry",
                "carrier",
            ),
        ]
    )
    df_links_output = _set_electrical_parameters_links_raw(df_links, config)
    df_links_comparison = df_links_output.compare(df_links_parameters_reference)
    pathlib.Path(buses_path).unlink(missing_ok=True)
    pathlib.Path(links_path).unlink(missing_ok=True)
    assert df_links_comparison.empty


def test_set_electrical_parameters_converters(
    buses_dataframe, config, converters_dataframe, italy_shape, tmpdir
):
    """
    Verify what returned by _set_electrical_parameters_converters.
    """
    df_converters_parameters_reference = pd.DataFrame(
        {
            "converter_id": "convert_5231_5232",
            "bus0": "5231",
            "bus1": "5232",
            "voltage": 380.0,
            "geometry": "LINESTRING(6.8884 45.6783 ,6.8894 45.6793)",
            "carrier": "B2B",
            "p_max_pu": 1.0,
            "p_min_pu": -1.0,
            "p_nom": 2000,
            "under_construction": False,
            "underground": False,
            "dc": False,  # ToDo Change once we todo is addressed in base_network.py
        },
        index=[0],
    )
    buses_path = pathlib.Path(tmpdir, "buses.csv")
    buses_dataframe.to_csv(buses_path, index=False)
    countries = config["countries"]
    df_buses = _load_buses(buses_path, italy_shape, countries, config)
    converters_path = pathlib.Path(tmpdir, "converters_exercise.csv")
    converters_dataframe.to_csv(converters_path, index=False)
    df_converters = (
        _load_converters_from_eg(df_buses, converters_path)
        .reset_index()
        .loc[:, ("converter_id", "bus0", "bus1", "voltage", "geometry", "carrier")]
    )
    df_converters_output = _set_electrical_parameters_converters(df_converters, config)
    df_converters_comparison = df_converters_output.compare(
        df_converters_parameters_reference
    )
    pathlib.Path(buses_path).unlink(missing_ok=True)
    pathlib.Path(converters_path).unlink(missing_ok=True)
    assert df_converters_comparison.empty


KM = 1e3
ORIGIN = (4321e3, 3210e3)


def _km_frame(
    rows: dict[str, tuple], column: str, country: str = "XX"
) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {column: list(rows), "country": country},
        geometry=[box(*(KM * c for c in coords)) for coords in rows.values()],
        crs="EPSG:3035",
    )


@pytest.fixture
def straddling_regions() -> gpd.GeoDataFrame:
    return _km_frame({"XX1": (0, 0, 40, 20), "XX2": (40, 0, 80, 20)}, "name")


@pytest.fixture
def coastal_zones() -> gpd.GeoDataFrame:
    return _km_frame({"A": (0, 0, 30, 20), "B": (30, 0, 75, 18)}, "zone_name")


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (20, 10, "A"),
        (35, 10, "B"),
        (78, 10, "B"),
        (32, 19.5, "B"),
        (30.5, 19.9, "A"),
    ],
)
def test_bidding_zone_shapes(straddling_regions, coastal_zones, x, y, expected):
    shapes = bidding_zone_shapes(straddling_regions, coastal_zones).set_index("admin")
    assert shapes.crs == straddling_regions.crs
    assert shapes.area.sum() == pytest.approx(80 * 20 * KM**2)
    assert list(shapes.index[shapes.contains(Point(x * KM, y * KM))]) == [expected]


@pytest.mark.parametrize(
    "x, expected", [(20, "A"), (78, "B"), (90, "YY00"), (110, "YY00")]
)
def test_bidding_zone_shapes_country_without_zones(
    straddling_regions, coastal_zones, x, expected
):
    regions = pd.concat(
        [straddling_regions, _km_frame({"YY1": (80, 0, 120, 20)}, "name", "YY")]
    )
    shapes = bidding_zone_shapes(regions, coastal_zones).set_index("admin")
    assert sorted(shapes.index) == ["A", "B", "YY00"]
    assert shapes.loc["YY00", "country"] == "YY"
    assert shapes.area["YY00"] == pytest.approx(40 * 20 * KM**2)
    assert list(shapes.index[shapes.contains(Point(x * KM, 10 * KM))]) == [expected]


@pytest.fixture
def two_country_inputs(straddling_regions, coastal_zones, tmpdir):
    regions = pd.concat(
        [
            straddling_regions,
            _km_frame(
                {"YY1": (80, 0, 100, 20), "YY2": (100, 0, 120, 20)}, "name", "YY"
            ),
        ]
    ).rename(columns={"name": "index"})
    regions = regions.assign(
        level3=regions["index"], level2=regions["country"], level1=regions["country"]
    )
    zones = pd.concat(
        [coastal_zones, _km_frame({"C": (80, 0, 120, 20)}, "zone_name", "YY")]
    )
    nuts3_path = str(tmpdir / "nuts3.geojson")
    zones_path = str(tmpdir / "zones.geojson")
    for frame, path in [(regions, nuts3_path), (zones, zones_path)]:
        frame.set_geometry(frame.translate(*ORIGIN)).to_crs(epsg=4326).to_file(path)

    points = gpd.GeoSeries(
        [Point(x * KM, 10 * KM) for x in (20, 60, 90, 110)], crs="EPSG:3035"
    )
    points = points.translate(*ORIGIN).to_crs(epsg=4326)
    n = pypsa.Network()
    n.add(
        "Bus",
        ["b1", "b2", "b3", "b4"],
        x=points.x.values,
        y=points.y.values,
        country=["XX", "XX", "YY", "YY"],
    )
    return n, nuts3_path, zones_path


@pytest.mark.parametrize(
    "admin_levels, expected",
    [
        ({"level": 3, "countries": {}}, ["XX1", "XX2", "YY1", "YY2"]),
        ({"level": "bz", "countries": {"YY": 3}}, ["A", "B", "YY1", "YY2"]),
        ({"level": 3, "countries": {"XX": "bz"}}, ["A", "B", "YY1", "YY2"]),
        ({"level": "bz", "countries": {"YY1": 3}}, ["A", "B", "C", "YY1"]),
    ],
)
def test_build_admin_shapes_bidding_zones(two_country_inputs, admin_levels, expected):
    n, nuts3_path, zones_path = two_country_inputs
    shapes = build_admin_shapes(
        n, nuts3_path, "administrative", admin_levels, ["XX", "YY"], zones_path
    )
    assert sorted(shapes.index) == expected
    assert (shapes["substations"] == 1).all()
    area = shapes.to_crs(epsg=3035).area.sum()
    assert area == pytest.approx(120 * 20 * KM**2, rel=1e-6)
