# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Loaders of the dashboard extracts and research variants and pure derived frames.
"""

import calendar
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from scripts.evaluate_price_backcast import align_prices, price_metrics
from scripts.lib.backcast_data import decile_errors, monthly_mae, top_errors
from scripts.lib.backcast_plotting import (
    area_prices,
    representative_buses,
    select_weeks,
)

ROOT = Path(__file__).parent.parent
CACHE = Path(__file__).parent / "cache"
VARIANTS = ROOT / "dev/day-ahead-backcast/phase3"
DEFAULT_YEAR = 2023
MONTH_NAMES = list(calendar.month_name)[1:]
PERIODS = ["year", *MONTH_NAMES, "winter week", "summer week", "custom"]
AGGREGATIONS = {"hourly": None, "daily": "D"}
TOP_SHARE = 0.01
KPIS = {
    "MAE": ("mae", "median"),
    "MAE / mean": ("rel_mae", "median"),
    "r": ("pearson_r", "median"),
    "|bias|": ("abs_bias", "median"),
}


@dataclass(frozen=True)
class RunData:
    """Extracted prices, dispatch and flows of one solved backcast run."""

    name: str
    model: pd.DataFrame
    observed: pd.DataFrame
    load: pd.DataFrame
    generation: pd.DataFrame
    flows: pd.DataFrame
    hydro: pd.DataFrame
    marginal: pd.DataFrame
    metrics: pd.DataFrame
    monthly_bias: pd.DataFrame
    regions: dict
    area_to_bus: pd.Series
    manifest: dict
    buses: pd.Series = field(init=False)

    def __post_init__(self) -> None:
        buses = representative_buses(self.area_to_bus, self.observed.columns)
        object.__setattr__(self, "buses", buses)

    @property
    def year(self) -> int:
        """Year of the backcast."""
        return self.manifest["year"]

    @property
    def area_model(self) -> pd.DataFrame:
        """Hourly model prices per area."""
        return area_prices(self.model, self.buses)

    @property
    def area_observed(self) -> pd.DataFrame:
        """Hourly observed prices per area."""
        return area_prices(self.observed, self.buses)

    @property
    def weeks(self) -> dict[str, pd.DatetimeIndex]:
        """Winter and summer week selected by the observed prices of the first area."""
        return select_weeks(self.area_observed.iloc[:, 0])


@dataclass(frozen=True)
class Comparison:
    """Second price series compared to run A: another run or a price-only variant aligned to run A."""

    name: str
    model: pd.DataFrame
    run: RunData | None

    def area_model(self, buses: pd.Series) -> pd.DataFrame:
        """Hourly prices per area."""
        return area_prices(self.model, buses)


def load_run(path: Path) -> RunData:
    """Read the extract of a run."""
    frames = {
        name: pd.read_parquet(path / f"{name}.parquet")
        for name in ["prices", "load", "generation", "flows", "hydro", "marginal"]
    }
    frames["metrics"] = pd.read_parquet(path / "metrics.parquet")
    monthly = pd.read_parquet(path / "monthly_bias.parquet")
    frames["monthly_bias"] = monthly.set_axis(pd.PeriodIndex(monthly.index, freq="M"))
    prices = frames.pop("prices")
    return RunData(
        name=path.name,
        model=prices["model"],
        observed=prices["observed"],
        regions=json.loads((path / "regions.geojson").read_text()),
        area_to_bus=pd.read_csv(path / "area_to_bus.csv", index_col="bus").iloc[:, 0],
        manifest=json.loads((path / "manifest.json").read_text()),
        **frames,
    )


def discover_runs(cache: Path) -> list[str]:
    """List the runs with an extract in the cache."""
    return sorted(p.parent.name for p in cache.glob("*/manifest.json"))


def variant_year(path: Path) -> int:
    """Year of a variant price file from its ``_YYYY`` suffix, else the default year."""
    match = re.search(r"_(\d{4})$", path.stem)
    return int(match.group(1)) if match else DEFAULT_YEAR


def discover_variants(root: Path) -> dict[str, Path]:
    """Map variant ids ``<experiment>/<name>`` to their price files."""
    paths = sorted(root.glob("*/prices_*.csv"))
    return {f"{p.parent.name}/{p.stem.removeprefix('prices_')}": p for p in paths}


def load_variant(path: Path, observed: pd.DataFrame) -> pd.DataFrame:
    """Read the bus prices of a variant aligned to the observed prices of its year."""
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    years = set(prices.index.year)
    assert years == {variant_year(path)}, f"{path} covers {years}"
    return align_prices(prices, observed)[0]


def stale_sources(manifest: dict) -> list[str]:
    """List sources modified after the extract or removed since."""
    return [
        source
        for source, mtime in manifest["sources"].items()
        if not Path(ROOT / source).exists()
        or Path(ROOT / source).stat().st_mtime > mtime
    ]


def period_index(
    index: pd.DatetimeIndex,
    preset: str,
    weeks: dict[str, pd.DatetimeIndex],
    dates: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DatetimeIndex:
    """Select the hours of a period preset: the year, a month by name, a week or custom dates."""
    if preset == "year":
        return index
    if preset in MONTH_NAMES:
        return index[index.month == MONTH_NAMES.index(preset) + 1]
    if preset in ("winter week", "summer week"):
        return index.intersection(weeks[preset.removesuffix(" week")])
    assert preset == "custom" and dates is not None, f"Unknown preset {preset}"
    start, end = dates
    return index[(index >= start) & (index < end + pd.Timedelta(days=1))]


def aggregate(frame: pd.DataFrame | pd.Series, aggregation: str) -> pd.DataFrame:
    """Average to the aggregation frequency."""
    freq = AGGREGATIONS[aggregation]
    return frame if freq is None else frame.resample(freq).mean()


def area_metrics(model: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """Compute the price metrics per area with absolute bias, relative MAE and shares of hours at or below zero."""
    metrics = price_metrics(model, observed).rename_axis("area")
    return metrics.assign(
        abs_bias=metrics.mean_bias.abs(),
        rel_mae=metrics.mae / metrics.mean_observed,
        nonpositive_model=metrics.zero_share_model + metrics.negative_share_model,
        nonpositive_observed=metrics.zero_share_observed
        + metrics.negative_share_observed,
    )


def kpis(metrics: pd.DataFrame) -> dict[str, float]:
    """Summarise area metrics to the headline numbers."""
    return {name: metrics[col].agg(how) for name, (col, how) in KPIS.items()}


def month_errors(error: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Mean bias and MAE per area and month."""
    months = error.index.to_period("M")
    return {"bias": error.groupby(months).mean().T, "MAE": monthly_mae(error).T}


def level_errors(error: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """Mean error per area and observed price decile with the top 1 % hours."""
    top = top_errors(error, observed, TOP_SHARE).rename(f"top {TOP_SHARE:.0%}")
    return decile_errors(error, observed).join(top)


def month_hour_errors(error: pd.Series, hours: pd.Index) -> pd.DataFrame:
    """Mean error per month and hour of day."""
    return error.groupby([error.index.month, hours]).mean().unstack()


def worst_hours(profile: pd.Series) -> tuple[int, int, float]:
    """Contiguous hours around the largest mean error sharing its sign and half its size, with their mean."""
    peak = profile.abs().idxmax()
    strong = (profile * profile[peak] > 0) & (profile.abs() >= profile.abs()[peak] / 2)
    first = last = peak
    while first - 1 in strong.index and strong[first - 1]:
        first -= 1
    while last + 1 in strong.index and strong[last + 1]:
        last += 1
    return first, last, profile.loc[first:last].mean()


def tail_shares(prices: pd.Series, levels: tuple[float, float] = (0.0, 150.0)) -> dict:
    """Shares of hours at or below the lower and above the upper price level."""
    low, high = levels
    return {
        f"≤ {low:.0f}": (prices <= low).mean(),
        f"> {high:.0f}": (prices > high).mean(),
    }


def source_generation(generation: pd.DataFrame, source: str, area: str) -> pd.DataFrame:
    """Hourly generation per carrier of an area and source, empty if the source lacks the area."""
    frame = generation[source]
    if area not in frame.columns.get_level_values("area"):
        return pd.DataFrame(index=frame.index)
    return frame[area]


def carrier_errors(
    generation: pd.DataFrame, area: str, hours: pd.DatetimeIndex
) -> pd.DataFrame:
    """Model and ENTSO-E energy in TWh per carrier of an area with absolute and relative error."""
    energy = {
        source: source_generation(generation, source, area).reindex(hours).sum() / 1e6
        for source in ["model", "entsoe"]
    }
    table = pd.DataFrame(energy).fillna(0.0)
    table = table[(table != 0).any(axis=1)]
    table["error"] = table.model - table.entsoe
    table["error %"] = 100 * table.error / table.entsoe.where(table.entsoe > 0)
    return table


def border_positions(
    flows: pd.DataFrame, area: str, hours: pd.DatetimeIndex
) -> pd.DataFrame:
    """Hourly export of an area per border for model and scheduled with the NTC in both directions."""
    columns = {}
    for border in flows["model"].columns:
        a, b = border.split("->")
        if area not in (a, b):
            continue
        sign, other = (1.0, b) if a == area else (-1.0, a)
        for source in ["model", "scheduled"]:
            if border in flows[source]:
                columns[(other, source)] = sign * flows[source][border]
        for key, direction in [
            ("ntc out", f"{area}->{other}"),
            ("ntc in", f"{other}->{area}"),
        ]:
            if direction in flows["ntc"]:
                columns[(other, key)] = flows["ntc"][direction]
    frame = pd.DataFrame(columns).reindex(hours)
    frame.columns.names = ["neighbour", "series"]
    return frame


def marginal_shares(marginal: pd.Series) -> pd.Series:
    """Share of hours per price-setting carrier."""
    return marginal.astype(str).value_counts(normalize=True).rename("share")
