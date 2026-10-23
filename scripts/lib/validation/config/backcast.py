# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Backcast configuration.

See docs in https://pypsa-eur.readthedocs.io/en/latest/configuration/#backcast_cf
"""

from typing import Annotated, Literal

from pydantic import Field, field_validator

from scripts.lib.validation.config._base import ConfigModel

MAX_INTERPOLATE_PRICE_GAP_HOURS = 3

GapHours = Annotated[int, Field(ge=MAX_INTERPOLATE_PRICE_GAP_HOURS)]


class BackcastConfig(ConfigModel):
    """Configuration for `backcast` settings."""

    exclude_price_areas: list[str] = Field(
        default_factory=list,
        description="ENTSO-E bidding zone areas without published day-ahead prices for the whole snapshot range, e.g. `GB` after 2020, `BA` and `XK`, or `AL`, `ME` and `MK` whose markets started during 2023. Buses mapped to these areas in `data/entsoe_area_to_bus.csv` are omitted from the observed day-ahead prices.",
    )
    max_price_gap_hours: GapHours = Field(
        MAX_INTERPOLATE_PRICE_GAP_HOURS,
        description=f"Longest gap in hours of observed day-ahead prices that is tolerated. Gaps of up to {MAX_INTERPOLATE_PRICE_GAP_HOURS} hours are interpolated, longer gaps up to this limit are kept as missing values, even longer gaps raise an error.",
    )
    max_price_gap_hours_by_area: dict[str, GapHours] = Field(
        default_factory=dict,
        description="Longest tolerated gap in hours of observed day-ahead prices per ENTSO-E area, overriding `max_price_gap_hours`, e.g. `IE_SEM: 25` as the ENTSO-E Transparency Platform publishes no prices for the 25 hour delivery day at the end of daylight saving time in 2022 and 2023.",
    )
    max_capacity_gap_hours: Annotated[int, Field(ge=0)] = Field(
        48,
        description="Longest gap in hours of the ENTSO-E cross-border capacity source of a border direction that is interpolated, used with `electricity: cross_border_capacities: ntc`. Longer gaps raise an error. The default covers up to two missing delivery days, e.g. of the offered capacities on the Nordic and Baltic borders in 2023.",
    )
    fixed_border_capacities: dict[
        str, Annotated[float, Field(ge=0)] | Literal["physical"]
    ] = Field(
        default_factory=dict,
        description="Constant capacity in MW per direction for bidding zone borders without published ENTSO-E capacities, used with `electricity: cross_border_capacities: ntc`. Keys are alphabetically sorted ENTSO-E area pairs like `DK_1-GB`; the value `physical` keeps the physical transmission capacity of the network and `0` disconnects the areas, e.g. for interconnectors commissioned after the snapshots or connections without a market border created by the clustering.",
    )

    exchange_proxy_borders: list[str] = Field(
        default_factory=list,
        description="Bidding zone borders that use the monthly maximum of the day-ahead scheduled commercial exchanges over both directions as capacity even if ENTSO-E publishes capacities, used with `electricity: cross_border_capacities: ntc`. Given as alphabetically sorted ENTSO-E area pairs like `BE-NL`, e.g. for borders within the flow-based Core region whose published net transfer capacities are outdated. `fixed_border_capacities` take precedence.",
    )

    @field_validator("fixed_border_capacities", "exchange_proxy_borders")
    @classmethod
    def check_border_keys(
        cls, value: dict[str, float | str] | list[str]
    ) -> dict[str, float | str] | list[str]:
        """Require borders as two alphabetically sorted areas `AREA0-AREA1`."""
        invalid = [
            k
            for k in value
            if len(k.split("-")) != 2 or k.split("-") != sorted(k.split("-"))
        ]
        assert not invalid, f"Borders must be given as sorted `AREA0-AREA1`: {invalid}"
        return value
