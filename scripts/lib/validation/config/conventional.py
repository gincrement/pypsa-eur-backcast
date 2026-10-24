# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Conventional generators configuration.

See docs in https://pypsa-eur.readthedocs.io/en/latest/configuration/#conventional_cf
"""

from pydantic import BaseModel, ConfigDict, Field

from scripts.lib.validation.config._base import ConfigModel


class _EfficiencyParameters(BaseModel):
    """Parameters of the linear efficiency heuristic for one carrier."""

    efficiency: float = Field(description="Efficiency of a plant built in `year`.")
    slope: float = Field(description="Efficiency gain per later build year.")
    year: int = Field(description="Reference build year of `efficiency`.")
    max: float = Field(description="Upper bound of the estimated efficiency.")


class _EstimateEfficienciesConfig(BaseModel):
    """Configuration for `conventional.estimate_efficiencies` settings."""

    enable: bool = Field(
        False,
        description="Estimate missing plant-level efficiencies from a carrier- and age-dependent linear heuristic. The build year is the year of the last retrofit or, if unknown, of commissioning. Plants without build year or carriers without parameters fall back to the technology cost data.",
    )
    reference_year: int = Field(
        2025,
        description="Year in which the age of power plants is evaluated for degradation.",
    )
    degradation_start: int = Field(
        10,
        description="Age in years after which efficiency degradation starts.",
    )
    degradation_rate: float = Field(
        0.001,
        description="Relative efficiency loss per year of age after `degradation_start`.",
    )
    parameters: dict[str, _EfficiencyParameters] = Field(
        default_factory=lambda: {
            carrier: _EfficiencyParameters(efficiency=e, slope=s, year=y, max=m)
            for carrier, (e, s, y, m) in {
                "lignite": (0.25, 0.003, 1960, 0.42),
                "coal": (0.28, 0.003, 1960, 0.44),
                "CCGT": (0.40, 0.004, 1980, 0.60),
                "OCGT": (0.28, 0.003, 1970, 0.41),
                "oil": (0.28, 0.002, 1960, 0.38),
                "nuclear": (0.33, 0.0, 1960, 0.33),
            }.items()
        },
        description="Per carrier: efficiency `efficiency` of a plant built in `year`, rising by `slope` per later build year up to `max`.",
    )


class _MarginalCostSpreadConfig(BaseModel):
    """Configuration for `conventional.marginal_cost_spread` settings."""

    enable: bool = Field(
        False,
        description="Let the marginal cost of each generator of `carriers` rise linearly with its output, from `mc * (1 - 2 * utilisation * spread)` at zero output to `mc * (1 + 2 * (1 - utilisation) * spread)` at `p_nom`, where `mc` is the mean marginal cost of the generator over all snapshots. This approximates the spread of heat rates and bids of the units behind an aggregated generator. It sets `marginal_cost_quadratic` and requires a solver supporting quadratic objectives. Applied after all fuel and emission costs are added.",
    )
    carriers: list[str] = Field(
        default_factory=lambda: ["CCGT", "OCGT", "coal", "lignite", "oil"],
        description="Generator carriers whose marginal cost is spread.",
    )
    spread: float = Field(
        0.3,
        ge=0,
        description="Half the rise of the marginal cost from zero output to `p_nom` relative to the mean marginal cost.",
    )
    utilisation: float = Field(
        0.35,
        ge=0,
        le=1,
        description="Output relative to `p_nom` at which the marginal cost equals its mean.",
    )


class ConventionalConfig(ConfigModel):
    """Configuration for `conventional` settings."""

    model_config = ConfigDict(extra="allow")

    estimate_efficiencies: _EstimateEfficienciesConfig = Field(
        default_factory=_EstimateEfficienciesConfig,
        description="Estimation of missing plant-level efficiencies.",
    )
    marginal_cost_spread: _MarginalCostSpreadConfig = Field(
        default_factory=_MarginalCostSpreadConfig,
        description="Linear increase of the marginal cost of thermal generators with their output.",
    )
    unit_commitment: bool = Field(
        False,
        description="Allow the overwrite of ramp_limit_up, ramp_limit_start_up, ramp_limit_shut_down, p_min_pu, min_up_time, min_down_time, and start_up_cost of conventional generators. Refer to the CSV file 'unit_commitment.csv'.",
    )
    dynamic_fuel_price: bool = Field(
        False,
        description="Consider the monthly fluctuating fuel prices for each conventional generator. Refer to the CSV file 'data/validation/monthly_fuel_price.csv'.",
    )
    fuel_price_rolling_window: int = Field(
        6,
        description="Monthly rolling mean window for fossil fuel prices smoothing.",
        ge=1,
    )
    fuel_price_nominal: bool = Field(
        False,
        description="Convert fossil fuel prices to nominal EUR using yearly exchange rates instead of deflating them to EUR2020.",
    )
    nuclear: dict[str, str | float] = Field(
        default_factory=lambda: {"p_max_pu": "data/nuclear_p_max_pu.csv"},
        description="For any carrier/technology overwrite attributes as listed below.",
    )
