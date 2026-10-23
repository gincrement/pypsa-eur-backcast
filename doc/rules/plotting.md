<!-- SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur> -->
<!---->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Plotting and Summaries

The rules on this page summarise and visualise solved networks and write under
`results/{run}/`.

## Summaries

{{ rules("make_summary", "plot_summary", "plot_base_statistics") }}

## Price backcast evaluation

Compares the modelled marginal prices of a dispatch-only backcast run with the
observed day-ahead prices of [build_day_ahead_prices][] and writes metrics and
plots under `results/{run}/backcast/`. Only buses with observed prices and hours
with an observed price are evaluated. Metrics are reported per bus, so buses in
the same bidding zone (e.g. `DE00` and `LUG1`) share the observed zone price.

{{ rules("evaluate_price_backcast") }}

## Maps

{{ rules("plot_base_network", "plot_clustered_network", "plot_power_network", "plot_hydrogen_network", "plot_gas_network", "plot_balance_map", "plot_balance_map_interactive", "plot_heat_source_map") }}

## Time series

{{ rules("plot_balance_timeseries", "plot_heatmap_timeseries", "plot_interactive_bus_balance") }}

## Heating diagnostics

{{ rules("build_ambient_air_temperature_yearly_average", "plot_cop_profiles") }}
