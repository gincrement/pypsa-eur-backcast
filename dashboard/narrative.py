# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Static prose of the overview and limitations pages, placeholders in braces are filled from the selected run.
"""

WHAT = """
The backcast replays a past year hour by hour and asks: if every power plant bid its fuel and CO2 cost,
which day-ahead price would each bidding zone have seen? It compares that price with the price the market
actually cleared, as published on ENTSO-E (European grid operators' data platform). The main year is 2023;
2019 is a validation year with a calm gas market.
"""

HOW = """
The model is PyPSA-Eur run as a pure dispatch problem. All capacities are fixed to the fleet of the backcast
year, so the optimisation only decides which plants run in each hour. Gurobi solves one linear program for
the whole year with perfect foresight. The model price is the dual of the energy balance at each bus: the
cost of serving one more MWh there.
"""

PIPELINE = {
    "Inputs": [
        "observed ENTSO-E load per zone",
        "wind and solar from weather data, wind and nuclear calibrated to ENTSO-E output",
        "plant fleet of the year with a cost spread per plant type",
        "monthly nominal fuel and CO2 prices",
        "hourly cross-border trading capacity (NTC)",
        "hydro inflow, pumped storage capped at country values",
    ],
    "Model": [
        "one bus per bidding zone",
        "zones trade over NTC links, no physical grid",
        "cost-minimal dispatch for all hours of the year",
        "one linear program, perfect foresight",
    ],
    "Output": [
        "hourly price per zone (dual of the energy balance)",
        "dispatch per carrier and cross-border flows",
        "scored against ENTSO-E day-ahead prices",
    ],
}

SCORING = """
For each zone the model price is compared with the observed hourly price over the full year. Three numbers
summarise a run, each as the median over the scored model buses:

- **MAE** (mean absolute error): the typical hourly miss in €/MWh.
- **Pearson r**: how well the hourly shape follows the market, 1 is perfect.
- **|bias|**: the absolute difference of the yearly mean prices.

A refinement is kept only if it lowers the median MAE in both 2023 and 2019. This guards against tuning to
one year. The 2019 prices of PL, RO, BG and GB are published in local currency and are not scored.
"""

WORKS = """
- The hourly shape follows the market in most zones: median r is {r} in {run}.
- In tight hours the model dispatches the same plants as the market; the gap is the price, not the dispatch.
- Most kept refinements are data fixes (wind calibration, plant data, zone shapes, pumped storage size),
  not tuning knobs, and they pass in both years.
"""

LIMITATIONS = """
| Limitation | Evidence in {run} | Where to look |
|---|---|---|
| Price spikes are too low. Real bids carry markups (scarcity rent, start-up costs) that a cost-based model cannot produce. | mean error in the 5 % most expensive hours: {top5} €/MWh (median over buses) | Overview, by price level |
| No hours at or below 0 €/MWh. Nuclear, biomass and Nordic reservoir hydro throttle at a positive cost and absorb every surplus; flexibility limits are missing. | hours ≤ 0: {nonpositive_observed} observed vs {nonpositive_model} modelled (mean over buses) | Overview, metric table |
| Germany exports too much. | net export {de_model} TWh modelled vs {de_scheduled} TWh scheduled | Cross border flows, zone DE_LU |
| Some zones carry a large level error: Italy is too expensive, Finland too cheap. | bias Italy {italy_bias}, FI {fi_bias} €/MWh | Overview, map metric bias |
| The Baltics and other edge zones miss the daily shape. Cheap night imports from flat Nordic prices compress the profile. | largest MAE: {worst_mae} | Bidding zone |
| The south-east chain (HU, SK, HR) shares one price more often than the market does. | | Cross border flows |
| Hydro inflow and its split between zones of one country (SE, NO, IT) is uncertain. Reservoirs have no water values from observed filling levels. | | Bidding zone, energy per carrier |
| Plant data has gaps; Dutch gas plants needed local corrections. Gas prices are monthly, not daily. There is no unit commitment or ramping. | | Bidding zone, energy per carrier |
"""

HISTORY = """
Median MAE in €/MWh as recorded when each step was tested (2023 / 2019).

| Step | Result | Verdict |
|---|---|---|
| Baseline with NTC links and nominal prices | 28.5 / 9.9 | start |
| Wind and nuclear calibrated to ENTSO-E, lignite fuel, flat hydro dispatch | 24.0 / - | kept |
| Must-run floors, thermal cost spread, hybrid Core capacities | 22.4 / 8.6 | kept |
| Iberian gas price cap, GB wind targets, wider cost spread, fixed external borders | 21.8 / 8.2 | kept |
| Bidding-zone shapes from zone polygons | 22.3 / 8.1 | kept as a data fix despite the 2023 loss |
| Pumped storage energy capped at Geth et al. 2015 | 21.2 / 8.0 | kept |
| Dutch CCGT data fix and cost spread referenced to observed output | 21.0 / 7.9 | kept |
| Biomass output capped at ENTSO-E capacity factors | 24.1 / 9.1 | rejected |
| Solar calibrated to ENTSO-E | 22.1 / 8.2 | rejected, worse in 2019 |
| Monthly thermal availability from observed peaks | 22.1 / 10.8 | rejected, load shedding in 2019 |
| Cost spread with fixed utilisation | level +7 to +10 | rejected |
| Annual hydro inflow scaling per zone | median gain only | rejected, r and mean MAE worse |
| Reservoir targets, unit commitment, negative renewable bids, reserve demand curves | no gain or too slow | rejected |
"""
