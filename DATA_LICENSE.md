# Data and material licenses

The MIT license applies to the author-created research software, not to the original electricity measurements or third-party dependencies.

## Real electricity data

The derived working panel and source caches in `datasets/upgrade5_uci_v1` and `results/r6_access_v1/source_cache`, and electricity-derived forecasting outputs, retain **CC BY 4.0** attribution to:

Artur Trindade (2015). *ElectricityLoadDiagrams20112014*. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C58C86 . Official source: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014 . License: https://creativecommons.org/licenses/by/4.0/ .

Modifications by Junjie Zhang's study: fixed selection of 80 customer columns from the training-period eligible roster; distinct development, calibration and confirmation groups; activation-prefix invalidation; preservation of subsequent genuine zero loads; daylight-saving transition validity masks; conversion of labels and generation of causal forecasting windows; fine/coarse transformations; source-selected predictions and statistical summaries. The units of the source observations and forecasts are kW. The records are not represented as a new original data collection by the software author. Attribution, this source citation, the license link and a description of any further changes must accompany redistribution.

Original provider ZIP and expanded text table are omitted. The small pilot8 artifacts are previous derived UCI preparation records, not a successful new retention study. Raw archives of historical London, Australian and ETT studies are not in this public release; their distinct original terms are identified in the rights audit.

## Generated data and artwork

The author-generated synthetic data, finite-support analytical examples, numerical result summaries and publication figures in this release are supplied under **CC BY 4.0**, with attribution to Junjie Zhang and this repository. Synthetic examples are explicitly generated; they are not human-participant observations. Trained checkpoint files are supplied for computational reuse under the author's MIT code license; this does not remove any attribution owed to the original UCI data or software dependency.

## Third-party software

Dependency licenses remain their own. In particular NeuralForecast 3.2.2 is Apache-2.0, and the SciencePlots style reference is MIT. See `THIRD_PARTY_NOTICES.md`. Neither runtime binaries nor vendored dependency source are included. Repository-level license metadata describes the software; it does not relicense the above datasets.
