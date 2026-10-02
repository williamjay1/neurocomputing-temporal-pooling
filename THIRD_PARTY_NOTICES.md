# Third-party attribution

**NeuralForecast**, Nixtla contributors, version 3.2.2, Apache License 2.0: https://github.com/Nixtla/neuralforecast/blob/main/LICENSE . The primary and independent-construction scripts import `NHITSBlock` and `_IdentityBasis` from `neuralforecast.models.nhits`. The model blocks are a third-party dependency, not an independently invented architecture. Study adapters, gate/provenance checks and independently computed scoring implement the reported controls. Install the dependency through the pinned environment; its installed source and binary runtime are not redistributed here.

**N-HiTS research:** Challu et al. (2023), *NHITS: Neural Hierarchical Interpolation for Time Series Forecasting*, AAAI 37(6), 6989–6997, https://doi.org/10.1609/aaai.v37i6.25854 . The study analyzes target-dependent access with matched controls; it does not claim to introduce N-HiTS.

**SciencePlots**, John Garrett and contributors, Copyright (c) 2018 John Garrett, MIT: https://github.com/garrettj403/SciencePlots/blob/master/LICENSE . The plotting script locally implements Matplotlib artists and refers to the `nature.mplstyle` parameters. No SciencePlots package or vendored style file is bundled. Reference commit and official Nature guidance are recorded in `results/r7_figures_v1/primary_style_references.json`. Styling does not imply Nature endorsement.

NumPy, SciPy, PyTorch, pandas, Matplotlib, scikit-learn, Pillow and other installed dependencies retain their respective licenses. They are installed separately and not bundled. Arial font availability affects rendering on other systems; do not copy proprietary system font files into this repository. The released PDF artwork uses embedded font subsets; SVG retains editable text and the font-family declaration.

Real data attribution and historical source terms are in `DATA_LICENSE.md` and `docs/PUBLIC_RELEASE_DATA_AND_RIGHTS_AUDIT_20261002.md`.
