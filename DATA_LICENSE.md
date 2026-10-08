# Data and material licenses

The MIT license applies to the author-created research software, not to the original electricity
measurements or to third-party dependencies.

## Author-generated synthetic data and results

The synthetic distributions, exact risk tables, oracle checks, trained checkpoints and result
summaries in `results/` are author-generated. Result tables and figures are released under
**CC BY 4.0**, with attribution to Junjie Zhang and this repository. Trained checkpoint files are
supplied for computational reuse under the MIT code license. The synthetic examples are explicitly
generated mathematical constructions and are not human-participant observations.

## Real electricity data

The complete-network evaluation reported in the paper uses the public ElectricityLoadDiagrams20112014
collection: Artur Trindade (2015), UCI Machine Learning Repository, DOI
https://doi.org/10.24432/C58C86 , license CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/ ). The derived working panel built from that
collection is **not redistributed in this version**; it forms part of version 1.0.0 of this archive.
Any redistribution of the original measurements requires the attribution and license above, and the
provider archive itself is not redistributed here.

## Third-party dependencies

NumPy, SciPy, PyTorch, pandas, Matplotlib, Pillow, SciencePlots and cmcrameri retain their
respective licenses. They are installed separately from the pinned requirements files and none of
them is bundled in this repository.
