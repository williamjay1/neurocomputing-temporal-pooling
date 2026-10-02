# Reproduce the frozen numerical evidence

This repository retains the full frozen 80-customer UCI panel, source caches,
selected checkpoints, all R6 forecasts and the independent result ledgers.
The historical scripts are preserved byte for byte. They refer to the author's
Windows workspace; `legacy_runtime.py` relocates their paths in memory without
changing their scientific functions, assertions, input arrays or SHA256 checks.
Only generated output directories are redirected. Saved evidence is read-only.

The source environment used Python 3.12.10 and CPU PyTorch. Exact observed
distribution versions are recorded in `environment_observed.json`. Install:

```sh
python -m venv .venv
# Activate .venv using your platform's usual command.
python -m pip install -r requirements-neural.txt
python reproducibility/run.py smoke
```

The release's actually executed checks are recorded in
[`validation_summary.json`](validation_summary.json) and
[`verified_runs/`](verified_runs/). All five commands (`smoke`, `all`,
`statistics`, `residual`, `figures`) completed successfully in the separately
located repository. Published reports replace only local workspace prefixes;
their numerical results and execution status remain unchanged. A clean
dependency installation was not performed in this check.

The smoke command performs no optimization. It executes the original engineering
and Gaussian-oracle checks, verifies all full source-cache bindings, restores all
36 selected N-HiTS checkpoints on the 104 source-validation rows, reconstructs
the 202 rational nonlinear-risk rows with 24 independent LP checks, recomputes
the original 5,000 paired bootstrap draws per phase and all 40 crossed REML fits,
then independently restores all 120 selected whole-input MLP checkpoints.
Both MLP source-competence gates fail; the runner preserves that outcome and
never generates external predictions for these families.

For the complete R6 outcome reconstruction, including all 30 original synthetic
forecasts and both complete customer/time cohorts, run:

```sh
python reproducibility/run.py all
python reproducibility/run.py statistics
python reproducibility/run.py residual
```

`all` recalculates the real cohort metrics and original percentile intervals
from every retained forecast and observed target. `statistics` also runs the
original dense-covariance engineering fixtures and reconstructs the seed-specific
customer/week responses from all original forecasts. `residual` reproduces all
288 local final-block perturbation responses. Individual `source`, `calibration`,
`confirmation`, `synthetic`, `engineering`, `oracle`, `r7-source`, `theory` modes
are available. Calibration/confirmation commands first execute the source audit.

Every command writes a new `reproducibility/runs/<UTC timestamp>_<unique suffix>`
directory and a `reproduction_report.json`; existing results are never replaced.
On the author's Windows workstation, outputs remain on drive D. On other systems,
the default output directory is below the cloned repository. An optional
`--output` argument must point to a new directory. Numerical tolerances separate
float32 model inference from float64 scoring; no numerical outcome is selected
to improve the manuscript.

Figure source code, editable SVG/PDF and native 1000 dpi PNGs are included.
`python reproducibility/run.py figures` regenerates all artwork and automatic
collision audits to a new folder. Arial is required to reproduce the original
font metrics exactly; a fallback font can change spacing. Automatic geometric
checks do not replace inspection of the rendered figure. The source release
contains the original visual and geometric audit records, including AI-assisted
image inspection. Human author review is not asserted by those records.

The exact-theory command calls the original pure rational functions rather than
the original script140 `main` entry point, whose drive-D assertion documents the
author's local storage policy. This bypasses only that entry point and its file
writing; it reproduces the numerical statements and checks the original saved
curves. It is not a new training experiment or an independent empirical cohort.

Installing the stated dependencies on Linux/macOS has not been tested in this
release. The adapter was actually executed in a separately located repository
on Windows. `tests/test_legacy_runtime.py` exercises path relocation and
unchanged-source loading without requiring any training or study data.
