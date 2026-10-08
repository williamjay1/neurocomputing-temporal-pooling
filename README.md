# Temporal pooling observation boundaries: research code and evidence

Reproducibility package for *When temporal pooling changes predictive risk: observation boundaries
and how rankings depend on the loss*. This repository publishes **code, frozen protocols and
numerical evidence** for the synthetic mechanism study. It does not publish the manuscript, the
supplementary document, the cover letter or any AI-use statement.

**Version 2.0.0.** Version 1.0.0 of this archive documents the earlier complete-network real-data
study. All versions are reachable through the concept DOI
[10.5281/zenodo.23100654](https://doi.org/10.5281/zenodo.23100654).

## What this version contains

- `scripts/loss_region_oracle.py` - exact finite-state risk algebra, the complete reversal region,
  the uniform box margins, the target-noise bound, the support-perturbation counterexample and the
  independent rational and numerical checks.
- `scripts/mechanism_experiments.py` - the frozen 325-fit neural protocol (Gaussian crossover
  neighbourhood; fixed target with separately optimized squared and absolute loss), checkpointing,
  scoring and the planned paired contrasts.
- `scripts/make_revision_figures.py` and `scripts/figure_house_style.py` - the three main figures
  rendered from the deposited result tables.
- `scripts/verify_delivery.py` - read-only integrity verification of the deposited results.
- `docs/experiment_protocol.json` - the authoritative frozen configuration;
  `docs/delivery_numeric_audit.json` and `docs/figure_QA.json` - executed verification records.
- `results/theory_region/` - parameter grid, oracle anchors, target-noise checks, the
  support-perturbation check and the rendered verification record.
- `results/neural/` - per-seed scores and condition summaries for all 325 fits, the 13 planned
  paired contrasts, the frozen training and test draws, and the 975 checkpoint files of the fitted
  networks.

## What this version does not contain

The journal manuscript, the supplementary document, the cover letter and the AI-use statement are
not part of this archive, and neither are provider electricity files. The complete-network study
and its derived UCI panel remain in version 1.0.0 of the same archive.

## Reproduce

Python 3.12 was used; the observed environment is pinned in `requirements.txt` and
`requirements-neural.txt`.

```bash
python -m pip install -r requirements-neural.txt
python scripts/loss_region_oracle.py                    # theory and oracle checks
python scripts/mechanism_experiments.py prepare
python scripts/mechanism_experiments.py train
python scripts/mechanism_experiments.py score
python scripts/verify_delivery.py                       # read-only check of the deposited results
```

The figure scripts additionally use SciencePlots and the cmcrameri colour maps; the supplied style
module falls back to plain Matplotlib if those packages are absent.

## Rights and attribution

MIT applies to the author-created code. Author-generated synthetic data, result tables and figures
are released under CC BY 4.0. The real-data study cited in the paper uses the public UCI
ElectricityLoadDiagrams20112014 collection (CC BY 4.0, DOI 10.24432/C58C86). See
[DATA_LICENSE.md](DATA_LICENSE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Archive

Concept DOI for all versions: [10.5281/zenodo.23100654](https://doi.org/10.5281/zenodo.23100654).
Each release mints its own version DOI under that concept.
