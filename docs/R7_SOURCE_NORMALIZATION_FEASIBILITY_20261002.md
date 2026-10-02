# R7 source-only normalization feasibility decision

Scope: read-only diagnosis of the original 60-cell raw-input MLP family. No additional fitting, checkpoint changes, external prediction reads, external scoring, or changes to the R6/R7 statistical results were performed. This is a feasibility decision before any new MLP external comparison, not evidence for a normalization mechanism or a new forecasting method.

## Executed source checks

All 21 source-cache file hashes matched their released manifest. All 60 original MLP checkpoint and report hashes matched `ALL60_SOURCE_FITS_COMPLETE`. The original global center and scale were independently reconstructed exactly from the 8,030,052 observed fine-history entries of the 11,952 source training rows. Entries in overlapping windows form the actual normalization multiset; they are not 8 million independent observations. Only these training entries, the frozen 104 source validation rows, and source-fit metadata were read. A metadata-only check found no MLP forecast-completion marker at the time of this diagnosis.

| Source-training observed-entry statistic | kW |
|---|---:|
| Minimum | 0 |
| 5th percentile | 3.8071064949 |
| 25th percentile | 50.8720932007 |
| Median | 128.0487823486 |
| 75th percentile | 335.4390258789 |
| 90th percentile | 1,234.375 |
| 95th percentile | 3,006.64453125 |
| 99th percentile | 26,208.333984375 |
| Maximum | 45,416.66796875 |
| Current mean | 1,201.172098395683 |
| Current population standard deviation | 4,589.924394603481 |
| Median absolute deviation about the median | 97.52552604675293 |
| 1.4826 × MAD, floored at 1 kW | 144.5913449169159 |

Values above 10,000 kW account for 3.903337% of entries, 73.647503% of their total load, and 94.198447% of the centered squared deviations used in the current variance. Source column 9 has an observed-entry mean of 21,807.9491 kW; source column 6 has a mean of 6.9167 kW. Source-row observed-history means range from 1.7355 to 27,275.1736 kW, with median 144.8880 kW; 42.9552% of rows have a mean below 100 kW. This documents heterogeneity and the dominance of high loads in global moments. It does not establish that any particular industrial or household category caused optimization failure.

The 104 fixed source validation rows cover eight source customers, 13 daily origins each, and 4,992 complete pairs. Every row has all 48 pairs available. Their common past-observation scales range from 1.7185 to 250.2785 kW, with median 98.1445 kW. Thus the held validation subset does not span the largest training load levels. Its limitations must remain visible when interpreting a source-selected normalizer.

## Independently reconstructed source competence references

For each validation row, the denominator is the original fine-history observed-value mean floored at 1 kW and then stored as float32, exactly as in script 138. Errors use both endpoints of complete pairs. Row errors are divided by that denominator and averaged over the fixed 104 rows. All arithmetic in this reconstruction is double precision after loading the stored values; it is not a new target-period metric.

| No-fit source reference | Common-weight NMAE | Row-macro MAE, kW |
|---|---:|---:|
| Previous day, original fine history | 0.11557961237049463 | 8.9849178525261 |
| Previous day, pair-mean repeated coarse history | 0.11588778686744207 | — |
| Previous week, original fine history | 0.11698235257180209 | 10.186327323484688 |
| Last-value persistence, original fine history | 0.35190480791704953 | 40.45292616993762 |

The daily reference is the final 96 fine-history slots; the weekly reference is the first 96 slots of the seven-day history. For the coarse daily reference the final 96 already causally prepared coarse slots are used. Neither reads future labels to construct forecasts. The scoring formula is `mean_rows(sum_endpoint(where(complete_pair_mask, abs(prediction - label), 0)) / endpoint_count / common_past_scale)`.

Original MLP source-validation results, checked against all 60 frozen source reports:

| Family summary | Common-weight NMAE |
|---|---:|
| Minimum among 60 source-selected checkpoints | 0.6813523173332214 |
| Maximum | 1.277750849723816 |
| Mean of all 60 | 0.9807605077823003 |
| Direct fine, three-seed mean | 1.0441297094027202 |
| Direct coarse, three-seed mean | 1.0008887648582458 |
| Signed Haar fine, three-seed mean | 0.889633854230245 |
| Signed Haar coarse, three-seed mean | 0.8542691270510355 |

All 60 checkpoints perform substantially worse than either same-history daily reference. Against the fine daily reference the ratio ranges from approximately 5.90 to 11.06. Twelve fits select the last validation checkpoint at update 2,000; the others select earlier updates. These reports do not distinguish underfitting, optimization, conditioning, architecture capacity, or other causes. The loss level alone does not establish that a successful solution is unreachable without normalization statistics or residual paths.

## Decision and smallest executable repair

**CONDITIONAL: one bounded source-only robust-affine repair is worth running; the original family cannot currently support a mechanism comparison.** Preserve the original 60 models, protocol, traces, and completion marker as the failed source-competence attempt. Do not reinterpret their adapter contrasts as evidence that removing normalization access or forecast residuals harms forecasting. The proposed repair costs approximately the same as the already completed small MLP family and does not add information to its input.

Before any new external prediction or score is read, freeze one additional complete family using the training-only constants `center = 128.0487823486328 kW` and `scale = max(1.4826 × MAD, 1 kW) = 144.5913449169159 kW`. Apply the same constants before every adapter and to every output inverse transform, all ten adapters, both histories, and all three seeds. Retain the original 2,000 updates, 64-row batches, frozen draws, optimizer and learning rate, gradient cap, model widths, common fine-history loss weights, complete-pair masks, validation schedule and selection rule. Add no per-window or per-customer statistics, residual bypass, clipping, additional mask inputs, or parameter differences. Recheck the existing compensated direct/signed initial-function fixture under the new constants. Use a new script, protocol, result directory and completion identity; do not overwrite script 138 or its outputs.

The parent accepted a stricter source-only repair rule after this diagnosis: the original mean/std family is a source-competence failure and must not proceed to external forecasts. Make exactly one fixed robust-affine attempt for the entire 60-fit family; do not select individual adapter, seed or history normalizers and do not return to the failed original branch for external forecasting. Retain both family-level source scores and all 120 source results. Equal-weight summaries over the fixed conditions are engineering summaries, not independent replication; some coarse representations are exactly identical. This changes the original source-only normalization protocol and must be declared as a post-review source-feasibility deviation before the new fits, rather than represented as an unchanged preplanned experiment.

The actual gate was frozen by the parent in `R7_SOURCE_CONDITIONING_ADDENDUM_20261002.md` before the repair fits: the mean selected source-validation NMAE of **all 36 six-core fits**, using the same common weights across both histories, must be less than **0.11557961237049463**. Report every individual cell, including failures. If the mean fails, stop external forecasting after this one repair; no third normalizer, learning-rate change, backbone enlargement, customer removal, output clipping or external winner selection is allowed. If it passes, forecast all 60 repaired conditions on both previously studied external groups. This is a source-feasibility gate, not a journal requirement. My earlier suggested direct/signed per-history cell gate was not adopted and must not substitute for, supplement or change the frozen family-mean gate after fitting began.

The robust scale is approximately 31.74 times smaller than the current standard deviation. It makes typical low-load variation larger in normalized coordinates and reduces the physical scale of random initial output fluctuations, which gives a plausible optimization rationale. It also makes the largest normalized input much larger: MAD scaling does **not** compress the heavy tail or guarantee numerical improvement. Fixed positive affine normalization leaves the core post-adapter information content unchanged and can be absorbed by affine network weights; it does change finite-budget optimization, initial physical predictions and gradient clipping. A successful repair would demonstrate training feasibility under a second source-only conditioning choice, not prove that conditioning was the sole original cause or that a theoretical access mechanism was identified.

Both available external periods and their outcomes were already examined by earlier R6 work. Even if no new MLP external forecast has been read, the repaired family remains post-review exploratory analysis on previously studied external periods, not a newly independent confirmation. Advance the entire robust family only after source competence and provenance checks, and retain every fixed adapter/history/seed external result under the original evaluation rules. If it fails the source gate, perform no external forecast or mechanism comparison from either raw-MLP branch.

## Bound source identities

- Original MLP script 138 SHA256: `26066ee2fc2ff5501a4cf75a27fc5fdc8e97139f620dd2e24d029fe8c1f11298`.
- Original MLP protocol SHA256: `a642d50021b89edae0259fdf2fac11efdb6ffd6c97907a631d54f5394576206c`.
- Original 60-fit completion SHA256: `b26488292b0c7b154a3fbbadefe0be255832eced7e7fb178adf0bb0c93279c36`.
- Frozen source-cache manifest SHA256: `a240d3c4c3f240bb798006ea10cbc4eed38fcc201d9ea71e936ca58790ba325a`.

No external data, prediction or evaluation identity is claimed by this note. No normalization repair has been implemented or executed by this subtask.

## Immutable original source outcome snapshot

The companion JSON saves the complete original protocol and completion, all 60 source reports including every validation trace, all checkpoint/report hash bindings, all 21 cache hash checks, and independently reconstructed no-fit scores for all 104 source validation rows. No external outcome record is included.

| Adapter | History | Seed | Source-selected update | Source NMAE |
|---|---|---:|---:|---:|
| direct | fine | 20261001 | 1200 | 1.031369686127 |
| direct | coarse | 20261001 | 1400 | 1.047667741776 |
| max_repeat | fine | 20261001 | 2000 | 1.143946170807 |
| max_repeat | coarse | 20261001 | 1400 | 1.047667741776 |
| avg_repeat | fine | 20261001 | 800 | 0.976655900478 |
| avg_repeat | coarse | 20261001 | 1400 | 1.047667741776 |
| haar_absolute | fine | 20261001 | 800 | 0.854339063168 |
| haar_absolute | coarse | 20261001 | 2000 | 0.959824800491 |
| haar_signed | fine | 20261001 | 1400 | 0.990179061890 |
| haar_signed | coarse | 20261001 | 2000 | 0.959824800491 |
| max_blur_repeat | fine | 20261001 | 1800 | 1.068006873131 |
| max_blur_repeat | coarse | 20261001 | 1400 | 1.277750849724 |
| mixed | fine | 20261001 | 1000 | 0.890028417110 |
| mixed | coarse | 20261001 | 1400 | 1.047667741776 |
| gated | fine | 20261001 | 1400 | 1.075983881950 |
| gated | coarse | 20261001 | 1400 | 1.047667741776 |
| attention | fine | 20261001 | 1800 | 1.058198213577 |
| attention | coarse | 20261001 | 1400 | 1.047667741776 |
| stride_conv | fine | 20261001 | 2000 | 1.227580547333 |
| stride_conv | coarse | 20261001 | 1400 | 1.034938097000 |
| direct | fine | 20261002 | 1000 | 0.877799689770 |
| direct | coarse | 20261002 | 1800 | 1.022714138031 |
| max_repeat | fine | 20261002 | 1200 | 0.845619797707 |
| max_repeat | coarse | 20261002 | 1800 | 1.022714138031 |
| avg_repeat | fine | 20261002 | 1400 | 0.881288528442 |
| avg_repeat | coarse | 20261002 | 1800 | 1.022714138031 |
| haar_absolute | fine | 20261002 | 1000 | 0.794892847538 |
| haar_absolute | coarse | 20261002 | 2000 | 0.681352317333 |
| haar_signed | fine | 20261002 | 2000 | 0.896699309349 |
| haar_signed | coarse | 20261002 | 2000 | 0.681352317333 |
| max_blur_repeat | fine | 20261002 | 1000 | 0.944840013981 |
| max_blur_repeat | coarse | 20261002 | 1600 | 1.156562685966 |
| mixed | fine | 20261002 | 1800 | 0.766036093235 |
| mixed | coarse | 20261002 | 1800 | 1.022714138031 |
| gated | fine | 20261002 | 1800 | 0.996221661568 |
| gated | coarse | 20261002 | 1800 | 1.022714138031 |
| attention | fine | 20261002 | 1600 | 1.045792698860 |
| attention | coarse | 20261002 | 1800 | 1.022714138031 |
| stride_conv | fine | 20261002 | 2000 | 0.920976638794 |
| stride_conv | coarse | 20261002 | 2000 | 0.904366254807 |
| direct | fine | 20261003 | 1200 | 1.223219752312 |
| direct | coarse | 20261003 | 1600 | 0.932284414768 |
| max_repeat | fine | 20261003 | 1400 | 0.955222427845 |
| max_repeat | coarse | 20261003 | 1600 | 0.932284414768 |
| avg_repeat | fine | 20261003 | 1000 | 1.026832938194 |
| avg_repeat | coarse | 20261003 | 1600 | 0.932284414768 |
| haar_absolute | fine | 20261003 | 400 | 1.017892837524 |
| haar_absolute | coarse | 20261003 | 2000 | 0.921630263329 |
| haar_signed | fine | 20261003 | 1800 | 0.782023191452 |
| haar_signed | coarse | 20261003 | 2000 | 0.921630263329 |
| max_blur_repeat | fine | 20261003 | 1400 | 0.904696702957 |
| max_blur_repeat | coarse | 20261003 | 1000 | 1.045363187790 |
| mixed | fine | 20261003 | 1800 | 0.927835524082 |
| mixed | coarse | 20261003 | 1600 | 0.932284414768 |
| gated | fine | 20261003 | 1400 | 0.941834151745 |
| gated | coarse | 20261003 | 1600 | 0.932284414768 |
| attention | fine | 20261003 | 1400 | 1.188584923744 |
| attention | coarse | 20261003 | 1600 | 0.932284414768 |
| stride_conv | fine | 20261003 | 1200 | 1.041878581047 |
| stride_conv | coarse | 20261003 | 2000 | 0.988560736179 |

Companion source-audit JSON SHA256: `afe91211d20aa4c90b46b6908bfdd7a6f399f0c444a4fef8c7fbf09febff796b`.

The source-audit JSON is preserved as the original diagnostic snapshot. Its field `engineering_decision.source_competence_gate_proposal` records the earlier unadopted suggestion, not the actual gate. The later `R7_SOURCE_NORMALIZATION_GATE_BINDING_20261002.json` explicitly binds the parent's actual frozen addendum and supersedes that proposal without changing any original source outcome, baseline or hash.
