# Independently verified R7 raw-input network source gate

Both complete raw-input MLP families failed the source-competence criterion. The original arithmetic-mean/population-SD branch had a six-core selected-checkpoint source NMAE mean of 0.9674671026. The one permitted fixed median/1.4826-MAD repair reduced this mean to 0.2138580085, still above the frozen daily-naive threshold of 0.1155796124. Every one of the 60 selected fits in each branch was individually worse than daily naive. The source-only addendum therefore stopped the extension before any new external forecast, score or bootstrap; no third repair was attempted.

All 120 source checkpoints, reports and ten-checkpoint validation traces are retained. The independent scorer imported no experiment producer, fitting factory or producer metric. It reconstructed the 11,952 training and 104 source-validation rows from the frozen panel, masks and row keys; independently reconstructed fine/coarse causal histories and shared minibatch draws; verified the earliest source-validation minimum; and restored all 104 source-validation forecasts for every selected checkpoint. It verified exact tensor schemas and 53,472 forecast-active core parameters, 53,473 mixed/attention parameters and 53,475 gated/stride parameters, without trainable padding. Only the two source-only global affine constants are nontrainable buffers; every learned tensor reaches the forecast.

| Source-selected model | Mean/SD fine | Mean/SD coarse | Median/MAD fine | Median/MAD coarse |
|---|---:|---:|---:|---:|
| direct | 1.044130 | 1.000889 | 0.199762 | 0.226560 |
| max_repeat | 0.981596 | 1.000889 | 0.202307 | 0.226560 |
| avg_repeat | 0.961592 | 1.000889 | 0.211282 | 0.226560 |
| haar_absolute | 0.889042 | 0.854269 | 0.203086 | 0.211562 |
| haar_signed | 0.889634 | 0.854269 | 0.204909 | 0.211562 |
| max_blur_repeat | 0.972515 | 1.159892 | 0.217962 | 0.224185 |
| mixed | 0.861300 | 1.000889 | 0.213523 | 0.226560 |
| gated | 1.004680 | 1.000889 | 0.206705 | 0.226560 |
| attention | 1.097525 | 1.000889 | 0.227839 | 0.226560 |
| stride_conv | 1.063479 | 0.975955 | 0.216243 | 0.206851 |

Each table entry averages losses from three separately fitted seeds; no forecast ensemble is scored. The full seed values and all 40 groups are in `source_competence_by_cell.csv` and each branch's independent source tables. The retained R6 N-HiTS source NMAE range is 0.095284–0.100142. Independently reconstructed source NMAEs are 0.115580 for daily naive, 0.116982 for weekly naive and 0.351905 for persistence, under identical complete-pair masks and raw-history row weights.

Original arithmetic-mean/SD scaling uses center 1201.172098395683 kW and scale 4589.924394603481 kW; training-only robust scaling uses median 128.0487823486328 kW and scaled MAD 144.5913449169159 kW. Independent source NMAE ranges are respectively 0.6813522499–1.2777508165 and 0.1874245511–0.2557894721. The maximum source metric discrepancy between independent float64 arithmetic and producer-reported float32 loss is 1.49×10⁻⁷ and 3.61×10⁻⁸, respectively. A second, independently written NumPy float64 forward differs from independent Torch float32 forecasts on eight validation rows per checkpoint by at most 0.0006751 kW and 0.00004888 kW, respectively. The independent daily-naive metric differs from the frozen float32 gate threshold by 2.12×10⁻¹⁰, far below either failed-gate margin.

Independent compensated initial-coordinate fixtures also reproduce the direct/signed initial function: maximum float32 discrepancies are 0.0008545 kW (mean/SD) and 0.00003052 kW (median/MAD) across the three fixed initializations. These are independently generated fixtures, distinct from the producer preflight's source-row figures.

Interpretation: the diagnostic successfully rejects an uncompetitive source model before transport evaluation. The source-only conditioning sensitivity does not establish what caused the original failure. Compared with R6, the backbone, capacity and normalization changed jointly; the experiment cannot identify a bypass-removal effect. Since external inference stopped, these raw-network references offer no external mechanism finding and cannot empirically isolate information loss in the complete N-HiTS network. The previously scored R6 periods/customers also remain unsuitable for any claim of new independent confirmation.

Reproduction: `scripts/144_r7_raw_independent_score_v1.py`; two branch source audits and source tables in `results/r7_raw_independent_v1`; `source_competence_gate.json`; `source_competence_by_cell.csv`; and `independent_completion_v1.json`, which binds all 120 retained source checkpoints/reports and every independent output. The fully specified external scoring implementation remains available but was not executed because the source gate failed.
