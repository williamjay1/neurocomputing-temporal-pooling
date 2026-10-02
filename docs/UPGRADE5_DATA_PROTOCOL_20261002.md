# UPGRADE5 UCI data protocol

Date: 2 October 2026. Route: SCI; Neurocomputing technical forecasting research. This stage supplies reliable, fixed data for testing irreversible online history retention at unchanged fine forecast endpoints. Its standard is information/chronology integrity and reproducible customer isolation. It performs no model fitting, tuning, final evaluation or manuscript revision; economic causal-identification standards are not applicable.

## Source, rights and storage

Use the official [UCI ElectricityLoadDiagrams20112014 record](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014), Artur Trindade (2015), DOI [10.24432/C58C86](https://doi.org/10.24432/C58C86). The checked official license is CC BY 4.0. Attribute the dataset and provider when reusing or sharing permitted derivatives.

The official dictionary describes 370 clients, native fifteen-minute kW, semicolon-delimited columns and Portuguese clock labels. Ordinary quarter-hour energy would be kW/4 in kWh. Customers introduced later have earlier zero placeholders. Spring clock-change data contain a synthetic zero hour; autumn records combine the repeated hour. These are limitations of physical interpretation even though the provider describes the table as having no missing values.

Official direct ZIP: `https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip`. The page labels its size 249.2 MB and the text 678.1 MB. The checked HEAD response is 200 without Content-Length. The completed streaming GET saved 261,335,609 bytes; ZIP CRC checks passed and the text member is 710,998,915 bytes. Raw SHA-256 is `f6c4d0e0df12ecdb9ea008dd6eef3518adb52c559d04a9bac2e1b81dcfc8d4e1`.

Raw file: `F:/AcademicData/Neurocomputing_OnlineRetention/raw/uci_v1/electricityloaddiagrams20112014_download_20261002_v1.zip`. It is exclusive-created and marked read-only. Existing raw files will never be overwritten, resumed into, moved or deleted; failed attempts, if any, remain preserved. Download logs/metadata are under `D:/MLWork/Neurocomputing_TailTransfer/results/upgrade5_data_v1`. All extraction and work arrays are under `D:/MLWork/Neurocomputing_TailTransfer/datasets/upgrade5_uci_v1`. Before download, D had 154,974,310,400 free bytes and F had 3,700,295,917,568 free bytes. No C-drive outputs, paid API, new package or large training job are required.

## Provider chronology and conservative arrival

Retain the provider-label interpretation instead of pretending that the nominal 96-slot days form a regular physical UTC raster. The first raw label is 2011-01-01 00:15 and the expected final label is 2015-01-01 00:00; confirm the entire timestamp sequence without converting holdout customer values. The fixed panel spans provider labels 2011-01-01 00:00 through 2014-12-31 23:45. The initial unrecorded label is explicitly missing and the last raw 2015 label is outside this label-defined study. The expected panel shape is 140,256 by 80.

The dictionary does not explicitly settle interval-start versus interval-end semantics. Therefore a record labelled t is conservatively released only at t+15 minutes. At a daily origin O=00:00, only labels strictly before O are available; row O belongs to the forecast, never the input. The seven-provider-day input has 672 native slots and the one-provider-day horizon has 96 slots. On DST weeks, seven provider days need not equal 168 elapsed UTC hours. This is an operational provider-label task with conservative availability, not a claim to recover information absent from the archive.

Use Europe/Lisbon as an explicit mainland-time interpretation for optional UTC annotation. `utc_labels.npy` holds UTC label instants where the label is unambiguous and unaffected by the provider's collapsed interval; ambiguous, nonexistent or conservatively excluded DST labels hold NaT. The official wording is Portuguese hour rather than an explicit timezone identifier, so this optional mapping is not an author-certified UTC timestamp field. Online order uses provider slot indices and the declared release rule.

For each year, identify the last Sunday in March and October from the timezone/calendar rule. The provider does not specify inclusion of the 01:00/02:00 boundary labels or their interval-end meaning, so conservatively mark five labels 01:00, 01:15, 01:30, 01:45 and 02:00 invalid on each transition date. Spring placeholders and autumn merged values are never used as individual fine endpoints or ordinary encoder observations. This excludes one boundary label beyond the ambiguous hour rather than manufacturing separate measurements. Calendar flags are fixed independently of holdout consumption values; exact flags and the underlying ambiguity are provided for later sensitivity protocols if needed.

## Train-only customer eligibility and fixed split

Before any post-2012 numeric conversion, read only 2011–2012 customer observations. Infer an activation proxy at each client's first finite, positive, nonnegative, non-DST record. Pre-proxy zeros are unavailable, while every finite nonnegative zero after this forward activation remains an observed zero unless the calendar rule invalidates it. The provider supplies no exact activation date; the first-positive proxy conservatively treats earlier zeros as unknown. It is not a claim that every individual early zero has a known missingness cause. Later all-zero operating periods are retained, not mechanically deleted.

A candidate must have at least 95% valid slots during the 35,136-label leap year 2012 and a finite observed population standard deviation greater than 1e-8. Customer eligibility and ordering never use 2013 or 2014 outcomes. Save the complete 370-client training qualification audit, including all excluded candidates. If fewer than 80 qualify, fail visibly rather than using holdout values to fill the quota.

Permute the lexically ordered eligible client IDs once with NumPy Generator(PCG64), seed 20261002. Allocate the first 32 to development, next 16 to calibration and next 32 to confirmation; the three ID sets are disjoint. These fixed 80 clients form the panel columns in this group order. No later coverage threshold, outcomes or model error can alter the roster. The confirmation group remains excluded from model/utility fitting, calibration and selection.

Periods are fixed as follows, with exclusive upper endpoints:

| Purpose | Clients | Forecast-label period |
|---|---|---|
| Forecaster fitting | Development 32 | 2012-01-01 to 2012-10-01; first origin 2012-01-08 after seven-day history |
| Deletion-utility training | Development 32 | 2012-10-01 to 2013-01-01 |
| Development evaluation | Development 32 | 2013-01-01 to 2013-07-01 |
| Independent calibration | Calibration 16 | 2013-07-01 to 2014-01-01 |
| Reserved confirmation | Confirmation 32 | 2014-01-01 to 2015-01-01 |

Earlier observations of the corresponding customer may supply arrived lag history at a later evaluation origin. Their use as inputs does not authorize fitting model parameters on reserved confirmation labels. The preparation computes only explicit validity counts for quality metadata after locking, not consumption summaries or prediction errors for holdout periods.

## Scaling, masks and streaming contract

Save physical kW as float32 and validity as a separate uint8 array. Invalid entries become NaN in the model-facing panel; the original archive remains unchanged. There is no interpolation, target filling, clipping of valid zeros or reconstruction of the lost autumn quarters in preparation.

The primary pilot scaler is a single pooled mean and population standard deviation fitted exclusively to valid development-32 observations in January–September 2012. It is fixed before utility-training, development, calibration or confirmation labels. Held-customer values do not enter this fit. Per-client 2012 support statistics are supplied as separately labelled past-only metadata, not automatically used by the primary transform; choosing another support/scaling protocol belongs to a later declared experiment. No normalized full-history array is created.

`panel_kw.npy` and `valid.npy` have shape [140256,80]. `provider_labels.npy` and `utc_labels.npy` have shape [140256]; `client_ids.npy` and `roster_v1.json` bind IDs to columns. Development columns are 0–31, calibration 32–47, confirmation 48–79. `logical_release_index.npy` specifies that a row releases one ordinal step after its label. `partitions_v1.json` and per-purpose origin arrays bind dates, indices and client columns. `dataset_manifest_v1.json` exposes `array_paths`, `customer_groups`, `customer_group_columns`, `client_id_column_order`, and `date_to_index` for the caller.

The offline scorer can read future truth in the prepared panel. A separate online state simulator must instead read newly released rows once in index order; its prediction interface receives only retained state. It must not reopen this panel, the text or the original ZIP to recover deleted historical values, and it must never expose unreleased outcome rows to the predictor. The presence of offline truth is not permission to bypass an irreversible state. This preparation defines the access contract; the subsequent simulator must independently test and enforce it.

## Small development panel and completion gate

Use the first eight development IDs for a small fixed pilot. `pilot8_panel_kw.npy`/`pilot8_valid.npy` cover 2013-01-01 through 2013-01-21 (2,016 by 8). The first seven days are arrival-only warmup; fourteen origins 2013-01-08 through 2013-01-21 each forecast 96 slots from 672 arrived slots. No calibration or confirmation customer/value appears in this pilot. This stage performs no model training.

Before converting post-2012 client values, write `study_lock_v1.json` binding the script, this protocol, raw/text hashes, training eligibility audit, roster, scaler, dates and seed. Then build the fixed panel without dropping later customers. Synthetic fixtures check conservative spring/autumn invalidation, preservation of post-activation zero and exclusion of origin/future labels. Validate native spacing, duplicates, chronology, panel-mask consistency, ID disjointness and origin bounds. Record data-quality masks, input/array hashes, array shapes, actual file sizes, CPU and wall time. `dataset_manifest_v1.json` is the final COMPLETE_PASS marker, written last. Any failure must remain recorded; a file merely written is not evidence that these checks passed.
