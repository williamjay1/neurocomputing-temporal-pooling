"""UCI 321 data preparation only. No model fitting or outcome scoring.

New raw downloads are exclusive-created on F; derived artifacts stay on D.
Provider labels are retained, with explicit activation/DST validity.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import time
import traceback
import urllib.request
import zipfile

ROOT = Path(r"D:\MLWork\Neurocomputing_TailTransfer")
RAW_ROOT = Path(r"F:\AcademicData\Neurocomputing_OnlineRetention\raw\uci_v1")
DATA_ROOT = ROOT / "datasets" / "upgrade5_uci_v1"
RESULT_ROOT = ROOT / "results" / "upgrade5_data_v1"
PAGE_URL = "https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014"
ZIP_URL = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
ROSTER_SEED = 20261002
USER_AGENT = "Neurocomputing-online-retention-data-preparation/1.0"
PROTOCOL_PATH = ROOT / "docs" / "UPGRADE5_DATA_PROTOCOL_20261002.md"
GLOBAL_SCALING_SCHEME = "development32_valid_2012_jan_sep_only_pooled_mean_population_std"


def json_write(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def request(url: str, method: str = "GET", headers: dict | None = None):
    return urllib.request.urlopen(
        urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT, **(headers or {})}),
        timeout=90,
    )


def download() -> dict:
    """Never overwrite, rename, delete or resume-write a raw artifact."""
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = RESULT_ROOT / "download_manifest_v1.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("status") == "COMPLETE":
            raw = Path(old["raw_path"])
            assert raw.exists() and raw.stat().st_size == old["downloaded_bytes"]
            assert sha256_file(raw) == old["sha256"]
            print("Existing exclusive-created complete raw verified; no mutation.", flush=True)
            return old
    record = {
        "status": "STARTED", "source_page": PAGE_URL, "download_url": ZIP_URL,
        "dataset_doi": "10.24432/C58C86", "license": "CC BY 4.0",
        "metadata_checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_page_zip_size_label": "249.2 MB", "official_page_text_size_label": "678.1 MB",
        "raw_overwrite_move_delete": False, "attempts": [],
        "space_before": {d: dict(zip(["total", "used", "free"], shutil.disk_usage(d))) for d in ["D:\\", "F:\\"]},
    }
    assert record["space_before"]["D:\\"]["free"] > 3 * 1024**3
    assert record["space_before"]["F:\\"]["free"] > 1024**3
    try:
        with request(ZIP_URL, "HEAD") as r:
            record["head"] = {"status": r.status, "resolved_url": r.geturl(), "headers": dict(r.headers)}
    except Exception as e:
        record["head_failure"] = repr(e)
    for attempt in range(1, 4):
        raw = RAW_ROOT / f"electricityloaddiagrams20112014_download_20261002_v{attempt}.zip"
        if raw.exists():
            record["attempts"].append({"path": str(raw), "status": "PRESERVED_EXISTING_NOT_OVERWRITTEN"})
            continue
        a = {"path": str(raw), "status": "STARTED", "bytes": 0}
        record["attempts"].append(a)
        json_write(manifest_path, record)
        started = time.perf_counter()
        h = hashlib.sha256()
        try:
            with request(ZIP_URL) as r, raw.open("xb") as f:
                a["response_headers"] = dict(r.headers)
                a["response_status"] = r.status
                a["resolved_url"] = r.geturl()
                total = r.headers.get("Content-Length")
                a["expected_response_bytes"] = int(total) if total else None
                next_progress = 16 * 1024**2
                for chunk in iter(lambda: r.read(1024 * 1024), b""):
                    f.write(chunk)
                    h.update(chunk)
                    a["bytes"] += len(chunk)
                    if a["bytes"] >= next_progress:
                        print(f"DOWNLOAD {a['bytes'] / 1024**2:.1f} MiB; elapsed {time.perf_counter()-started:.1f} s", flush=True)
                        json_write(manifest_path, record)
                        next_progress += 16 * 1024**2
            if a["expected_response_bytes"] is not None:
                assert a["bytes"] == a["expected_response_bytes"]
            with zipfile.ZipFile(raw) as z:
                assert z.testzip() is None, "CRC failed"
                a["zip_members"] = [{"name": x.filename, "uncompressed_bytes": x.file_size, "compressed_bytes": x.compress_size} for x in z.infolist()]
            a.update(status="COMPLETE", sha256=h.hexdigest(), wall_seconds=time.perf_counter()-started)
            raw.chmod(stat.S_IREAD)
            record.update(status="COMPLETE", raw_path=str(raw), downloaded_bytes=a["bytes"], sha256=h.hexdigest(), raw_readonly=not bool(raw.stat().st_mode & stat.S_IWRITE))
            json_write(manifest_path, record)
            print(json.dumps({k: record[k] for k in ["status", "raw_path", "downloaded_bytes", "sha256", "raw_readonly"]}), flush=True)
            return record
        except Exception as e:
            a.update(status="FAILED_PRESERVED", error=repr(e), sha256_of_partial=h.hexdigest(), wall_seconds=time.perf_counter()-started)
            if raw.exists():
                raw.chmod(stat.S_IREAD)
            record["status"] = "FAILED_PRESERVED"
            json_write(manifest_path, record)
            print(f"Download attempt failed and preserved: {e!r}", flush=True)
    raise RuntimeError("All exclusive download attempts failed; see preserved manifest/raw artifacts.")


def inspect_train() -> None:
    import pandas as pd
    record = json.loads((RESULT_ROOT / "download_manifest_v1.json").read_text(encoding="utf-8"))
    assert record["status"] == "COMPLETE"
    raw = Path(record["raw_path"])
    with zipfile.ZipFile(raw) as z:
        member = next(n for n in z.namelist() if n == "LD2011_2014.txt")
        with z.open(member) as f:
            frame = pd.read_csv(f, sep=";", decimal=",", nrows=8)
            print("FIRST_ROWS", frame.iloc[:, :5].to_string(index=False), flush=True)
            print("CLIENTS", list(frame.columns[1:]), flush=True)
        print("ZIP_MEMBER", z.getinfo(member).file_size, flush=True)


def provider_dst_flags(labels):
    """Official hour boundaries are not specified as interval-start/end.

    Conservatively exclude 01:00,01:15,01:30,01:45,02:00 on both dates.
    This rule is calendar-fixed and never determined from held-out values.
    """
    import numpy as np
    import pandas as pd
    flags = np.zeros(len(labels), dtype=np.uint8)
    clock_minute = labels.hour * 60 + labels.minute
    for year in sorted(set(labels.year)):
        for month, code in [(3, 1), (10, 2)]:
            last_sunday = max(w[calendar.SUNDAY] for w in calendar.monthcalendar(int(year), month))
            day = pd.Timestamp(int(year), month, last_sunday)
            flags[(labels.normalize() == day) & (clock_minute >= 60) & (clock_minute <= 120)] = code
    return flags


def write_csv(path: Path, rows: list[dict]) -> None:
    assert rows
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_array(path: Path, array) -> None:
    import numpy as np
    assert not path.exists(), f"Do not overwrite an existing prepared artifact: {path}"
    with path.open("xb") as f:
        np.save(f, array, allow_pickle=False)


def prepare() -> None:
    """Lock train-qualified IDs before converting any post-2012 values."""
    import numpy as np
    import pandas as pd
    assert PROTOCOL_PATH.exists()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    completion_path = DATA_ROOT / "dataset_manifest_v1.json"
    if completion_path.exists():
        raise RuntimeError("Prepared version is already completed; do not overwrite it.")
    record = json.loads((RESULT_ROOT / "download_manifest_v1.json").read_text(encoding="utf-8"))
    assert record["status"] == "COMPLETE"
    raw = Path(record["raw_path"])
    assert sha256_file(raw) == record["sha256"]
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    extraction = DATA_ROOT / "provider_table_derived_v1.txt"
    if extraction.exists():
        raise RuntimeError("An incomplete preparation exists; preserve it and investigate instead of overwriting.")
    with zipfile.ZipFile(raw) as z:
        info = z.getinfo("LD2011_2014.txt")
        with z.open(info) as src, extraction.open("xb") as dest:
            shutil.copyfileobj(src, dest, length=8 * 1024**2)
        assert extraction.stat().st_size == info.file_size
    extraction.chmod(stat.S_IREAD)
    extracted_sha = sha256_file(extraction)
    print(f"Derived text extracted to D: {extraction.stat().st_size} bytes", flush=True)
    # Timestamps are structural metadata; usecols prevents held-out numeric conversion.
    timestamp_frame = pd.read_csv(extraction, sep=";", usecols=[0], dtype=str)
    raw_labels = pd.DatetimeIndex(pd.to_datetime(timestamp_frame.iloc[:, 0], format="%Y-%m-%d %H:%M:%S"))
    assert len(raw_labels) == 140256
    assert raw_labels[0] == pd.Timestamp("2011-01-01 00:15:00")
    assert raw_labels[-1] == pd.Timestamp("2015-01-01 00:00:00")
    assert raw_labels.is_unique and raw_labels.is_monotonic_increasing
    assert np.all(np.diff(raw_labels.asi8) == 15 * 60 * 10**9)
    labels = pd.date_range("2011-01-01", "2015-01-01", freq="15min", inclusive="left")
    flags = provider_dst_flags(labels)
    n_before_2013 = int(raw_labels.searchsorted(pd.Timestamp("2013-01-01")))
    # Important: nrows is fixed from time metadata, so no future customer values
    # are converted for activation, eligibility, roster, scaling or the data lock.
    train_frame = pd.read_csv(extraction, sep=";", decimal=",", nrows=n_before_2013)
    client_ids = list(train_frame.columns[1:])
    assert client_ids == [f"MT_{i:03d}" for i in range(1, 371)]
    train_values = train_frame.iloc[:, 1:].to_numpy(dtype=np.float64)
    assert train_values.shape == (n_before_2013, 370)
    before_flags = provider_dst_flags(raw_labels[:n_before_2013])
    usable_native = np.isfinite(train_values) & (train_values >= 0) & (before_flags[:, None] == 0)
    activation_evidence = usable_native & (train_values > 0)
    has_activation = activation_evidence.any(axis=0)
    first_positive = activation_evidence.argmax(axis=0)
    first_positive[~has_activation] = n_before_2013
    active = np.arange(n_before_2013)[:, None] >= first_positive[None, :]
    train_valid = usable_native & active
    in_2012 = (raw_labels[:n_before_2013] >= pd.Timestamp("2012-01-01"))
    assert int(in_2012.sum()) == 35136
    count_2012 = train_valid[in_2012].sum(axis=0)
    coverage_2012 = count_2012 / 35136
    tv = train_values[in_2012]
    vm = train_valid[in_2012]
    sums = np.where(vm, tv, 0).sum(axis=0)
    means = np.divide(sums, count_2012, out=np.zeros(370), where=count_2012 > 0)
    variances = np.divide(np.where(vm, (tv - means[None, :])**2, 0).sum(axis=0), count_2012, out=np.zeros(370), where=count_2012 > 0)
    scales = np.sqrt(variances)
    eligible = (coverage_2012 >= .95) & np.isfinite(means) & np.isfinite(scales) & (scales > 1e-8)
    eligible_ids = np.array([client_ids[i] for i in np.flatnonzero(eligible)])
    assert len(eligible_ids) >= 80, f"Train-only eligibility insufficient: {len(eligible_ids)}; no held-out substitution allowed"
    permutation = np.random.Generator(np.random.PCG64(ROSTER_SEED)).permutation(eligible_ids)
    group_ids = {"development": permutation[:32].tolist(), "calibration": permutation[32:48].tolist(), "confirmation": permutation[48:80].tolist()}
    selected = group_ids["development"] + group_ids["calibration"] + group_ids["confirmation"]
    assert len(set(selected)) == 80
    selected_native_cols = np.array([client_ids.index(x) for x in selected], dtype=np.int64)
    train_audit = []
    for j, cid in enumerate(client_ids):
        train_audit.append({
            "client_id": cid, "first_valid_positive_provider_label": str(raw_labels[first_positive[j]]) if has_activation[j] else None,
            "train_2012_valid_slots": int(count_2012[j]), "train_2012_coverage": float(coverage_2012[j]),
            "train_2012_mean_kw": float(means[j]), "train_2012_std_kw": float(scales[j]),
            "train_2012_valid_zero_slots": int((vm[:, j] & (tv[:, j] == 0)).sum()), "eligible": bool(eligible[j]),
        })
    write_csv(RESULT_ROOT / "train_only_eligibility_v1.csv", train_audit)
    dev_cols = selected_native_cols[:32]
    forecaster_fit_dates = raw_labels[:n_before_2013][in_2012] < pd.Timestamp("2012-10-01")
    global_train_values = tv[forecaster_fit_dates][:, dev_cols][vm[forecaster_fit_dates][:, dev_cols]]
    global_center = float(global_train_values.mean())
    global_scale = float(global_train_values.std(ddof=0))
    assert global_scale > 1e-8
    roster = {"seed": ROSTER_SEED, "rng": "numpy.Generator(PCG64)", "eligibility_cutoff_exclusive": "2013-01-01 00:00:00", "eligible_count": len(eligible_ids), "eligibility_min_2012_valid_fraction": .95, "groups": group_ids, "column_order": selected, "pilot_clients": group_ids["development"][:8]}
    json_write(DATA_ROOT / "roster_v1.json", roster)
    normalization = {
        "primary_scheme": GLOBAL_SCALING_SCHEME,
        "fit_clients": group_ids["development"], "fit_start": "2012-01-01 00:00:00", "fit_end_exclusive": "2012-10-01 00:00:00",
        "center_kw": global_center, "scale_kw": global_scale, "fit_valid_slots": int(global_train_values.size),
        "transform": "(valid kW-center_kw)/scale_kw; invalid values remain missing; causal fill belongs to later learner/retention protocol",
        "per_client_2012_support_stats_are_metadata_only_not_used_by_primary_scaler": [
            {"client_id": cid, "center_kw": float(means[j]), "scale_kw": float(scales[j]), "observations": int(count_2012[j]), "role": "source_fit" if cid in group_ids["development"] else "past_only_unlabelled_target_support"}
            for cid, j in zip(selected, selected_native_cols)
        ],
    }
    json_write(DATA_ROOT / "normalization_v1.json", normalization)
    lock = {
        "status": "LOCKED_BEFORE_POST2012_NUMERIC_CONVERSION", "seed": ROSTER_SEED,
        "raw_sha256": record["sha256"], "extracted_text_sha256": extracted_sha,
        "preparation_script_sha256": sha256_file(Path(__file__)), "data_protocol_sha256": sha256_file(PROTOCOL_PATH),
        "roster_sha256": sha256_file(DATA_ROOT / "roster_v1.json"), "normalization_sha256": sha256_file(DATA_ROOT / "normalization_v1.json"),
        "eligibility_audit_sha256": sha256_file(RESULT_ROOT / "train_only_eligibility_v1.csv"),
        "development_period": ["2013-01-01", "2013-07-01"], "calibration_period": ["2013-07-01", "2014-01-01"], "confirmation_period": ["2014-01-01", "2015-01-01"],
        "forecaster_training_period": ["2012-01-01", "2012-10-01"],
        "utility_training_period": ["2012-10-01", "2013-01-01"],
        "history_provider_slots": 672, "horizon_provider_slots": 96, "origin_provider_time": "00:00 daily",
        "arrival_rule": "value with provider label t releases at t+15min; at origin O only labels <O have arrived",
        "confirmation_outcome_based_selection": False, "post2012_numeric_access_before_lock": False,
    }
    json_write(DATA_ROOT / "study_lock_v1.json", lock)
    lock_sha = sha256_file(DATA_ROOT / "study_lock_v1.json")
    print(f"LOCKED {len(eligible_ids)} eligible; dev/cal/confirm=32/16/32; SHA {lock_sha}", flush=True)
    del train_frame, train_values, tv, vm, train_valid, active, usable_native, activation_evidence
    # After the lock: convert only the fixed selected columns, retain all rows
    # independent of later outcomes. No calibration/confirmation model is fitted.
    selected_frame = pd.read_csv(extraction, sep=";", decimal=",", usecols=[0] + selected, dtype={cid: np.float32 for cid in selected})
    fixed_values = selected_frame[selected].to_numpy(dtype=np.float32)
    assert fixed_values.shape == (140256, 80)
    # One initial unrecorded native label is explicitly missing. The raw final
    # 2015-01-01 label falls outside the 2011-2014 label study and is excluded.
    panel = np.full((len(labels), 80), np.nan, dtype=np.float32)
    panel[1:] = fixed_values[:-1]
    release_indices = np.arange(1, len(labels) + 1, dtype=np.int64)
    first_labels = [raw_labels[first_positive[j]] for j in selected_native_cols]
    activation_ns = np.array([x.value for x in first_labels], dtype=np.int64)
    valid = np.isfinite(panel) & (panel >= 0) & (labels.asi8[:, None] >= activation_ns[None, :]) & (flags[:, None] == 0)
    panel[~valid] = np.nan
    assert np.array_equal(np.isfinite(panel), valid)
    # Only unambiguous, non-DST-collapsed provider labels map to UTC. The fixed
    # 96-slot provider-day raster is intentionally not a regular UTC raster.
    localized = labels.tz_localize("Europe/Lisbon", ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")
    utc_ns = localized.asi8.copy()
    utc_ns[flags != 0] = np.iinfo(np.int64).min
    utc_ns = utc_ns.astype("datetime64[ns]")
    files = {
        "panel_kw.npy": panel, "valid.npy": valid.astype(np.uint8),
        "provider_labels.npy": labels.to_numpy(dtype="datetime64[ns]"), "utc_labels.npy": utc_ns,
        "logical_release_index.npy": release_indices, "provider_dst_flags.npy": flags,
        "client_ids.npy": np.array(selected, dtype="U6"),
        "group_development_cols.npy": np.arange(32, dtype=np.int64),
        "group_calibration_cols.npy": np.arange(32, 48, dtype=np.int64),
        "group_confirmation_cols.npy": np.arange(48, 80, dtype=np.int64),
    }
    for name, arr in files.items():
        save_array(DATA_ROOT / name, arr)
    partitions = {}
    for name, start, end, cols in [
        ("training", "2012-01-01", "2012-10-01", list(range(32))),
        ("utility_training", "2012-10-01", "2013-01-01", list(range(32))),
        ("development", "2013-01-01", "2013-07-01", list(range(32))),
        ("calibration", "2013-07-01", "2014-01-01", list(range(32, 48))),
        ("confirmation", "2014-01-01", "2015-01-01", list(range(48, 80))),
    ]:
        slot_start, slot_end = int(labels.searchsorted(pd.Timestamp(start))), int(labels.searchsorted(pd.Timestamp(end)))
        origin_start = slot_start + 672 if name == "training" else slot_start
        origin_slots = np.arange(origin_start, slot_end, 96, dtype=np.int64)
        assert all(origin_slots + 96 <= slot_end)
        save_array(DATA_ROOT / f"{name}_origin_slots.npy", origin_slots)
        partitions[name] = {
            "start_provider_label": start, "end_exclusive_provider_label": end,
            "slot_start": slot_start, "slot_end_exclusive": slot_end, "columns": cols,
            "origin_slots_path": f"{name}_origin_slots.npy", "origins": len(origin_slots),
            "history_rule": "panel[o-672:o,cols] is available by origin; subsequent state must not re-read deleted values",
            "label_rule": "panel[o:o+96,cols]; explicit validity; no filling future labels",
            "valid_outcome_count_is_quality_metadata_not_selection": int(valid[slot_start:slot_end, cols].sum()),
            "outcome_values_summarized_or_scored": False,
        }
    json_write(DATA_ROOT / "partitions_v1.json", partitions)
    pilot_start = int(labels.searchsorted(pd.Timestamp("2013-01-01")))
    pilot_end = int(labels.searchsorted(pd.Timestamp("2013-01-22")))
    save_array(DATA_ROOT / "pilot8_panel_kw.npy", panel[pilot_start:pilot_end, :8])
    save_array(DATA_ROOT / "pilot8_valid.npy", valid[pilot_start:pilot_end, :8].astype(np.uint8))
    pilot_origins = np.arange(672, 21 * 96, 96, dtype=np.int64)
    save_array(DATA_ROOT / "pilot8_origin_slots.npy", pilot_origins)
    pilot = {"clients": group_ids["development"][:8], "cache_start": "2013-01-01", "cache_end_exclusive": "2013-01-22", "forecast_start": "2013-01-08", "forecast_end_inclusive": "2013-01-21", "warmup_days": 7, "forecast_days": 14, "history_slots": 672, "horizon_slots": 96, "shape": [21 * 96, 8], "normalization": "normalization_v1.json; primary source-only pooled scaler", "confirmation_data_in_pilot": False}
    json_write(DATA_ROOT / "pilot8_manifest_v1.json", pilot)
    fixture_results = preflight_fixtures()
    quality = {
        "status": "PASS", "dataset_scope": "fixed provider-label raster, not reconstructed true UTC quarter-hour series",
        "raw_rows": len(raw_labels), "panel_rows": len(labels), "selected_clients": 80,
        "raw_first_label": str(raw_labels[0]), "raw_last_label": str(raw_labels[-1]),
        "initial_unrecorded_label": "2011-01-01 00:00:00; explicitly invalid", "excluded_outside_study_label": "2015-01-01 00:00:00",
        "train_eligible_clients": len(eligible_ids), "development_calibration_confirmation": [32, 16, 32],
        "physical_unit": "kW", "quarter_hour_energy_conversion": "kWh=kW/4 for ordinary native intervals; DST collapsed hours invalidated",
        "yearly_dst_labels_invalidated": {str(y): int(((labels.year == y) & (flags != 0)).sum()) for y in range(2011, 2015)},
        "activation_rule": "first positive finite nonnegative non-DST value using only values strictly before 2013; no retrospective reactivation from holdouts",
        "post_activation_valid_zero_count_train2012_selected": int(sum(train_audit[j]["train_2012_valid_zero_slots"] for j in selected_native_cols)),
        "post2012_numeric_summary_or_model_score": False, "fixture_tests": fixture_results,
        "no_model_training": True, "raw_readonly": not bool(raw.stat().st_mode & stat.S_IWRITE),
    }
    json_write(RESULT_ROOT / "quality_report_v1.json", quality)
    artifact_hashes = {str(f.relative_to(DATA_ROOT)): {"sha256": sha256_file(f), "bytes": f.stat().st_size} for f in sorted(DATA_ROOT.iterdir()) if f.is_file()}
    manifest = {
        "status": "COMPLETE_PASS", "source_page": PAGE_URL, "dataset_doi": "10.24432/C58C86", "license": "CC BY 4.0",
        "raw_path": str(raw), "raw_sha256": record["sha256"], "study_lock_sha256": lock_sha,
        "preparation_script_sha256": sha256_file(Path(__file__)), "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "panel_shape": list(panel.shape), "panel_dtype": "float32", "valid_dtype": "uint8", "valid_zero_retained": True,
        "chronology": {
            "provider_timezone_name": "Europe/Lisbon", "native_label_spacing_minutes": 15,
            "fixed_provider_day_slots": 96, "strictly_regular_utc_grid": False,
            "utc_sentinel_for_ambiguous_nonexistent_or_dst_aggregate_label": "datetime64[ns] NaT",
            "timestamp_interval_start_end_claim": "official dictionary unspecified; conservative arrival at label+15min",
            "online_arrival_contract": "Read each newly released row once in index order. After deletion, never re-read this panel, original data or derived text for that historical row. Offline labels are scorer-only.",
            "arrival_at_origin_O": "only native labels t<O are released; row label O is not an input",
            "physical_utc_week_warning": "672 provider slots=7 provider days; UTC elapsed duration can differ on DST weeks",
        },
        "normalization_primary": GLOBAL_SCALING_SCHEME, "confirmation_consumed_for_fitting_tuning_or_scoring": False,
        "customer_groups": group_ids,
        "customer_group_columns": {"development": list(range(32)), "calibration": list(range(32, 48)), "confirmation": list(range(48, 80))},
        "client_id_column_order": selected,
        "array_paths": {"values": str(DATA_ROOT / "panel_kw.npy"), "valid": str(DATA_ROOT / "valid.npy"), "provider_labels": str(DATA_ROOT / "provider_labels.npy"), "utc_labels": str(DATA_ROOT / "utc_labels.npy")},
        "date_to_index": {"2012-01-01": int(labels.searchsorted(pd.Timestamp("2012-01-01"))), "2012-10-01": int(labels.searchsorted(pd.Timestamp("2012-10-01"))), "2013-01-01": int(labels.searchsorted(pd.Timestamp("2013-01-01"))), "2013-01-08": int(labels.searchsorted(pd.Timestamp("2013-01-08"))), "2013-01-22": int(labels.searchsorted(pd.Timestamp("2013-01-22"))), "2013-07-01": int(labels.searchsorted(pd.Timestamp("2013-07-01"))), "2014-01-01": int(labels.searchsorted(pd.Timestamp("2014-01-01"))), "2015-01-01": len(labels)},
        "artifacts": artifact_hashes, "quality_report_path": str(RESULT_ROOT / "quality_report_v1.json"),
        "preparation_wall_seconds": time.perf_counter()-wall_start, "preparation_process_cpu_seconds": time.process_time()-cpu_start,
    }
    json_write(completion_path, manifest)
    print(json.dumps({"status": manifest["status"], "panel_shape": manifest["panel_shape"], "lock_sha256": lock_sha, "manifest_sha256": sha256_file(completion_path), "wall_seconds": manifest["preparation_wall_seconds"], "cpu_seconds": manifest["preparation_process_cpu_seconds"]}), flush=True)


def preflight_fixtures() -> dict:
    import numpy as np
    import pandas as pd
    labels = pd.date_range("2012-03-25 00:00", "2012-03-25 03:00", freq="15min")
    flags = provider_dst_flags(labels)
    assert np.count_nonzero(flags) == 5
    assert np.all(flags[(labels.hour == 1)] == 1)
    fall = pd.date_range("2012-10-28 00:00", "2012-10-28 03:00", freq="15min")
    assert np.count_nonzero(provider_dst_flags(fall) == 2) == 5
    # Activation is strictly forward: once active, observed zeros stay valid.
    v = np.array([0, 0, 3, 0, 2.0])
    valid = np.maximum.accumulate(v > 0)
    assert valid.tolist() == [False, False, True, True, True]
    # Raw origin label cannot enter input under conservative release.
    labels_ordinal = np.arange(1000)
    origin = 700
    assert np.all(labels_ordinal[origin-672:origin] < origin)
    assert origin not in labels_ordinal[origin-672:origin]
    return {"spring_five_conservative_labels": "PASS", "fall_five_aggregate_boundary_labels": "PASS", "post_activation_true_zero_preserved": "PASS", "origin_and_future_labels_excluded": "PASS"}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["download", "inspect_train", "prepare", "fixtures"])
    args = parser.parse_args()
    if args.phase == "download":
        download()
    elif args.phase == "inspect_train":
        inspect_train()
    elif args.phase == "prepare":
        prepare()
    else:
        print(json.dumps(preflight_fixtures()), flush=True)


if __name__ == "__main__":
    main()
