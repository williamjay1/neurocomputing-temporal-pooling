"""Resume frozen 125 preparation after a pre-read pandas usecols rejection.

Only repair: replace the mixed positional time-column selector 0 with its
actual string header. Reuse locked roster/scaling/activation and original tail.
No training, new qualification, future summaries or model selection.
"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import inspect
import json
import sys
import textwrap
import time

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
root = Path(r"D:\MLWork\Neurocomputing_TailTransfer")
data = root / "datasets" / "upgrade5_uci_v1"
results = root / "results" / "upgrade5_data_v1"
script = root / "scripts" / "125_upgrade5_uci_prepare_v1.py"
lock_path = data / "study_lock_v1.json"
lock = json.loads(lock_path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


assert sha(script) == lock["preparation_script_sha256"]
assert sha(root / "docs" / "UPGRADE5_DATA_PROTOCOL_20261002.md") == lock["data_protocol_sha256"]
assert sha(data / "roster_v1.json") == lock["roster_sha256"]
assert sha(data / "normalization_v1.json") == lock["normalization_sha256"]
assert sha(results / "train_only_eligibility_v1.csv") == lock["eligibility_audit_sha256"]
assert not (data / "dataset_manifest_v1.json").exists()
assert not (data / "panel_kw.npy").exists()

spec = importlib.util.spec_from_file_location("upgrade5_frozen125", script)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
source = inspect.getsource(mod.prepare)
lines = source.splitlines()
tail_start = next(i for i, line in enumerate(lines) if line.lstrip().startswith("selected_frame = pd.read_csv("))
tail = textwrap.dedent("\n".join(lines[tail_start:]))
record = json.loads((results / "download_manifest_v1.json").read_text(encoding="utf-8"))
raw = Path(record["raw_path"])
extraction = data / "provider_table_derived_v1.txt"
assert sha(raw) == lock["raw_sha256"]
assert sha(extraction) == lock["extracted_text_sha256"]
roster = json.loads((data / "roster_v1.json").read_text(encoding="utf-8"))
with (results / "train_only_eligibility_v1.csv").open(encoding="utf-8", newline="") as f:
    train_audit = list(csv.DictReader(f))
for row in train_audit:
    row["train_2012_valid_zero_slots"] = int(row["train_2012_valid_zero_slots"])
client_ids = [r["client_id"] for r in train_audit]
eligible_ids = [r["client_id"] for r in train_audit if r["eligible"] == "True"]
assert len(eligible_ids) == roster["eligible_count"] == 321
labels = pd.date_range("2011-01-01", "2015-01-01", freq="15min", inclusive="left")
native_read_csv = pd.read_csv
header = native_read_csv(extraction, sep=";", nrows=0).columns[0]
timestamp_frame = native_read_csv(extraction, sep=";", usecols=[header], dtype=str)
raw_labels = pd.DatetimeIndex(pd.to_datetime(timestamp_frame.iloc[:, 0], format="%Y-%m-%d %H:%M:%S"))
n_before_2013 = int(raw_labels.searchsorted(pd.Timestamp("2013-01-01")))
first_positive = np.array([
    int(raw_labels.searchsorted(pd.Timestamp(r["first_valid_positive_provider_label"])))
    if r["first_valid_positive_provider_label"] else n_before_2013
    for r in train_audit
], dtype=np.int64)
group_ids = roster["groups"]
selected = roster["column_order"]
selected_native_cols = np.array([client_ids.index(cid) for cid in selected], dtype=np.int64)
assert all(first_positive[selected_native_cols] < n_before_2013)
failure = {
    "status": "FAILED_PRESERVED", "phase": "after_train_lock_before_selected_post2012_numeric_read",
    "error": "pandas ValueError: usecols must be all strings/all integers or callable; original mixed 0 and selected client IDs",
    "original_script_sha256": lock["preparation_script_sha256"], "original_lock_sha256": sha(lock_path),
    "post2012_values_converted_before_failure": False,
    "failed_attempt_cpu_seconds": None, "failed_attempt_wall_seconds": None,
    "timing_note": "Original attempt did not reach its final timer report; unavailable failure timing is not imputed.",
}
mod.json_write(results / "failed_prepare_attempt1_v1.json", failure)
repair = {
    "status": "SEALED_BEFORE_RESUMED_POST2012_CONVERSION",
    "wrapper_path": str(Path(__file__)), "wrapper_sha256": sha(Path(__file__)),
    "original_script_path": str(script), "original_script_sha256": sha(script),
    "original_lock_sha256": sha(lock_path), "protocol_sha256": lock["data_protocol_sha256"],
    "roster_sha256": lock["roster_sha256"], "normalization_sha256": lock["normalization_sha256"],
    "raw_sha256": lock["raw_sha256"], "original_tail_sha256": hashlib.sha256(tail.encode()).hexdigest(),
    "sole_read_csv_repair": {"from_mixed_integer_selector": 0, "to_actual_string_header": header},
    "roster_eligibility_scaling_dates_and_masks_changed": False, "reselection": False,
    "prior_failure_record_sha256": sha(results / "failed_prepare_attempt1_v1.json"),
}
repair_path = results / "postlock_repair_manifest_v1.json"
mod.json_write(repair_path, repair)
repair_sha = sha(repair_path)


def repaired_read_csv(*args, **kwargs):
    cols = kwargs.get("usecols")
    if isinstance(cols, list) and cols and cols[0] == 0 and all(isinstance(c, str) for c in cols[1:]):
        kwargs["usecols"] = [header] + cols[1:]
    return native_read_csv(*args, **kwargs)


pd.read_csv = repaired_read_csv
namespace = dict(mod.__dict__)
namespace.update(
    np=np, pd=pd, record=record, raw=raw, extraction=extraction,
    raw_labels=raw_labels, labels=labels, flags=mod.provider_dst_flags(labels),
    first_positive=first_positive, client_ids=client_ids, train_audit=train_audit,
    eligible_ids=eligible_ids, group_ids=group_ids, selected=selected,
    selected_native_cols=selected_native_cols, lock_sha=sha(lock_path),
    completion_path=data / "dataset_manifest_original_tail_v1.json",
    wall_start=time.perf_counter(), cpu_start=time.process_time(),
)
exec(compile(tail, str(script) + "::original-postlock-tail", "exec"), namespace)
pd.read_csv = native_read_csv
manifest = namespace["manifest"]
manifest["preparation_original_tail_manifest_sha256"] = sha(data / "dataset_manifest_original_tail_v1.json")
manifest["executed_io_repair"] = {
    "wrapper_path": str(Path(__file__)), "wrapper_sha256": repair["wrapper_sha256"],
    "repair_manifest_path": str(repair_path), "repair_manifest_sha256": repair_sha,
    "original_preparation_code_immutable": True,
    "frozen_selection_unchanged": True,
}
manifest["prior_failed_attempt_retained"] = failure
manifest["preparation_timing_scope"] = "resumed selected-value conversion through arrays/metadata/hash completion only; initial extraction/eligibility failed-attempt time not available and not included"
mod.json_write(data / "dataset_manifest_v1.json", manifest)
print("FINAL_REPAIRED_COMPLETE_PASS", sha(data / "dataset_manifest_v1.json"), flush=True)
