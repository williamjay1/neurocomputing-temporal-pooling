"""Recompute the frozen manuscript evidence without fitting another model.

Usage: python reproducibility/run.py smoke
       python reproducibility/run.py all
Every invocation writes a new directory.  No archived output is overwritten.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from fractions import Fraction as F
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.dont_write_bytecode = True
from legacy_runtime import LegacyRuntime, digest


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def compare_csv(actual, expected, atol=3e-12, rtol=3e-12):
    """Compare all columns; runtime and output-location fields are excluded."""
    import numpy as np
    a, b = rows(actual), rows(expected)
    assert len(a) == len(b), (actual, "row count")
    ignored = {"fit_wall_seconds", "fit_process_cpu_seconds"}
    differences = []
    for i, (aa, bb) in enumerate(zip(a, b)):
        assert set(aa) == set(bb), (actual, "columns")
        for key in aa:
            if key in ignored:
                continue
            try:
                x, y = float(aa[key]), float(bb[key])
            except (ValueError, TypeError):
                assert aa[key] == bb[key], (actual, i, key, aa[key], bb[key])
                continue
            if np.isnan(x) and np.isnan(y):
                continue
            assert np.isclose(x, y, atol=atol, rtol=rtol), (actual, i, key, x, y)
            differences.append(abs(x-y))
    return {"rows": len(a), "max_numeric_difference": max(differences, default=0.)}


def theory(runtime):
    """Invoke original rational functions; do not invoke Windows-only main IO."""
    m = runtime.load("140_r7_nonlinear_and_loss_boundaries_v1.py")
    out = runtime.output / "r7_theory_numeric"
    out.mkdir()
    base = m.states([(F(-1, 10), F(9, 10)), (F(9, 10), F(1, 10))],
                    [(F(1), F(9, 10)), (F(2), F(1, 10))])
    rr = m.target(base, F(9, 20), F(1, 10), lambda d: d)
    saved = json.loads((runtime.root / "results/r7_theory_v1/verified_results.json").read_text(encoding="utf-8"))
    checks, optimal = [], {}
    for access in ("C", "U", "magnitude", "raw"):
        optimal[access] = {}
        for loss in ("mse", "mae"):
            risk, _ = m.oracle(rr, access, loss)
            expected = F(saved["loss_case"]["risks"][access][loss]["risk"]["fraction"])
            assert risk == expected
            optimal[access][loss] = str(risk)
        lp = m.lp_mae(rr, access)
        assert abs(lp["risk"]-float(F(optimal[access]["mae"]))) < 1e-10
        checks.append({"case": "loss", "access": access, "LP_risk": lp["risk"]})
    base = m.states([(F(-1), F(1, 3)), (F(0), F(1, 3)), (F(1), F(1, 3))],
                    [(F(1), F(1, 3)), (F(2), F(1, 3)), (F(3), F(1, 3))])
    cases = {"nonlinear_persistence": lambda d: d+(d*d-F(2, 3))/2,
             "nonlinear_boundary": lambda d: d*d-F(2, 3)}
    curve = []
    for name, g in cases.items():
        moment = m.moments(base, g)
        for key, value in moment.items():
            assert value == F(saved["nonlinear"][name]["moments"][key]["fraction"])
        for i in range(101):
            alpha = F(i, 100)
            rr = m.target(base, alpha, F(1, 2), g)
            risk = {a: {loss: m.oracle(rr, a, loss)[0] for loss in ("mse", "mae")}
                    for a in ("C", "U", "magnitude", "raw")}
            gap = risk["U"]["mse"]-risk["C"]["mse"]
            assert gap == moment["W_D"]-2*alpha*moment["K"]-alpha*alpha*moment["H_g"]
            curve.append(dict(case=name, alpha=float(alpha), mse_C=float(risk["C"]["mse"]),
                              mse_U=float(risk["U"]["mse"]), mse_gap=float(gap),
                              mae_C=float(risk["C"]["mae"]), mae_U=float(risk["U"]["mae"]),
                              mae_gap=float(risk["U"]["mae"]-risk["C"]["mae"]), exact_mse_gap=str(gap)))
            if i in (0, 25, 50, 75, 100):
                for a in ("C", "U"):
                    lp = m.lp_mae(rr, a)
                    assert abs(lp["risk"]-float(risk[a]["mae"])) < 1e-10
                    checks.append({"case": name, "alpha": float(alpha), "access": a, "LP_risk": lp["risk"]})
    path = out / "nonlinear_risk_curves.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0])); w.writeheader(); w.writerows(curve)
    comparison = compare_csv(path, runtime.root / "results/r7_theory_v1/nonlinear_risk_curves.csv", atol=1e-14, rtol=1e-14)
    assert len(checks) == 24
    return {"status": "PASS", "exact_curve_rows": 202, "independent_LP_checks": 24,
            "same_target_risks": optimal, "comparison": comparison,
            "scope": "Original rational functions, not neural fits; Windows local-drive assertion in main is not invoked."}


def statistics_smoke(runtime):
    """Recompute bootstrap multiplicity and all40 crossed fits from full inputs."""
    import numpy as np
    from scipy.stats import norm
    m = runtime.load("139_r7_statistical_extensions_v1.py")
    results = {}
    for prefix in ("uci", "confirmation"):
        folder = runtime.root / "results/r6_independent_v1"
        with np.load(folder / (prefix+"_paired_week_sufficient_statistics.npz"), allow_pickle=False) as z:
            cells = list(zip(z["cell_adapter"].tolist(), z["cell_history"].tolist()))
            counts = z["observed_pair_counts"].copy()
            sums = z["three_seed_mean_error_sums"][z["metric_names"].tolist().index("endpoint_nmae")].copy()
        matrix = m.coefficient_matrix(cells)
        estimates = matrix @ (sums.sum(2)/counts.sum(1)[None, :]).mean(1)
        draws = m.paired_bootstrap(sums, counts) @ matrix.T
        with np.load(runtime.root / "results/r7_statistics_v1" / (prefix+"_matched_bootstrap_draws.npz"), allow_pickle=False) as z:
            discrepancy = float(np.abs(draws-z["bootstrap_draws"]).max())
            assert discrepancy < 2e-12
            assert np.max(np.abs(estimates-z["point_estimates"])) < 2e-12
        se = draws.std(0, ddof=1)
        critical = float(np.quantile(np.max(np.abs((draws[:, 1:]-estimates[None, 1:])/se[None, 1:]), axis=1), .95))
        intervals = rows(runtime.root / "results/r7_statistics_v1" / (prefix+"_multiplicity_intervals.csv"))
        assert all(abs(float(r["conditional_bootstrap_se"])-se[j]) < 2e-12 for j, r in enumerate(intervals))
        assert all(abs(float(r["maxT_family_critical"])-critical) < 2e-12 for r in intervals[1:])
        precision = rows(runtime.root / "results/r7_statistics_v1" / (prefix+"_design_precision.csv"))
        assert all(abs(float(r["approx_mdes80_two_sided_normal_alpha05"])-(norm.ppf(.975)+norm.ppf(.8))*se[j]) < 2e-12
                   for j, r in enumerate(precision))
        weekly = rows(runtime.root / "results/r7_statistics_v1" / (prefix+"_paired_customer_week_contrasts.csv"))
        grouped = {}
        for r in weekly:
            grouped.setdefault((r["contrast"], r["fit_seed"]), []).append(float(r["paired_week_mean_nmae_difference"]))
        oldfits = rows(runtime.root / "results/r7_statistics_v1" / (prefix+"_crossed_reml.csv"))
        assert len(oldfits) == len(grouped) == 20
        for r in oldfits:
            y = np.asarray(grouped[r["contrast"], r["fit_seed"]]).reshape(counts.shape)
            fit = m.crossed_reml(y)
            for key, value in fit.items():
                if isinstance(value, bool):
                    assert r[key] == str(value)
                else:
                    assert np.isclose(float(r[key]), value, atol=2e-11, rtol=2e-11), (prefix, key, value, r[key])
        results[prefix] = {"bootstrap_draws": 5000, "crossed_REML_fits": 20,
                           "max_bootstrap_discrepancy": discrepancy, "maxT_critical": critical}
    return {"status": "PASS", "phases": results, "new_fits": 0}


def r7_source(runtime):
    m = runtime.bind_output(runtime.load("144_r7_raw_independent_score_v1.py"), "r7_raw_independent_v1")
    summaries = []
    original = json.loads((runtime.root / "results/r7_raw_independent_v1/source_competence_gate.json").read_text(encoding="utf-8"))
    for branch, family_name in (("mean_std", "r7_raw_access_v1"), ("median_mad", "r7_raw_access_robust_v1")):
        family = runtime.root / "results" / family_name
        assert not (family / "calibration").exists() and not (family / "confirmation").exists()
        m.source(runtime.Path(family), branch)
        scored = rows(m.OUT / branch / "source_checkpoint_validation.csv")
        core = [float(r["independent_source_validation_nmae"]) for r in scored if r["adapter"] in m.CORE]
        refs = rows(m.OUT / branch / "source_quality_references.csv")
        threshold = float(next(r["source_validation_nmae"] for r in refs if r["reference"] == "daily_naive"))
        mean = sum(core)/len(core)
        old = next(b for b in original["branches"] if b["branch"] == branch)
        assert len(core) == 36 and len(scored) == 60 and mean >= threshold
        assert abs(mean-old["mean_independent_core_source_nmae"]) < 3e-6
        # The declared gate threshold is frozen. The independently recomputed
        # naive score uses float64 arithmetic rather than the producer's source
        # weighting, differing by approximately2.1e-10 in the observed runtime.
        # Verify that numerical discrepancy, then retain the original threshold.
        assert abs(threshold-old["frozen_daily_naive_threshold"]) < 1e-8
        summaries.append({"branch": branch, "source_checkpoints": 60, "core_mean_nmae": mean,
                          "frozen_daily_threshold": old["frozen_daily_naive_threshold"],
                          "independently_recomputed_daily_nmae": threshold, "gate": "FAIL_STOP_EXTERNAL_FORECASTS"})
    return {"status": "PASS", "source_only": True, "external_forecasts_generated": 0, "branches": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "all", "source", "calibration", "confirmation", "synthetic",
                                             "oracle", "engineering", "r7-source", "statistics", "theory", "figures", "residual"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output = (args.output or root / "reproducibility/runs" / stamp).resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory: " + str(output))
    if os.name == "nt" and output.drive.upper() != "D:":
        raise ValueError("On the author's workstation computational outputs must remain on D:")
    output.mkdir(parents=True)
    runtime = LegacyRuntime(root, output)
    started = time.perf_counter()
    report = {"status": "RUNNING", "command": args.command, "new_training_fits": 0, "stages": {},
              "runtime": {"python": sys.version, "packages": {p: metadata.version(p) for p in ("numpy", "scipy")}},
              "adapter_policy": "Only legacy Path/import resolution and fresh output roots; scientific functions/assertions unchanged."}
    path = output / "reproduction_report.json"
    try:
        cmd = args.command
        if cmd in ("smoke", "all", "engineering", "oracle", "source", "calibration", "confirmation", "synthetic"):
            m = runtime.bind_output(runtime.load("133_r6_independent_score_v1.py"), "r6_independent_v1")
            functions = []
            if cmd in ("smoke", "all", "engineering"): functions.append(("r6_engineering", m.engineering))
            if cmd in ("smoke", "all", "oracle"): functions.append(("r6_oracle", m.oracle))
            if cmd in ("smoke", "all", "source", "calibration", "confirmation"): functions.append(("r6_source", m.source_audit))
            if cmd in ("all", "synthetic"): functions.append(("r6_synthetic", m.synthetic))
            if cmd in ("all", "calibration"): functions.append(("r6_calibration", m.uci))
            if cmd in ("all", "confirmation"): functions.append(("r6_confirmation", lambda: m.uci("confirmation")))
            for name, function in functions:
                print("Running " + name, flush=True); stage_start = time.perf_counter(); function()
                report["stages"][name] = {"status": "PASS", "seconds": time.perf_counter()-stage_start}
            pairs = [("source_checkpoint_validation_independent.csv", 4e-6)]
            if cmd in ("all", "synthetic"): pairs.append(("synthetic_metrics_by_seed.csv", 3e-12))
            for prefix in (("uci",) if cmd == "calibration" else ("confirmation",) if cmd == "confirmation" else ("uci", "confirmation") if cmd == "all" else ()):
                pairs += [(prefix+"_metrics_summary.csv", 3e-12), (prefix+"_contrasts_conditional_bootstrap.csv", 3e-12)]
            for name, tol in pairs:
                if (m.OUT / name).exists():
                    report["stages"]["compare_"+name] = compare_csv(m.OUT/name, root/"results/r6_independent_v1"/name, atol=tol, rtol=tol)
        if cmd in ("smoke", "all", "theory"):
            report["stages"]["r7_exact_theory"] = theory(runtime)
        if cmd in ("smoke", "all"):
            report["stages"]["r7_statistics"] = statistics_smoke(runtime)
        if cmd in ("smoke", "all", "r7-source"):
            report["stages"]["r7_source"] = r7_source(runtime)
        if cmd == "statistics":
            m = runtime.bind_output(runtime.load("139_r7_statistical_extensions_v1.py"), "r7_statistics_v1")
            m.ARTICLE = runtime.Path(output/"r7_article_v1")
            m.selftest(); m.run()
            for prefix in ("uci", "confirmation"):
                for suffix in ("multiplicity_intervals.csv", "design_precision.csv", "crossed_reml.csv"):
                    name = prefix+"_"+suffix
                    report["stages"][name] = compare_csv(m.OUT/name, root/"results/r7_statistics_v1"/name, atol=2e-11, rtol=2e-11)
        if cmd == "residual":
            # Script137 imports133 explicitly.  Its loader is relocated too.
            m = runtime.bind_output(runtime.load("137_r6_independent_residual_response_v1.py"), "r6_independent_v1")
            m.run()
            report["stages"]["residual"] = compare_csv(m.OUT/"source_residual_response_independent.csv",
                                                        root/"results/r6_independent_v1/source_residual_response_independent.csv", atol=2e-7, rtol=2e-7)
        if cmd == "figures":
            m = runtime.bind_output(runtime.load("146_r7_nature_figures_v1.py"), "r7_figures_v1")
            m.FIGURES = runtime.Path(output/"figures")
            m.CAPTION_DIR = runtime.Path(output/"r7_article_v1")
            (m.CAPTION_DIR/"tables").mkdir(parents=True)
            for p in (root/"results/r7_article_v1/tables").glob("*.csv"):
                shutil.copy2(p, m.CAPTION_DIR/"tables"/p.name)
            m.main()
            report["stages"]["figures"] = {"status": "PASS_GEOMETRY", "manual_visual_inspection": "NOT_PERFORMED_BY_THIS_RUNNER"}
        report["status"] = "PASS_EXECUTED_NO_NEW_TRAINING"
    except BaseException as exc:
        report["status"] = "FAIL"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["seconds"] = time.perf_counter()-started
        report["immutable_original_script_sha256"] = runtime.loaded
        report["runtime_path_relocations"] = runtime.path_mappings
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        print("Reproduction report: " + str(path), flush=True)


if __name__ == "__main__":
    main()
