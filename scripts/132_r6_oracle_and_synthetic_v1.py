"""Constructive R6 Gaussian access oracle and fixed-family neural check.

Synthetic centered/residual variables, not measured or simulated physical loads.
Prepare/seal before formal fitting. Test reading is gated on all 30 completed
checkpoints and a root-authored protocol bound by the experiment lock.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import log_ndtr, ndtr, roots_hermitenorm, roots_legendre

ROOT = Path("D:/MLWork/Neurocomputing_TailTransfer")
OUT = ROOT / "results/r6_oracle_v1"
CODE = Path(__file__).resolve()
ADAPTERS = ("direct", "max_repeat", "avg_repeat", "mean_abs", "mean_signed")
SEEDS = (20261001, 20261002, 20261003)
SCENARIOS = ((0.0, 0.5), (1.0, 0.5))
SETTINGS = {
    "study_type": "constructive synthetic mechanism after historical exploration",
    "physical_measurements": False,
    "sigma_c": 1.0,
    "sigma_a": 1.0,
    "noise_standard_deviation_each_coordinate": 0.1,
    "oracle_sigma_a_grid": [0.25, 1.0, 2.0],
    "oracle_alpha_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
    "oracle_beta_grid": [0.0, 0.5],
    "quadrature_resolutions": [96, 192],
    "half_normal_positive_cutoff_sigma": 12.0,
    "data_counts": {"train": 20000, "validation": 10000, "test": 100000},
    "data_seeds": {"train": 13220261001, "validation": 13220261002, "test": 13220261003},
    "latent_columns": ["C", "A", "epsilon_c", "epsilon_a"],
    "input_pair": ["C-A", "C+A"],
    "future_mean": "C + alpha * (abs(A)-sigma_a*sqrt(2/pi)) + epsilon_c",
    "future_detail": "beta*A + epsilon_a",
    "future_endpoints": ["future_mean-future_detail", "future_mean+future_detail"],
    "adapters": list(ADAPTERS),
    "neural_scenarios": [{"alpha": a, "beta": b} for a, b in SCENARIOS],
    "model_seeds": list(SEEDS),
    "hidden_widths": [64, 64],
    "output_width": 2,
    "parameter_count": 4482,
    "updates": 2000,
    "batch_size": 256,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "validation_every_updates": 200,
    "checkpoint_criterion": "validation pair-average MSE, earliest exact tie",
    "primary_score": "test pair-average endpoint MSE",
    "secondary_score": "test endpoint MAE",
    "normalization": "none; no clipping; fixed synthetic distributions",
    "torch_threads": 2,
    "test_read_before_all_training_complete": False,
    "score_before_all_predictions_complete": False,
    "same_minibatch_rows_per_model_seed": True,
    "direct_signed_initial_functions_matched": True,
    "mean_abs_uses_signed_rotated_first_weight": True,
    "neural_bayes_attainment_claimed": False,
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict, immutable: bool = True) -> None:
    if immutable and path.exists():
        raise FileExistsError(f"Preserve existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def normal_pdf(x):
    return np.exp(-0.5 * np.asarray(x) ** 2) / math.sqrt(2 * math.pi)


def posterior_variance(z, sigma_c: float, sigma_a: float):
    """Var(|A| | C+|A|=z); posterior is a lower-truncated Gaussian."""
    total = sigma_c**2 + sigma_a**2
    tau2 = sigma_c**2 * sigma_a**2 / total
    t = np.asarray(z, dtype=np.float64) * sigma_a / (sigma_c * math.sqrt(total))
    mills = np.exp(-0.5 * t**2 - 0.5 * math.log(2 * math.pi) - log_ndtr(t))
    v = 1.0 - mills * (mills + t)
    if np.any(~np.isfinite(v)) or np.min(v) < -1e-9:
        raise ArithmeticError("Truncated-normal variance numerical failure")
    # Roundoff guard only; reports state when this is activated.
    return tau2 * np.maximum(v, 0.0)


def oracle_w_product(sigma_c: float, sigma_a: float, n: int):
    """Positive half-normal integration times independent Gaussian quadrature."""
    hnodes, hweights = roots_hermitenorm(n)
    lnodes, lweights = roots_legendre(n)
    a = 6.0 * sigma_a * (lnodes + 1.0)
    aw = lweights * 6.0 * sigma_a * 2.0 * normal_pdf(a / sigma_a) / sigma_a
    c = sigma_c * hnodes
    cw = hweights / math.sqrt(2 * math.pi)
    values = posterior_variance(a[:, None] + c[None, :], sigma_c, sigma_a)
    return float(aw @ values @ cw), float(aw.sum()), float(cw.sum())


def oracle_w_density(sigma_c: float, sigma_a: float):
    """Separate one-dimensional integral of the skew-normal density."""
    s = math.sqrt(sigma_c**2 + sigma_a**2)

    def integrand(z):
        t = z * sigma_a / (sigma_c * s)
        density = 2.0 * float(normal_pdf(z / s)) / s * float(ndtr(t))
        return density * float(posterior_variance(z, sigma_c, sigma_a))

    value, estimated_error = quad(integrand, -12 * s, 12 * s, epsabs=2e-11, epsrel=2e-11, limit=300)
    return value, estimated_error


def oracle_rows():
    rows, checks = [], []
    for sa in SETTINGS["oracle_sigma_a_grid"]:
        wlow, half_mass_low, c_mass_low = oracle_w_product(1.0, sa, 96)
        whigh, half_mass_high, c_mass_high = oracle_w_product(1.0, sa, 192)
        cross, error = oracle_w_density(1.0, sa)
        v = sa**2 * (1.0 - 2.0 / math.pi)
        if not 0 < whigh < v:
            raise AssertionError("Oracle requires nondegenerate 0<W<V")
        if max(abs(whigh - wlow), abs(whigh - cross)) > 2e-8:
            raise AssertionError("Quadrature cross-check tolerance exceeded")
        threshold = math.sqrt(whigh) / (math.sqrt(whigh) + math.sqrt(v))
        checks.append({
            "sigma_c": 1.0, "sigma_a": sa, "V": v,
            "W_96": wlow, "W_192": whigh, "W_density_quad": cross,
            "density_quad_estimated_error": error,
            "resolution_absolute_difference": abs(whigh - wlow),
            "independent_integral_absolute_difference": abs(whigh - cross),
            "alpha_switch": threshold,
            "positive_half_normal_mass_96": half_mass_low,
            "positive_half_normal_mass_192": half_mass_high,
            "gaussian_mass_96": c_mass_low, "gaussian_mass_192": c_mass_high,
            "tail_bound_scope": "12-sigma truncation; estimated quadrature accuracy, not a closed-form W",
        })
        for alpha in SETTINGS["oracle_alpha_grid"]:
            for beta in SETTINGS["oracle_beta_grid"]:
                common = 0.02 + beta**2 * sa**2
                risks = {
                    "direct": 0.02, "mean_signed": 0.02,
                    "mean_abs": common,
                    "avg_repeat": common + alpha**2 * v,
                    "max_repeat": common + (alpha - 1.0)**2 * whigh,
                }
                for adapter, risk in risks.items():
                    rows.append({
                        "sigma_c": 1.0, "sigma_a": sa, "alpha": alpha, "beta": beta,
                        "adapter": adapter, "bayes_pair_mse": risk,
                        "V": v, "W": whigh, "alpha_switch": threshold,
                        "max_minus_avg_bayes_pair_mse": (alpha - 1.0)**2 * whigh - alpha**2 * v,
                        "bayes_mae_floor_claimed": False,
                    })
    return rows, checks


def latent_draws(n: int, seed: int):
    r = np.random.default_rng(seed)
    x = r.standard_normal((n, 4)).astype(np.float64)
    x[:, 2:] *= 0.1
    return x


def pair_data(latents, alpha: float, beta: float):
    c, a, ec, ea = np.asarray(latents, dtype=np.float64).T
    x = np.column_stack((c - a, c + a))
    mean = c + alpha * (np.abs(a) - math.sqrt(2.0 / math.pi)) + ec
    detail = beta * a + ea
    y = np.column_stack((mean - detail, mean + detail))
    return x, y


def adapter_numpy(x, name):
    x = np.asarray(x)
    if name == "direct":
        return x
    if name == "max_repeat":
        return np.repeat(x.max(axis=1, keepdims=True), 2, axis=1)
    if name == "avg_repeat":
        return np.repeat(x.mean(axis=1, keepdims=True), 2, axis=1)
    c = (x[:, 0] + x[:, 1]) / math.sqrt(2.0)
    a = (x[:, 1] - x[:, 0]) / math.sqrt(2.0)
    return np.column_stack((c, np.abs(a) if name == "mean_abs" else a))


def torch_setup(seed: int):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    return torch


def model_factory(seed: int, name: str):
    torch = torch_setup(seed)
    from torch import nn
    net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
    if name in ("mean_signed", "mean_abs"):
        q = torch.tensor([[1., 1.], [-1., 1.]]) / math.sqrt(2.0)
        with torch.no_grad():
            net[0].weight.copy_(net[0].weight @ q.T)
    assert sum(p.numel() for p in net.parameters()) == 4482
    return net


def preflight():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "preflight.json").exists():
        raise FileExistsError("Preserve original preflight")
    started = time.perf_counter()
    orows, checks = oracle_rows()
    l = latent_draws(2048, 1329001)
    x, y = pair_data(l, 0.0, 0.5)
    torch = torch_setup(20261001)
    raw, signed = model_factory(20261001, "direct"), model_factory(20261001, "mean_signed")
    with torch.no_grad():
        p0 = raw(torch.tensor(x.astype(np.float32))).numpy()
        p1 = signed(torch.tensor(adapter_numpy(x, "mean_signed").astype(np.float32))).numpy()
    rotation_error = float(np.max(np.abs(p0 - p1)))
    if rotation_error > 2e-6:
        raise AssertionError("Direct/signed initial function mismatch")
    xx = torch.tensor(x.astype(np.float32))
    yy = torch.tensor(y.astype(np.float32))
    opt = torch.optim.Adam(raw.parameters(), lr=0.001)
    rows = np.random.default_rng(1329002)
    fit_start = time.perf_counter()
    for _ in range(100):
        ix = rows.integers(0, len(x), size=256)
        opt.zero_grad()
        loss = (raw(xx[ix]) - yy[ix]).square().mean()
        loss.backward()
        opt.step()
    hundred_steps = time.perf_counter() - fit_start
    # Explicit synthetic identities/adapter information fixtures.
    m = x.mean(1)
    a = (x[:, 1] - x[:, 0]) / 2
    assert np.allclose(x.max(1), m + np.abs(a), atol=1e-12)
    swapped = x[:, ::-1]
    assert np.array_equal(adapter_numpy(x, "max_repeat"), adapter_numpy(swapped, "max_repeat"))
    assert np.allclose(adapter_numpy(x, "mean_abs"), adapter_numpy(swapped, "mean_abs"))
    direct_prediction = np.column_stack((l[:, 0] - 0.5 * l[:, 1], l[:, 0] + 0.5 * l[:, 1]))
    e = direct_prediction - y
    mean_error = e.mean(1)
    detail_error = (e[:, 1] - e[:, 0]) / 2
    assert np.allclose((e * e).mean(1), mean_error**2 + detail_error**2, atol=1e-12)
    report = {
        "status": "PASS_SOURCE_ONLY_SYNTHETIC_PREFLIGHT",
        "code_sha256": sha(CODE), "settings": SETTINGS,
        "test_arrays_read": False, "formal_models_fit": 0,
        "initial_direct_signed_maximum_function_difference": rotation_error,
        "all_parameter_counts": 4482,
        "hundred_update_wall_seconds": hundred_steps,
        "estimated_30_fit_update_seconds_linear": hundred_steps * 20 * 30,
        "preflight_wall_seconds": time.perf_counter() - started,
        "oracle_checks": checks,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "resource_change_required": False,
    }
    write_json(OUT / "preflight.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("settings", "oracle_checks")}))


def prepare():
    if not (OUT / "preflight.json").exists():
        raise RuntimeError("Preflight must complete before preparation")
    if (OUT / "preparation_manifest.json").exists():
        raise FileExistsError("Preserve completed preparation")
    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, n in SETTINGS["data_counts"].items():
        path = OUT / f"latent_{split}.npy"
        if path.exists():
            raise FileExistsError(path)
        np.save(path, latent_draws(n, SETTINGS["data_seeds"][split]), allow_pickle=False)
        files[split] = {"path": str(path), "sha256": sha(path), "rows": n, "dtype": "float64", "columns": 4}
    write_json(OUT / "preparation_manifest.json", {
        "status": "FIXED_SYNTHETIC_DRAWS_COMPLETE_UNSCORED",
        "code_sha256": sha(CODE), "preflight_sha256": sha(OUT / "preflight.json"),
        "files": files, "settings": SETTINGS,
        "test_generated_but_not_read_by_fitting": True,
    })
    print("Synthetic draws prepared; no formal fitting or test score.")


def seal(protocol_path: Path):
    if not protocol_path.is_file():
        raise FileNotFoundError("Root-authored protocol must exist")
    pm = json.loads((OUT / "preparation_manifest.json").read_text(encoding="utf-8"))
    if pm["code_sha256"] != sha(CODE):
        raise RuntimeError("Code changed after preparation")
    files = {
        "code": {"path": str(CODE), "sha256": sha(CODE)},
        "root_protocol": {"path": str(protocol_path.resolve()), "sha256": sha(protocol_path)},
        "preflight": {"path": str(OUT / "preflight.json"), "sha256": sha(OUT / "preflight.json")},
        "preparation": {"path": str(OUT / "preparation_manifest.json"), "sha256": sha(OUT / "preparation_manifest.json")},
    }
    for split, binding in pm["files"].items():
        if sha(Path(binding["path"])) != binding["sha256"]:
            raise RuntimeError("Prepared draw hash mismatch")
        files[f"data_{split}"] = {"path": binding["path"], "sha256": binding["sha256"]}
    write_json(OUT / "experiment_lock.json", {
        "status": "LOCKED_BEFORE_FORMAL_FITTING_AND_TEST_SCORING",
        "prospective_relative_to_real_outcomes": False,
        "constructed_after_historical_exploration": True,
        "settings": SETTINGS, "files": files, "formal_fits": 30,
    })
    print("Experiment lock sealed:", sha(OUT / "experiment_lock.json"))


def lock():
    path = OUT / "experiment_lock.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["settings"] != SETTINGS:
        raise RuntimeError("Settings differ from sealed scope")
    for entry in value["files"].values():
        if sha(Path(entry["path"])) != entry["sha256"]:
            raise RuntimeError(f"Changed sealed input: {entry['path']}")
    return value, sha(path)


def jobs():
    return [(a, b, name, seed) for a, b in SCENARIOS for name in ADAPTERS for seed in SEEDS]


def job_name(alpha, beta, adapter, seed):
    return f"alpha{int(alpha)}_beta05_{adapter}_seed{seed}"


def predict_net(net, x, chunk=8192):
    import torch
    parts = []
    net.eval()
    with torch.no_grad():
        for start in range(0, len(x), chunk):
            parts.append(net(torch.tensor(x[start:start + chunk], dtype=torch.float32)).numpy())
    return np.concatenate(parts)


def train():
    _, lh = lock()
    if (OUT / "training_completion.json").exists():
        raise FileExistsError("All training already complete")
    # Only training and validation values are read here.
    lt = np.load(OUT / "latent_train.npy", allow_pickle=False)
    lv = np.load(OUT / "latent_validation.npy", allow_pickle=False)
    proofs = {}
    for alpha, beta, adapter, seed in jobs():
        label = job_name(alpha, beta, adapter, seed)
        report_path = OUT / "models" / f"{label}.json"
        checkpoint = OUT / "models" / f"{label}.npz"
        trace_path = OUT / "models" / f"{label}_trace.csv"
        if report_path.exists():
            prior = json.loads(report_path.read_text(encoding="utf-8"))
            if prior["lock_sha256"] != lh or sha(checkpoint) != prior["checkpoint_sha256"] or sha(trace_path) != prior["trace_sha256"]:
                raise RuntimeError("Existing model provenance mismatch")
            proofs[label] = {"report_path": str(report_path), "report_sha256": sha(report_path)}
            continue
        if checkpoint.exists() or trace_path.exists():
            raise RuntimeError("Partial job preserved; do not silently overwrite")
        OUT.joinpath("models").mkdir(parents=True, exist_ok=True)
        x, y = pair_data(lt, alpha, beta)
        vx, vy = pair_data(lv, alpha, beta)
        x = adapter_numpy(x, adapter).astype(np.float32)
        vx = adapter_numpy(vx, adapter).astype(np.float32)
        torch = torch_setup(seed)
        net = model_factory(seed, adapter)
        initial_state = {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}
        xt = torch.tensor(x)
        yt = torch.tensor(y.astype(np.float32))
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        row_rng = np.random.default_rng(seed + 132000)
        best, best_step, best_state = math.inf, None, None
        trace = []
        started_wall, started_cpu = time.perf_counter(), time.process_time()
        for step in range(1, 2001):
            net.train()
            ix = row_rng.integers(0, len(x), size=256)
            optimizer.zero_grad()
            loss = (net(xt[ix]) - yt[ix]).square().mean()
            loss.backward()
            optimizer.step()
            if step % 200 == 0:
                vp = predict_net(net, vx).astype(np.float64)
                vmse = float(np.mean((vp - vy)**2))
                trace.append({"update": step, "training_batch_mse": float(loss.detach()), "validation_pair_mse": vmse})
                if vmse < best:
                    best, best_step = vmse, step
                    best_state = {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}
        wall, cpu = time.perf_counter() - started_wall, time.process_time() - started_cpu
        assert best_state is not None
        np.savez(checkpoint, **best_state)
        write_csv(trace_path, trace)
        ih = hashlib.sha256()
        for k, v in initial_state.items():
            ih.update(k.encode())
            ih.update(v.tobytes())
        write_json(report_path, {
            "status": "COMPLETE_SOURCE_SELECTED_CHECKPOINT_TEST_UNREAD",
            "lock_sha256": lh, "alpha": alpha, "beta": beta, "adapter": adapter, "seed": seed,
            "parameter_count": 4482, "updates_run": 2000, "selected_update": best_step,
            "best_validation_pair_mse": best, "same_seed_batch_row_rule": "default_rng(seed+132000)",
            "initial_state_hash": ih.hexdigest(), "checkpoint_sha256": sha(checkpoint),
            "trace_sha256": sha(trace_path), "fit_wall_seconds": wall, "fit_cpu_seconds": cpu,
            "test_values_read": False, "neural_bayes_attainment_claimed": False,
        })
        proofs[label] = {"report_path": str(report_path), "report_sha256": sha(report_path)}
        print(json.dumps({"job": label, "completed": len(proofs), "validation_pair_mse": best, "wall": wall}), flush=True)
    write_json(OUT / "training_completion.json", {
        "status": "ALL_30_SOURCE_SELECTED_CHECKPOINTS_COMPLETE_TEST_UNREAD",
        "jobs": 30, "lock_sha256": lh, "reports": proofs,
    })


def training_gate():
    _, lh = lock()
    p = OUT / "training_completion.json"
    c = json.loads(p.read_text(encoding="utf-8"))
    if c["status"] != "ALL_30_SOURCE_SELECTED_CHECKPOINTS_COMPLETE_TEST_UNREAD" or c["jobs"] != 30 or c["lock_sha256"] != lh:
        raise RuntimeError("Training completion gate failed")
    if set(c["reports"]) != {job_name(*j) for j in jobs()}:
        raise RuntimeError("Incomplete fixed model family")
    for b in c["reports"].values():
        if sha(Path(b["report_path"])) != b["report_sha256"]:
            raise RuntimeError("Changed model report")
        r = json.loads(Path(b["report_path"]).read_text(encoding="utf-8"))
        label = Path(b["report_path"]).stem
        if r["parameter_count"] != 4482 or r["updates_run"] != 2000 or not r["test_values_read"] is False:
            raise RuntimeError("Model scope changed")
        if sha(OUT / "models" / f"{label}.npz") != r["checkpoint_sha256"] or sha(OUT / "models" / f"{label}_trace.csv") != r["trace_sha256"]:
            raise RuntimeError("Checkpoint/trace changed")
    return lh, sha(p)


def forecast():
    lh, th = training_gate()
    if (OUT / "prediction_completion.json").exists():
        raise FileExistsError("Predictions already complete")
    latent = np.load(OUT / "latent_test.npy", allow_pickle=False)
    import torch
    proofs = {}
    OUT.joinpath("predictions").mkdir(parents=True, exist_ok=True)
    for alpha, beta, adapter, seed in jobs():
        label = job_name(alpha, beta, adapter, seed)
        path = OUT / "predictions" / f"{label}.npz"
        if path.exists():
            raise FileExistsError("Preserve partial forecast family")
        x, _ = pair_data(latent, alpha, beta)
        x = adapter_numpy(x, adapter).astype(np.float32)
        net = model_factory(seed, adapter)
        with np.load(OUT / "models" / f"{label}.npz", allow_pickle=False) as data:
            net.load_state_dict({k: torch.tensor(data[k]) for k in net.state_dict()})
        started = time.perf_counter()
        pred = predict_net(net, x)
        wall = time.perf_counter() - started
        np.savez(path, predictions=pred, row_indices=np.arange(len(latent), dtype=np.int64))
        proofs[label] = {"path": str(path), "sha256": sha(path), "forecast_wall_seconds": wall}
    write_json(OUT / "prediction_completion.json", {
        "status": "ALL_30_FIXED_FAMILY_TEST_FORECASTS_COMPLETE_UNSCORED",
        "jobs": 30, "lock_sha256": lh, "training_completion_sha256": th,
        "predictions": proofs,
    })
    print("All 30 forecasts complete, unscored.")


def score():
    lh, th = training_gate()
    pp = OUT / "prediction_completion.json"
    c = json.loads(pp.read_text(encoding="utf-8"))
    if c["status"] != "ALL_30_FIXED_FAMILY_TEST_FORECASTS_COMPLETE_UNSCORED" or c["jobs"] != 30 or c["lock_sha256"] != lh or c["training_completion_sha256"] != th:
        raise RuntimeError("Prediction completion gate failed")
    if set(c["predictions"]) != {job_name(*j) for j in jobs()}:
        raise RuntimeError("Incomplete prediction family")
    for b in c["predictions"].values():
        if sha(Path(b["path"])) != b["sha256"]:
            raise RuntimeError("Changed forecast")
    if (OUT / "scoring_manifest.json").exists():
        raise FileExistsError("Preserve completed scores")
    latent = np.load(OUT / "latent_test.npy", allow_pickle=False)
    orows, checks = oracle_rows()
    lookup = {(r["alpha"], r["beta"], r["adapter"]): r["bayes_pair_mse"]
              for r in orows if r["sigma_a"] == 1.0}
    scores = []
    for alpha, beta, adapter, seed in jobs():
        label = job_name(alpha, beta, adapter, seed)
        _, y = pair_data(latent, alpha, beta)
        with np.load(c["predictions"][label]["path"], allow_pickle=False) as predfile:
            pred = predfile["predictions"].astype(np.float64)
            assert np.array_equal(predfile["row_indices"], np.arange(len(y)))
        e = pred - y
        em = e.mean(1)
        ed = (e[:, 1] - e[:, 0]) / 2.0
        pe = (e * e).mean(1)
        identity_error = float(np.max(np.abs(pe - em**2 - ed**2)))
        if identity_error > 2e-12:
            raise AssertionError("Pair MSE identity failed")
        r = json.loads((OUT / "models" / f"{label}.json").read_text(encoding="utf-8"))
        scores.append({
            "alpha": alpha, "beta": beta, "adapter": adapter, "seed": seed,
            "test_pairs": len(y), "parameter_count": r["parameter_count"],
            "updates_run": r["updates_run"], "selected_update": r["selected_update"],
            "validation_pair_mse": r["best_validation_pair_mse"],
            "test_pair_mse": float(pe.mean()),
            "test_endpoint_mae": float(np.abs(e).mean()),
            "test_mean_mse": float(np.mean(em**2)),
            "test_detail_mse": float(np.mean(ed**2)),
            "test_pair_mse_iid_mc_standard_error": float(pe.std(ddof=1) / math.sqrt(len(pe))),
            "population_bayes_pair_mse": lookup[alpha, beta, adapter],
            "measured_minus_population_bayes_pair_mse": float(pe.mean()) - lookup[alpha, beta, adapter],
            "pair_identity_maximum_error": identity_error,
            "fit_wall_seconds": r["fit_wall_seconds"],
            "fit_cpu_seconds": r["fit_cpu_seconds"],
            "forecast_wall_seconds": c["predictions"][label]["forecast_wall_seconds"],
            "bayes_attainment_claimed": False,
        })
    summaries = []
    for alpha, beta in SCENARIOS:
        for adapter in ADAPTERS:
            group = [r for r in scores if (r["alpha"], r["beta"], r["adapter"]) == (alpha, beta, adapter)]
            summaries.append({
                "alpha": alpha, "beta": beta, "adapter": adapter, "fits": 3,
                "test_pair_mse_mean": float(np.mean([r["test_pair_mse"] for r in group])),
                "test_pair_mse_seed_min": min(r["test_pair_mse"] for r in group),
                "test_pair_mse_seed_max": max(r["test_pair_mse"] for r in group),
                "test_endpoint_mae_mean": float(np.mean([r["test_endpoint_mae"] for r in group])),
                "test_mean_mse_mean": float(np.mean([r["test_mean_mse"] for r in group])),
                "test_detail_mse_mean": float(np.mean([r["test_detail_mse"] for r in group])),
                "population_bayes_pair_mse": lookup[alpha, beta, adapter],
                "seed_range_is_not_confidence_interval": True,
            })
    contrasts = []
    for alpha, beta in SCENARIOS:
        for seed in SEEDS:
            g = {r["adapter"]: r for r in scores if (r["alpha"], r["beta"], r["seed"]) == (alpha, beta, seed)}
            contrasts.append({
                "alpha": alpha, "beta": beta, "seed": seed,
                "max_minus_avg_pair_mse": g["max_repeat"]["test_pair_mse"] - g["avg_repeat"]["test_pair_mse"],
                "max_minus_avg_endpoint_mae": g["max_repeat"]["test_endpoint_mae"] - g["avg_repeat"]["test_endpoint_mae"],
                "mean_abs_minus_signed_pair_mse": g["mean_abs"]["test_pair_mse"] - g["mean_signed"]["test_pair_mse"],
                "direct_minus_signed_pair_mse": g["direct"]["test_pair_mse"] - g["mean_signed"]["test_pair_mse"],
                "oracle_max_minus_avg_pair_mse": lookup[alpha, beta, "max_repeat"] - lookup[alpha, beta, "avg_repeat"],
            })
    write_csv(OUT / "oracle_grid.csv", orows)
    write_json(OUT / "oracle_quadrature_checks.json", {"status": "PASS", "checks": checks})
    write_csv(OUT / "metrics_by_seed.csv", scores)
    write_csv(OUT / "metrics_summary.csv", summaries)
    write_csv(OUT / "contrasts_by_seed.csv", contrasts)
    outputs = {n: sha(OUT / n) for n in (
        "oracle_grid.csv", "oracle_quadrature_checks.json", "metrics_by_seed.csv",
        "metrics_summary.csv", "contrasts_by_seed.csv")}
    write_json(OUT / "scoring_manifest.json", {
        "status": "PASS_COMPLETE_FIXED_SYNTHETIC_FAMILY",
        "lock_sha256": lh, "training_completion_sha256": th,
        "prediction_completion_sha256": sha(pp), "output_sha256": outputs,
        "fit_count": 30, "test_pairs_per_fit": 100000,
        "unfavorable_cells_retained": True,
        "population_oracle_not_claimed_attained_by_neural_training": True,
        "uncertainty": "Three training seeds; iid Monte Carlo SE of each fixed synthetic test score; no empirical-household interpretation",
    })
    print(json.dumps({"status": "PASS", "fit_count": 30, "outputs": outputs}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("preflight", "prepare", "seal", "train", "forecast", "score"))
    p.add_argument("--protocol-path", type=Path)
    a = p.parse_args()
    if a.command == "preflight":
        preflight()
    elif a.command == "prepare":
        prepare()
    elif a.command == "seal":
        if a.protocol_path is None:
            p.error("seal requires --protocol-path")
        seal(a.protocol_path)
    elif a.command == "train":
        train()
    elif a.command == "forecast":
        forecast()
    else:
        score()


if __name__ == "__main__":
    main()
