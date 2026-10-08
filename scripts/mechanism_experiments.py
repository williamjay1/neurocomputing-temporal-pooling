"""Prospectively frozen neural checks of two synthetic pooling mechanisms.

Synthetic data are mathematical constructions, not observed electricity loads.
The finite-support experiment evaluates approximation of exact population risks;
its checkpoint criterion is the complete known population, not a held-out split.
All fits are fixed before either independent test draw is loaded for scoring.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import log_ndtr, ndtr, roots_hermitenorm, roots_legendre

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "neural"
PROTOCOL = ROOT / "docs" / "experiment_protocol.json"
ACCESS = ("direct", "max_repeat", "avg_repeat", "mean_abs", "mean_signed")
SEEDS = tuple(range(2026100801, 2026100806))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Preserve prior artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normal_pdf(x):
    return np.exp(-0.5 * np.asarray(x)**2) / math.sqrt(2*math.pi)


def posterior_h(z):
    z = np.asarray(z, dtype=np.float64)
    t = z / math.sqrt(2)
    mills = np.exp(-0.5*t*t - 0.5*math.log(2*math.pi) - log_ndtr(t))
    return z/2 + mills/math.sqrt(2)


def posterior_var(z):
    t = np.asarray(z, dtype=np.float64)/math.sqrt(2)
    mills = np.exp(-0.5*t*t - 0.5*math.log(2*math.pi) - log_ndtr(t))
    result = (1-mills*(mills+t))/2
    if np.min(result) < -1e-9 or np.any(~np.isfinite(result)):
        raise ArithmeticError("Conditional variance integration failed")
    return np.maximum(result, 0)


def gaussian_oracle_constants():
    hn, hw = roots_hermitenorm(192)
    ln, lw = roots_legendre(192)
    a = 6*(ln+1)
    aw = 12*lw*normal_pdf(a)
    cw = hw/math.sqrt(2*math.pi)
    w_product = float(aw @ posterior_var(a[:, None]+hn[None, :]) @ cw)
    def integrand(z):
        return 2*float(normal_pdf(z/math.sqrt(2)))/math.sqrt(2)*float(ndtr(z/math.sqrt(2)))*float(posterior_var(z))
    w_density, tolerance = quad(integrand, -12*math.sqrt(2), 12*math.sqrt(2), epsabs=2e-11, epsrel=2e-11, limit=300)
    if abs(w_product-w_density) > 2e-8:
        raise AssertionError("Independent W integrals disagree")
    v = 1-2/math.pi
    switch = math.sqrt(w_product)/(math.sqrt(w_product)+math.sqrt(v))
    return dict(V=v, W=w_product, W_density=w_density,
                integral_discrepancy=abs(w_product-w_density), quadrature_error=tolerance,
                alpha_switch=switch)


def theory_module():
    path = ROOT/"scripts"/"loss_region_oracle.py"
    spec = importlib.util.spec_from_file_location("independent_loss_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def latent_draws(n, seed):
    result = np.random.default_rng(seed).standard_normal((n, 4))
    result[:, 2:] *= 0.1
    return result


def gaussian_pair(latents, alpha, beta=0.5):
    c, a, ec, ea = latents.T
    mean = c + alpha*(np.abs(a)-math.sqrt(2/math.pi)) + ec
    detail = beta*a + ea
    return np.column_stack((c-a,c+a)), np.column_stack((mean-detail,mean+detail))


def adapter(x, name):
    if name == "direct":
        return x
    if name == "max_repeat":
        return np.repeat(x.max(1,keepdims=True),2,axis=1)
    if name == "avg_repeat":
        return np.repeat(x.mean(1,keepdims=True),2,axis=1)
    c = (x[:,0]+x[:,1])/math.sqrt(2)
    a = (x[:,1]-x[:,0])/math.sqrt(2)
    return np.column_stack((c,np.abs(a) if name == "mean_abs" else a))


def gaussian_conditional_mean(latents, alpha, access):
    c, a = latents[:,:2].T
    h = np.abs(a)
    m = c+alpha*(h-math.sqrt(2/math.pi))
    b = 0.5*a
    if access == "avg_repeat":
        m,b = c,np.zeros_like(c)
    elif access == "max_repeat":
        z = c+h
        m,b = z+(alpha-1)*posterior_h(z)-alpha*math.sqrt(2/math.pi),np.zeros_like(c)
    elif access == "mean_abs":
        b = np.zeros_like(c)
    return np.column_stack((m-b,m+b))


def gaussian_floor(alpha, access, constants):
    if access in ("direct","mean_signed"):
        return 0.02
    if access == "mean_abs":
        return 0.27
    if access == "avg_repeat":
        return 0.27+alpha*alpha*constants["V"]
    return 0.27+(alpha-1)**2*constants["W"]


def finite_data(alpha):
    module = theory_module()
    rr = module.support(Fraction(str(alpha)),Fraction(1,10))
    x,y,weights = [],[],[]
    xs = {a:[] for a in ACCESS}
    for row in rr:
        c,a = row["C"],row["A"]
        x.append([float(c-a),float(c+a)])
        y.append([float(row["Y1"]),float(row["Y2"])])
        weights.append(float(row["p"]))
        xs["direct"].append(x[-1])
        xs["max_repeat"].append([float(c+abs(a))]*2)
        xs["avg_repeat"].append([float(c)]*2)
        xs["mean_abs"].append([float(c)*math.sqrt(2),float(abs(a))*math.sqrt(2)])
        xs["mean_signed"].append([float(c)*math.sqrt(2),float(a)*math.sqrt(2)])
    # Exact fractions define the observation classes; float artifacts cannot split ties.
    mapping = {"direct":"raw","mean_signed":"raw","mean_abs":"magnitude","avg_repeat":"C","max_repeat":"U"}
    floors = {}
    for access in ACCESS:
        floors[access] = {}
        for loss in ("mse","mae"):
            risk = module.endpoint_oracle(rr,mapping[access],loss)
            floors[access][loss] = float(risk)
    formula = module.formulas(Fraction(str(alpha)),Fraction(1,10))
    assert floors["avg_repeat"]["mse"] == float(formula["C_mse"])
    assert floors["max_repeat"]["mse"] == float(formula["U_mse"])
    assert floors["avg_repeat"]["mae"] == float(formula["C_mae"])
    assert floors["max_repeat"]["mae"] == float(formula["U_mae"])
    return {k:np.asarray(v,dtype=np.float64) for k,v in xs.items()}, np.asarray(y), np.asarray(weights), floors


def setup_torch(seed):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return torch


def model(seed, access):
    torch = setup_torch(seed)
    from torch import nn
    net = nn.Sequential(nn.Linear(2,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,2))
    if access in ("mean_abs","mean_signed"):
        q = torch.tensor([[1.,1.],[-1.,1.]])/math.sqrt(2)
        with torch.no_grad():
            net[0].weight.copy_(net[0].weight @ q.T)
    assert sum(p.numel() for p in net.parameters()) == 4482
    return net


def predict(net, x):
    import torch
    net.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0,len(x),8192):
            pieces.append(net(torch.as_tensor(x[start:start+8192],dtype=torch.float32)).numpy())
    return np.concatenate(pieces).astype(np.float64)


def prepare():
    assert ROOT.drive.upper() == "D:", "Compute and output must stay on D:"
    if PROTOCOL.exists() or (OUT/"experiment_lock.json").exists():
        raise FileExistsError("Existing frozen protocol cannot be replaced")
    OUT.mkdir(parents=True,exist_ok=True)
    constants = gaussian_oracle_constants()
    switch = constants["alpha_switch"]
    protocol = dict(
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        stage="post-review constructive mechanism extension; prospective only relative to these new neural outcomes",
        study_type="synthetic mathematical mechanism; no physical measurements",
        adapters=list(ACCESS), model_seeds=list(SEEDS), architecture=[2,64,64,2], activation="ReLU",
        parameter_count=4482, optimizer="Adam", learning_rate=0.001, updates=2000,
        batch_size_gaussian=256, evaluation_every_updates=200, normalization="none",
        initialization="Same base weights for every access/loss/alpha within seed; signed and magnitude first layer compensated by orthogonal rotation",
        checkpoint_gaussian="minimum validation endpoint-average MSE; earliest exact tie",
        checkpoint_finite="minimum exact weighted population training objective for its separately optimized loss; earliest exact tie",
        finite_scope="Known 8-state distribution and complete weighted support: approximation experiment, not generalization evaluation",
        test_read_gate="All 325 complete checkpoint reports must be frozen before either independent test draw is loaded for scoring",
        gaussian=dict(alpha_grid=[0.35,switch-0.02,switch-0.005,switch,switch+0.005,switch+0.02,0.56], beta=0.5,
            latent_sd_C=1,latent_sd_A=1,coordinate_noise_sd=0.1,
            sample_counts=dict(train=20000,validation=10000,test=100000),
            latent_seeds=dict(train=2026100811,validation=2026100812,test=2026100813),
            minibatch_seed_rule="default_rng(model_seed+132000), reset in each fit",
            shared_latents_all_conditions=True, optimization_loss="mse", formal_fits=175,
            reported_scores=["test endpoint MSE","analytic Bayes MSE","nonnegative conditional-mean projection regret","test MSE minus analytic floor"],
            oracle=constants),
        finite=dict(alpha_grid=[0.43,0.45,0.47],p=0.1,beta=0.1,formal_fits=150,
            optimized_losses=["mse","mae"],full_weighted_support_states=8,
            independent_test_draw_n=100000,independent_test_seed=2026100814,
            training_rule="Full support weighted population loss; no resampled minibatches and no invented data split",
            reported_scores=["exact population risk under the optimized loss","loss-specific exact Bayes floor","exact approximation gap","independent-draw risk"]),
        paired_uncertainty=dict(gaussian="2000 iid paired-row bootstrap replicates of across-seed mean max-minus-average MSE; separate seed range",
            finite="Exact population contrast first; 2000 multinomial paired-row bootstrap replicates on independent draw; separate seed range",
            replicate_count=2000,bootstrap_seed=2026100821, confidence_level=0.95,
            scope="Pointwise synthetic sampling uncertainty conditional on these five trained seeds; not a confidence interval for optimizer-seed population"),
        test_outcomes_used_for_design=False, permitted_adjustments="Technical preflight bugs only, before formal fits, with explicit record",
        no_neural_bayes_attainment_assumption=True)
    torch = setup_torch(SEEDS[0])
    check = latent_draws(2048,2026100891)
    x,y = gaussian_pair(check,switch)
    compensation=[]
    finite_x,_,_,_=finite_data(0.45)
    for seed in SEEDS:
        direct,signed = model(seed,"direct"),model(seed,"mean_signed")
        err = float(np.max(np.abs(predict(direct,x)-predict(signed,adapter(x,"mean_signed")))))
        finite_err=float(np.max(np.abs(predict(direct,finite_x["direct"])-predict(signed,finite_x["mean_signed"]))))
        if max(err,finite_err)>2e-6:
            raise AssertionError("Initial direct/signed function compensation failed")
        compensation.append(dict(seed=seed,gaussian_max_abs_difference=err,finite_max_abs_difference=finite_err))
    err=max(row["gaussian_max_abs_difference"] for row in compensation)
    direct=model(SEEDS[0],"direct")
    started = time.perf_counter()
    optimizer = torch.optim.Adam(direct.parameters(),lr=0.001)
    xt,yt = torch.tensor(x,dtype=torch.float32),torch.tensor(y,dtype=torch.float32)
    rng = np.random.default_rng(2026100892)
    for _ in range(100):
        ix = rng.integers(0,len(x),size=256)
        optimizer.zero_grad()
        loss = (direct(xt[ix])-yt[ix]).square().mean()
        loss.backward(); optimizer.step()
    for alpha in protocol["finite"]["alpha_grid"]:
        finite_data(alpha)
    write_json(OUT/"preflight.json",dict(status="PASS_SOURCE_ONLY_PREFLIGHT",test_values_read=False,
        formal_fits=0,initial_direct_signed_max_abs_difference=err,hundred_update_wall_seconds=time.perf_counter()-started,
        oracle=constants,python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        technical_adjustments=[],all_parameter_counts=4482,all_seed_initial_compensation=compensation))
    write_json(PROTOCOL,protocol)
    bindings = []
    for split,n in protocol["gaussian"]["sample_counts"].items():
        path = OUT/f"gaussian_{split}_latents.npy"
        if path.exists(): raise FileExistsError(path)
        np.save(path,latent_draws(n,protocol["gaussian"]["latent_seeds"][split]),allow_pickle=False)
        bindings.append(dict(path=str(path),sha256=sha(path)))
    finite_path = OUT/"finite_test_state_indices.npy"
    if finite_path.exists(): raise FileExistsError(finite_path)
    _,_,weights,_ = finite_data(0.45)
    indices = np.random.default_rng(2026100814).choice(8,size=100000,p=weights)
    np.save(finite_path,indices,allow_pickle=False)
    bindings.append(dict(path=str(finite_path),sha256=sha(finite_path)))
    for path in (Path(__file__),PROTOCOL,ROOT/"scripts"/"loss_region_oracle.py",OUT/"preflight.json"):
        bindings.append(dict(path=str(path.resolve()),sha256=sha(path)))
    write_json(OUT/"experiment_lock.json",dict(status="FROZEN_BEFORE_FORMAL_TRAINING_AND_TEST_SCORING",files=bindings,formal_fits=325))
    print(json.dumps(dict(status="PREPARED_AND_FROZEN",alpha_switch=switch,formal_fits=325,initial_compensation_error=err)),flush=True)


def locked():
    lock = json.loads((OUT/"experiment_lock.json").read_text(encoding="utf-8"))
    for binding in lock["files"]:
        if sha(binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"Changed frozen input: {binding['path']}")
    return json.loads(PROTOCOL.read_text(encoding="utf-8")),sha(OUT/"experiment_lock.json")


def jobs(protocol):
    for group in ("gaussian","finite"):
        for ai,alpha in enumerate(protocol[group]["alpha_grid"]):
            for access in ACCESS:
                for loss in (["mse"] if group=="gaussian" else ["mse","mae"]):
                    for seed in SEEDS:
                        yield dict(group=group,alpha_index=ai,alpha=alpha,access=access,loss=loss,seed=seed,
                            label=f"{group}_a{ai}_{access}_{loss}_s{seed}")


def state_hash(state):
    h = hashlib.sha256()
    for key,value in state.items():
        h.update(key.encode()); h.update(value.tobytes())
    return h.hexdigest()


def train():
    protocol,lock_sha = locked()
    if (OUT/"training_completion.json").exists():
        raise FileExistsError("Complete formal training cannot be silently rerun")
    # Test latent/indices are not loaded here. Their SHA bindings check integrity only.
    lt = np.load(OUT/"gaussian_train_latents.npy",allow_pickle=False)
    lv = np.load(OUT/"gaussian_validation_latents.npy",allow_pickle=False)
    model_dir = OUT/"models"; model_dir.mkdir(exist_ok=True)
    reports=[]
    begin=time.perf_counter()
    for job in jobs(protocol):
        label=job["label"]
        rp,cp,tp = (model_dir/f"{label}.{suffix}" for suffix in ("json","npz","csv"))
        if rp.exists():
            prior=json.loads(rp.read_text(encoding="utf-8"))
            if prior["lock_sha256"]!=lock_sha or sha(cp)!=prior["checkpoint_sha256"] or sha(tp)!=prior["trace_sha256"]:
                raise RuntimeError("Prior checkpoint provenance mismatch")
            reports.append(dict(path=str(rp),sha256=sha(rp))); continue
        if cp.exists() or tp.exists(): raise RuntimeError(f"Preserve incomplete fit: {label}")
        torch=setup_torch(job["seed"])
        net=model(job["seed"],job["access"])
        initial={k:v.detach().numpy().copy() for k,v in net.state_dict().items()}
        if job["group"]=="gaussian":
            raw,y=gaussian_pair(lt,job["alpha"])
            rawv,yv=gaussian_pair(lv,job["alpha"])
            x=adapter(raw,job["access"]); xv=adapter(rawv,job["access"])
            weights=None
        else:
            xs,y,weights,_=finite_data(job["alpha"])
            x=xs[job["access"]]; xv=x; yv=y
        xt,yt=torch.tensor(x,dtype=torch.float32),torch.tensor(y,dtype=torch.float32)
        wt=None if weights is None else torch.tensor(weights[:,None],dtype=torch.float32)
        rng=np.random.default_rng(job["seed"]+132000)
        optimizer=torch.optim.Adam(net.parameters(),lr=0.001)
        best,best_step,best_state=math.inf,None,None
        trace=[]; start=time.perf_counter(); startcpu=time.process_time()
        for step in range(1,2001):
            net.train(); optimizer.zero_grad()
            if weights is None:
                ix=rng.integers(0,len(x),size=256)
                error=net(xt[ix])-yt[ix]
                objective=error.square().mean()
            else:
                error=net(xt)-yt
                per=error.square() if job["loss"]=="mse" else error.abs()
                objective=(wt*per).sum()/2
            objective.backward(); optimizer.step()
            if step%200==0:
                prediction=predict(net,xv)
                per=(prediction-yv)**2 if job["loss"]=="mse" else np.abs(prediction-yv)
                score=float(np.mean(per)) if weights is None else float(weights @ per.mean(1))
                trace.append(dict(update=step,training_objective=float(objective.detach()),selection_objective=score))
                if score<best:
                    best,best_step=score,step
                    best_state={k:v.detach().numpy().copy() for k,v in net.state_dict().items()}
        np.savez(cp,**best_state); write_csv(tp,trace)
        report=dict(**job,status="COMPLETE_CHECKPOINT_FIXED_TEST_UNREAD",lock_sha256=lock_sha,
            parameter_count=4482,updates_run=2000,selected_update=best_step,selection_objective=best,
            initial_state_sha256=state_hash(initial),checkpoint_sha256=sha(cp),trace_sha256=sha(tp),
            fit_wall_seconds=time.perf_counter()-start,fit_cpu_seconds=time.process_time()-startcpu,
            independent_test_values_read=False)
        write_json(rp,report); reports.append(dict(path=str(rp),sha256=sha(rp)))
        if len(reports)%5==0:
            print(json.dumps(dict(completed=len(reports),total=325,last_job=label,selection_objective=best,elapsed_seconds=time.perf_counter()-begin)),flush=True)
    assert len(reports)==325
    # Matching basic weights for the two objectives is checked from actual reports.
    hashes={}
    for binding in reports:
        value=json.loads(Path(binding["path"]).read_text())
        key=(value["alpha_index"],value["access"],value["seed"])
        if value["group"]=="finite":
            if key in hashes: assert hashes[key]==value["initial_state_sha256"]
            hashes[key]=value["initial_state_sha256"]
    write_json(OUT/"training_completion.json",dict(status="ALL_325_CHECKPOINTS_COMPLETE_BEFORE_TEST_READ",lock_sha256=lock_sha,
        completed_fits=325,reports=reports,independent_test_values_read=False,wall_seconds=time.perf_counter()-begin))
    print("ALL_325_FORMAL_FITS_COMPLETE_TEST_UNREAD",flush=True)


def bootstrap_mean_interval(values,rng,reps=2000):
    values=np.asarray(values,dtype=np.float64)
    draws=[]
    for start in range(0,reps,20):
        ix=rng.integers(0,len(values),size=(min(20,reps-start),len(values)))
        draws.extend(values[ix].mean(1))
    return [float(x) for x in np.quantile(draws,[0.025,0.975])]


def score():
    protocol,lock_sha=locked()
    if (OUT/"evaluation_completion.json").exists(): raise FileExistsError("Preserve complete test evaluation")
    completion=json.loads((OUT/"training_completion.json").read_text())
    assert completion["completed_fits"]==325 and completion["lock_sha256"]==lock_sha
    for binding in completion["reports"]:
        assert sha(binding["path"])==binding["sha256"]
        report=json.loads(Path(binding["path"]).read_text())
        checkpoint=OUT/"models"/f"{report['label']}.npz"
        assert sha(checkpoint)==report["checkpoint_sha256"]
    # First evaluation read follows the complete checkpoint gate.
    test=np.load(OUT/"gaussian_test_latents.npy",allow_pickle=False)
    indices=np.load(OUT/"finite_test_state_indices.npy",allow_pickle=False)
    rowdir=OUT/"row_losses"; rowdir.mkdir(exist_ok=True)
    scores=[]; rowstore={}; finite_per_state={}; start=time.perf_counter()
    for job in jobs(protocol):
        net=model(job["seed"],job["access"])
        torch=setup_torch(job["seed"])
        checkpoint=OUT/"models"/f"{job['label']}.npz"
        with np.load(checkpoint,allow_pickle=False) as state:
            net.load_state_dict({k:torch.tensor(state[k]) for k in state.files})
        if job["group"]=="gaussian":
            raw,y=gaussian_pair(test,job["alpha"])
            pred=predict(net,adapter(raw,job["access"]))
            conditional=gaussian_conditional_mean(test,job["alpha"],job["access"])
            per=((pred-y)**2).mean(1)
            oracle_per=((conditional-y)**2).mean(1)
            regret=((pred-conditional)**2).mean(1)
            floor=gaussian_floor(job["alpha"],job["access"],protocol["gaussian"]["oracle"])
            score=dict(**job,analytic_bayes_risk=floor,test_risk=float(per.mean()),
                exact_population_risk="",projection_regret=float(regret.mean()),
                test_minus_analytic_floor=float(per.mean()-floor),oracle_test_risk=float(oracle_per.mean()),
                population_approximation_gap="")
            arrays=dict(loss=per.astype(np.float32),conditional_mean_loss=oracle_per.astype(np.float32),projection_regret=regret.astype(np.float32))
        else:
            xs,y,weights,floors=finite_data(job["alpha"])
            pred=predict(net,xs[job["access"]])
            per=((pred-y)**2 if job["loss"]=="mse" else np.abs(pred-y)).mean(1)
            population=float(weights @ per); floor=floors[job["access"]][job["loss"]]
            if population-floor < -1e-10: raise AssertionError("Finite predictor below exact population optimum")
            score=dict(**job,analytic_bayes_risk=floor,test_risk=float(per[indices].mean()),
                exact_population_risk=population,projection_regret="",test_minus_analytic_floor=float(per[indices].mean()-floor),
                oracle_test_risk="",population_approximation_gap=population-floor)
            arrays=dict(loss=per[indices].astype(np.float32),support_loss=per,predictions=pred,support_weights=weights)
            finite_per_state[(job["alpha_index"],job["access"],job["loss"],job["seed"])]=per
        np.savez_compressed(rowdir/f"{job['label']}.npz",**arrays)
        scores.append(score)
        if job["access"] in ("max_repeat","avg_repeat"):
            rowstore[(job["group"],job["alpha_index"],job["access"],job["loss"],job["seed"])]=arrays["loss"]
    write_csv(OUT/"all_seed_scores.csv",scores)
    aggregates=[]
    for group in ("gaussian","finite"):
        for ai,alpha in enumerate(protocol[group]["alpha_grid"]):
            for access in ACCESS:
                for loss in (["mse"] if group=="gaussian" else ["mse","mae"]):
                    rows=[r for r in scores if r["group"]==group and r["alpha_index"]==ai and r["access"]==access and r["loss"]==loss]
                    risk=[r["test_risk"] if group=="gaussian" else r["exact_population_risk"] for r in rows]
                    gap=[r["projection_regret"] if group=="gaussian" else r["population_approximation_gap"] for r in rows]
                    aggregates.append(dict(group=group,alpha_index=ai,alpha=alpha,access=access,loss=loss,n_seeds=5,
                        analytic_bayes_risk=rows[0]["analytic_bayes_risk"],risk_mean=float(np.mean(risk)),risk_seed_min=min(risk),risk_seed_max=max(risk),
                        approximation_gap_mean=float(np.mean(gap)),approximation_gap_seed_min=min(gap),approximation_gap_seed_max=max(gap),
                        independent_test_risk_mean=float(np.mean([r["test_risk"] for r in rows]))))
    write_csv(OUT/"condition_summary.csv",aggregates)
    comparisons=[]; rng=np.random.default_rng(2026100821)
    count=np.bincount(indices,minlength=8)
    for group in ("gaussian","finite"):
        for ai,alpha in enumerate(protocol[group]["alpha_grid"]):
            for loss in (["mse"] if group=="gaussian" else ["mse","mae"]):
                differences=[]; seedgaps=[]; exactgaps=[]
                for seed in SEEDS:
                    k=(group,ai,"max_repeat",loss,seed); q=(group,ai,"avg_repeat",loss,seed)
                    differences.append(rowstore[k].astype(np.float64)-rowstore[q].astype(np.float64))
                    seedgaps.append(float(differences[-1].mean()))
                    if group=="finite":
                        d=finite_per_state[(ai,"max_repeat",loss,seed)]-finite_per_state[(ai,"avg_repeat",loss,seed)]
                        _,_,weights,_=finite_data(alpha); exactgaps.append(float(weights @ d))
                meanrow=np.mean(differences,axis=0)
                if group=="gaussian":
                    interval=bootstrap_mean_interval(meanrow,rng)
                    analytic=(alpha-1)**2*protocol["gaussian"]["oracle"]["W"]-alpha*alpha*protocol["gaussian"]["oracle"]["V"]
                    exact=""
                else:
                    dmean=np.mean([finite_per_state[(ai,"max_repeat",loss,s)]-finite_per_state[(ai,"avg_repeat",loss,s)] for s in SEEDS],axis=0)
                    boot=rng.multinomial(len(indices),count/len(indices),size=2000) @ dmean / len(indices)
                    interval=[float(x) for x in np.quantile(boot,[.025,.975])]
                    _,_,_,floors=finite_data(alpha)
                    analytic=floors["max_repeat"][loss]-floors["avg_repeat"][loss]
                    exact=float(np.mean(exactgaps))
                comparisons.append(dict(group=group,alpha_index=ai,alpha=alpha,loss=loss,contrast="max_minus_average",
                    analytic_bayes_contrast=analytic,exact_population_neural_contrast=exact,
                    test_across_seed_mean_contrast=float(meanrow.mean()),paired_row_bootstrap_low=interval[0],paired_row_bootstrap_high=interval[1],
                    test_seed_contrast_min=min(seedgaps),test_seed_contrast_max=max(seedgaps),
                    exact_seed_contrast_min=min(exactgaps) if exactgaps else "",exact_seed_contrast_max=max(exactgaps) if exactgaps else ""))
    write_csv(OUT/"paired_contrasts.csv",comparisons)
    bindings=[dict(path=str(OUT/name),sha256=sha(OUT/name)) for name in ("all_seed_scores.csv","condition_summary.csv","paired_contrasts.csv")]
    write_json(OUT/"evaluation_completion.json",dict(status="COMPLETE_325_FIXED_CHECKPOINT_EVALUATION",lock_sha256=lock_sha,
        training_completion_sha256=sha(OUT/"training_completion.json"),evaluated_fits=325,score_tables=bindings,
        per_row_archives=325,evaluation_wall_seconds=time.perf_counter()-start,uncertainty_scope=protocol["paired_uncertainty"]["scope"],
        generated_data_are_synthetic=True,finite_experiment_is_population_approximation=True))
    print(json.dumps(dict(status="EVALUATION_COMPLETE",evaluated_fits=325,wall_seconds=time.perf_counter()-start)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("prepare","train","score"))
    args=parser.parse_args()
    {"prepare":prepare,"train":train,"score":score}[args.stage]()
