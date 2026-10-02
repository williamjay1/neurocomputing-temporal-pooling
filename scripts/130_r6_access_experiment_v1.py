"""R6 UCI matched final-residual access experiment; new artifacts only.

Source fitting reads 2012 development customers. Calibration inference requires
an explicit root-supplied gate binding this script, protocol and source fits.
The 2014 confirmation outcomes have no evaluation entry point in this script.
"""
from __future__ import annotations
import argparse, hashlib, inspect, json, math, platform, sys, time
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path('D:/MLWork/Neurocomputing_TailTransfer')
DATA = ROOT/'datasets/upgrade5_uci_v1'
OUT = ROOT/'results/r6_access_v1'
ADAPTERS = ('direct','max_repeat','avg_repeat','haar_absolute','haar_signed','max_blur_repeat')
HISTORIES = ('fine','coarse')
QUALITY_REFERENCES = ('persistence','daily_naive','weekly_naive')
SEEDS = (20261001,20261002,20261003)
HISTORY, HORIZON, UPDATES, BATCH = 672,96,2000,64
PARAMETERS = 240036
PREDICTION_ACTIVE_PARAMETERS = PARAMETERS-(HISTORY*64+HISTORY)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''): h.update(block)
    return h.hexdigest()


def write_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def load_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_preflight_bindings():
    proof=load_json(OUT/'preflight.json')
    if proof.get('status')!='PASS': raise ValueError('engineering preflight not complete')
    for path,record in proof['bindings'].items():
        if sha(path)!=record['sha256']: raise ValueError('preflight input changed '+path)
    if sha(OUT/'proposed_protocol.json')!=proof['protocol_sha256']:
        raise ValueError('protocol changed after engineering preflight')
    return proof


def verify_source_cache():
    verify_preflight_bindings()
    folder=OUT/'source_cache';manifest=load_json(folder/'manifest.json')
    for name,record in manifest['files'].items():
        if sha(folder/name)!=record['sha256']: raise ValueError('source cache changed '+name)
    if manifest['production_script_sha256']!=sha(__file__): raise ValueError('source code changed')
    if manifest['proposed_protocol_sha256']!=sha(OUT/'proposed_protocol.json'): raise ValueError('source protocol changed')
    return manifest


def setup(seed):
    import torch
    torch.set_num_threads(2);torch.manual_seed(int(seed))
    return torch


def model_factory(adapter):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from neuralforecast.models.nhits import NHITSBlock,_IdentityBasis
    if adapter not in ADAPTERS: raise ValueError('undeclared adapter')

    class PairAdapter(nn.Module):
        def __init__(self,kind):
            super().__init__();self.kind=kind
            self.register_buffer('blur',torch.tensor([.25,.5,.25]).reshape(1,1,3),persistent=False)
        def forward(self,x):
            if self.kind=='direct': return x
            if self.kind=='max_blur_repeat':
                # Right replication makes stride-one max preserve L.
                y=F.max_pool1d(F.pad(x,(0,1),mode='replicate'),2,stride=1)
                # Replicated endpoints, centered binomial blur, stride two.
                y=F.conv1d(F.pad(y,(1,1),mode='replicate'),self.blur,stride=2)
                return y.repeat_interleave(2,dim=-1)
            p=x.reshape(*x.shape[:-1],x.shape[-1]//2,2)
            if self.kind=='max_repeat': return p.max(-1).values.repeat_interleave(2,-1)
            if self.kind=='avg_repeat': return p.mean(-1).repeat_interleave(2,-1)
            c=(p[...,0]+p[...,1])/math.sqrt(2)
            a=(p[...,0]-p[...,1])/math.sqrt(2)
            if self.kind=='haar_absolute': a=a.abs()
            return torch.stack((c,a),dim=-1).flatten(-2)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks=nn.ModuleList([NHITSBlock(input_size=HISTORY,h=HORIZON,
                n_theta=HISTORY+HORIZON//down,mlp_units=[[64,64]]*2,
                basis=_IdentityBasis(HISTORY,HORIZON,'linear',1),
                futr_input_size=0,hist_input_size=0,stat_input_size=0,
                n_pool_kernel_size=pool,pooling_mode='MaxPool1d',
                dropout_prob=0.,activation='ReLU')
                for down,pool in zip((8,4,1),(4,2,1))])
            self.blocks[-1].pooling_layer=PairAdapter(adapter)
            if adapter in ('haar_signed','haar_absolute'):
                # W_H = W_D T^T; T is pairwise orthogonal and symmetric.
                with torch.no_grad():
                    first=self.blocks[-1].layers[0]
                    w=first.weight.reshape(first.out_features,HISTORY//2,2).clone()
                    transformed=torch.stack(((w[...,0]+w[...,1])/math.sqrt(2),
                                             (w[...,0]-w[...,1])/math.sqrt(2)),dim=-1)
                    first.weight.copy_(transformed.flatten(-2))
        def forward(self,x,mask):
            count=mask.sum(1,keepdim=True).clamp(min=1)
            mean=(x*mask).sum(1,keepdim=True)/count
            var=((x-mean).square()*mask).sum(1,keepdim=True)/count
            scale=torch.maximum(var.sqrt(),torch.maximum(mean.abs()*.05,torch.ones_like(mean)))
            norm=(x-mean)/scale;residual=norm.flip(-1);coverage=mask.flip(-1)
            pred=norm[:,-1:].expand(-1,HORIZON)
            for block in self.blocks:
                back,future=block(residual,None,None,None)
                residual=(residual-back)*coverage;pred=pred+future[:,:,0]
            return pred*scale+mean
    return Model()


def causal_fill(x,coverage):
    x=np.asarray(x,np.float32).copy();coverage=np.asarray(coverage,np.float32)
    if x.ndim!=2 or x.shape!=coverage.shape: raise ValueError('batch history shape')
    last=np.zeros(len(x),np.float32)
    for k in range(x.shape[1]):
        live=coverage[:,k]>0
        last=np.where(live,x[:,k],last)
        x[:,k]=last
    return x,coverage


def history_representation(values,observed,history):
    v=np.asarray(values,np.float32);ob=np.asarray(observed,bool)
    if v.shape!=ob.shape or v.ndim!=2 or v.shape[1]!=HISTORY: raise ValueError('raw history shape')
    if not np.isfinite(v[ob]).all(): raise ValueError('nonfinite observed history')
    if history=='fine': return causal_fill(v,ob.astype(np.float32))
    if history!='coarse': raise ValueError('undeclared history')
    pv=v.reshape(len(v),HISTORY//2,2);pm=ob.reshape(len(v),HISTORY//2,2)
    counts=pm.sum(-1).astype(np.uint8)
    means=np.divide(np.where(pm,pv,0).sum(-1,dtype=np.float64),counts,
        out=np.full(counts.shape,np.nan,dtype=np.float64),where=counts>0).astype(np.float32)
    expanded=np.repeat(means,2,axis=-1);coverage=np.repeat(counts.astype(np.float32)/2,2,axis=-1)
    return causal_fill(expanded,coverage)


def source_data():
    m=load_json(DATA/'dataset_manifest_v1.json')
    labels=np.load(DATA/'provider_labels.npy',allow_pickle=False)
    ids=np.load(DATA/'client_ids.npy',allow_pickle=False).tolist()
    v=np.load(DATA/'panel_kw.npy',mmap_mode='r');valid=np.load(DATA/'valid.npy',mmap_mode='r')
    groups=m['customer_groups'];cols=[ids.index(i) for i in groups['development'][:24]]
    idx=lambda date:int(np.searchsorted(labels,np.datetime64(date)))
    lo,hi=idx('2012-01-08'),idx('2012-09-15')
    rows=[(c,o) for c in cols for o in range(lo,hi-96,48)
        if valid[o:o+96,c].mean()>=.95 and valid[o-672:o,c].mean()>=.95]
    vlo,vhi=idx('2012-09-16'),idx('2012-09-30')
    held=[(c,o) for c in cols[:8] for o in range(vlo,vhi-96,96)
        if valid[o:o+96,c].mean()>=.95]
    def collect(keys):
        x=np.stack([v[o-HISTORY:o,c] for c,o in keys]).astype(np.float32)
        xm=np.stack([valid[o-HISTORY:o,c] for c,o in keys]).astype(np.uint8)
        y=np.stack([v[o:o+HORIZON,c] for c,o in keys]).astype(np.float32)
        ym=np.stack([valid[o:o+HORIZON,c] for c,o in keys]).astype(np.uint8)
        if any(o+HORIZON>idx('2012-10-01') for _,o in keys): raise ValueError('source time boundary')
        return x,xm,np.nan_to_num(y,nan=0.),ym,np.asarray(keys,np.int64)
    return collect(rows),collect(held),groups


def protocol():
    return {
        'stage':'SOURCE_ONLY_IMPLEMENTED_CALIBRATION_INFERENCE_GATED',
        'analysis_status':'R6 development family; existing R4/R5 results known; new calibration results not used for design',
        'adapters':list(ADAPTERS),'histories':list(HISTORIES),'seeds':list(SEEDS),'formal_fit_count':36,
        'history_slots':HISTORY,'horizon_slots':HORIZON,'parameter_count':PARAMETERS,
        'prediction_active_parameter_count':PREDICTION_ACTIVE_PARAMETERS,
        'inactive_parameter_explanation':'last theta head has672 backcast rows unused by the final forecast;43680 registered coefficients; identical for all cells',
        'width':64,'blocks':3,'native_pool_constructor':[4,2,1],'forecast_downsampling':[8,4,1],
        'intervention':'replace only final pooling_layer; all learned shapes and remaining forward unchanged',
        'haar_scaling':'orthonormal (r1+r2)/sqrt2,(r1-r2)/sqrt2 in reversed residual pair order',
        'haar_initialization':'last first affine W_H=W_direct*T^T for signed and absolute Haar; signed preserves initial direct function',
        'blur_definition':'right-replicate input; max kernel2 stride1; replicate both endpoints; binomial [1,2,1]/4 convolution stride2; repeat2',
        'blur_is_standard_style_reference_not_author_benchmark_reproduction':True,
        'normalization':'own decoded past observed/coverage weighted mean and population variance; scale floor max(1 kW,.05*abs(mean)); no fine-detail normalization for coarse',
        'fine_history':'672 float32 values with native observed-position availability',
        'coarse_history':'336 observed-only float32 pair means plus uint8 constituent counts, repeated to672 with count/2 coverage',
        'source_customers':'first24 fixed development IDs; held-time validation first8 of these same IDs',
        'source_train_origins':'range(index2012-01-08,index2012-09-15-96,48); exact row keys retained',
        'source_validation_origins':'range(index2012-09-16,index2012-09-30-96,96); exact row keys retained',
        'updates':UPDATES,'batch_size':BATCH,'optimizer':'Adam','lr':.001,'gradient_clip':1.,
        'scheduler':'none, as R5','threads_per_source_process':2,
        'fitting_loss':'per-row complete-pair mean endpoint MAE divided by own decoded observed-window mean floored1kW; mean over batch',
        'checkpoint_rule':'lowest complete-pair history-normalized source-held-time MAE at updates200,400,...,2000',
        'shared_draws':'one stored 2000x64 index array per seed, same rows across all12 adapter/history cells',
        'history_cache':'fine and coarse representations precomputed once from source-only history windows; fitting reads only these D arrays',
        'primary_evaluation':'customer-macro complete-pair endpoint MAE divided by fixed pre-calibration support mean floored1kW',
        'secondary_evaluation':['physical customer-macro endpoint MAE kW','endpoint RMSE','pair mean/detail MSE'],
        'calibration_customers':'all16 fixed calibration IDs, columns32:48; no outcome-dependent removal',
        'calibration_forecast_dates':'2013-07-01 through2013-12-31 daily;184 origins',
        'evaluation_support':'2013-06-24 through2013-06-30; observed mean floored1kW, frozen per customer',
        'quality_references':{
            'scope':'three no-fit fine-history quality references, not architecture or resource comparisons',
            'input':'same causal forward-filled fine672 history with leading zero if no observed predecessor',
            'persistence':'repeat last available causal-filled value for96 future slots',
            'daily_naive':'repeat the final96 lag slots as the next96 predictions',
            'weekly_naive':'reuse the first96 of the672 lag slots (origin-672 through origin-577)',
            'mask_and_scales':'same original complete-future-pair mask and fixed pre-calibration support scales as neural forecasts',
            'selection':'all three retained; no best-reference selection'},
        'common_target_mask':'original observed future pairs with both endpoints present; identical across36 fits',
        'prediction_clipping':'none, as R5 Portuguese predictor',
        'reserved_confirmation':'columns48:80 /2014; no evaluation code path',
        'retention_gate_or_deleted_state':'none; no R5 winner or ridge-gate reuse',
        'root_gate_required_before_any_calibration_inference':True,
        'no_function_class_equivalence_claim_for_noninvertible_adapters':True,
        'prediction_schema':{'customer_ids':'[16] string','forecast_dates':'[184] datetime64[D]',
            'origin_indices':'[184] int64','predictions_kw':'[16,184,96] float32'},
        'source_residual_diagnostics':'not included in primary runs; any new diagnostic requires explicit addendum'
    }


def preflight():
    OUT.mkdir(parents=True,exist_ok=True)
    torch=setup(SEEDS[0]);rng=np.random.default_rng(7301)
    raw=rng.normal(100,20,(8,HISTORY)).astype(np.float32);ob=np.ones(raw.shape,np.uint8)
    ob[0,:9]=0;ob[1,100]=0;raw[ob==0]=np.nan
    xf,mf=history_representation(raw,ob,'fine');xc,mc=history_representation(raw,ob,'coarse')
    poisoned=raw.copy();poisoned[ob==0]=1e20
    for hist in HISTORIES:
        clean=history_representation(raw,ob,hist);bad=history_representation(poisoned,ob,hist)
        assert all(np.array_equal(a,b) for a,b in zip(clean,bad))
    assert mc[1,100]==mc[1,101]==.5
    pair_means=np.repeat(rng.normal(100,10,(4,HISTORY//2)),2,-1).astype(np.float32)
    alt=pair_means.copy();alt[:,0::2]+=3;alt[:,1::2]-=3
    assert all(np.array_equal(a,b) for a,b in zip(history_representation(pair_means,np.ones_like(pair_means),'coarse'),
                                                history_representation(alt,np.ones_like(alt),'coarse')))
    checks={};reference=None;reference_shapes=None;direct_hash=None;signed_max_difference=None
    for adapter in ADAPTERS:
        setup(SEEDS[0]);net=model_factory(adapter)
        assert sum(p.numel() for p in net.parameters())==PARAMETERS
        shapes={k:list(v.shape) for k,v in net.named_parameters()}
        if reference_shapes is None: reference_shapes=shapes
        assert shapes==reference_shapes
        x=torch.from_numpy(xf);mask=torch.from_numpy(mf)
        out=net(x,mask);assert out.shape==(8,HORIZON) and torch.isfinite(out).all()
        out.mean().backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
        if adapter=='direct': reference=out.detach().clone()
        if adapter=='haar_signed':
            torch.testing.assert_close(out.detach(),reference,rtol=3e-5,atol=5e-5)
            signed_max_difference=float((out.detach()-reference).abs().max())
        initial_hash=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in net.parameters())).hexdigest()
        if adapter=='direct': direct_hash=initial_hash
        if adapter not in ('haar_signed','haar_absolute'): assert initial_hash==direct_hash
        checks[adapter]={'parameters':PARAMETERS,'prediction_active_parameters':PREDICTION_ACTIVE_PARAMETERS,'shape_and_finite_gradients':True,
            'initial_parameter_sha256':initial_hash}
    from neuralforecast.models.nhits import NHITSBlock
    bindings={str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in
        [Path(__file__),DATA/'dataset_manifest_v1.json',DATA/'panel_kw.npy',DATA/'valid.npy',
         DATA/'provider_labels.npy',DATA/'client_ids.npy',Path(inspect.getfile(NHITSBlock))]}
    write_json(OUT/'proposed_protocol.json',protocol())
    write_json(OUT/'preflight.json',{'status':'PASS','synthetic_only':True,'no_calibration_or_confirmation_values_read':True,
        'adapter_checks':checks,'fine_coarse_missing_poison_pass':True,'coarse_counts_coverage_pass':True,
        'coarse_does_not_use_original_detail_pass':True,'signed_initial_function_alignment_pass':True,
        'signed_initial_function_max_absolute_difference_kw':signed_max_difference,
        'nonhaar_all_initial_learned_tensor_bytes_identical':True,
        'bindings':bindings,'protocol_sha256':sha(OUT/'proposed_protocol.json'),
        'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version()})
    print(json.dumps({'preflight':'PASS','script_sha256':sha(__file__),'protocol_sha256':sha(OUT/'proposed_protocol.json')}),flush=True)


def prepare_sources():
    verify_preflight_bindings();wall=time.perf_counter();cpu=time.process_time()
    if (OUT/'source_cache/manifest.json').exists(): raise FileExistsError('source cache already sealed')
    train,val,groups=source_data();folder=OUT/'source_cache';folder.mkdir(exist_ok=True)
    records={}
    for label,data in [('train',train),('validation',val)]:
        for key,a in zip(('values','observed','labels','label_observed','row_keys'),data):
            p=folder/f'{label}_{key}.npy';np.save(p,a);records[p.name]={'sha256':sha(p),'bytes':p.stat().st_size,'shape':list(a.shape)}
        for history in HISTORIES:
            decoded,coverage=history_representation(data[0],data[1],history)
            for key,a in (('decoded',decoded),('coverage',coverage)):
                p=folder/f'{label}_{history}_{key}.npy';np.save(p,a)
                records[p.name]={'sha256':sha(p),'bytes':p.stat().st_size,'shape':list(a.shape)}
    for seed in SEEDS:
        draws=np.random.default_rng(seed).integers(len(train[0]),size=(UPDATES,BATCH),dtype=np.int32)
        p=folder/f'draws_seed{seed}.npy';np.save(p,draws);records[p.name]={'sha256':sha(p),'bytes':p.stat().st_size,'shape':list(draws.shape)}
    write_json(folder/'manifest.json',{'status':'SOURCE_CACHE_COMPLETE','train_rows':len(train[0]),'validation_rows':len(val[0]),
        'training_customers':groups['development'][:24],'validation_customers':groups['development'][:8],
        'data_manifest_sha256':sha(DATA/'dataset_manifest_v1.json'),'production_script_sha256':sha(__file__),
        'proposed_protocol_sha256':sha(OUT/'proposed_protocol.json'),'files':records,
        'all_numeric_samples_strictly_pre2012_october':True,
        'construction_wall_seconds':time.perf_counter()-wall,'construction_process_cpu_seconds':time.process_time()-cpu})
    print(json.dumps({'source_cache':'COMPLETE','train_rows':len(train[0]),'validation_rows':len(val[0])}),flush=True)


def complete_pair_loss(pred,target,observed,x,coverage):
    import torch
    live=observed.reshape(-1,HORIZON//2,2).all(-1)
    errors=(pred-target).abs().reshape(-1,HORIZON//2,2).mean(-1)
    row=(errors*live).sum(1)/live.sum(1).clamp(min=1)
    scale=torch.maximum((x*coverage).sum(1)/coverage.sum(1).clamp(min=1),torch.ones(len(x)))
    return (row/scale).mean()


def fit_one(adapter,history,seed):
    torch=setup(seed);cache=OUT/'source_cache';manifest=load_json(cache/'manifest.json')
    if manifest['production_script_sha256']!=sha(__file__): raise ValueError('source plan code mismatch')
    tag=f'{adapter}_{history}_seed{seed}';folder=OUT/'source_fits';folder.mkdir(exist_ok=True)
    report=folder/f'{tag}.json';checkpoint=folder/f'{tag}.npz'
    if report.exists() or checkpoint.exists(): raise FileExistsError(tag)
    arrays={k:np.load(cache/f'train_{k}.npy',mmap_mode='r') for k in ('labels','label_observed')}
    arrays['decoded']=np.load(cache/f'train_{history}_decoded.npy',mmap_mode='r')
    arrays['coverage']=np.load(cache/f'train_{history}_coverage.npy',mmap_mode='r')
    vx=np.load(cache/f'validation_{history}_decoded.npy');vm=np.load(cache/f'validation_{history}_coverage.npy')
    vy=np.load(cache/'validation_labels.npy');vym=np.load(cache/'validation_label_observed.npy').astype(bool)
    draws=np.load(cache/f'draws_seed{seed}.npy',mmap_mode='r')
    net=model_factory(adapter);optimizer=torch.optim.Adam(net.parameters(),lr=.001)
    best=math.inf;saved=None;selected=None;trace=[];started=time.perf_counter();cpu=time.process_time()
    tvx,tvm,tvy,tvym=map(torch.from_numpy,(vx,vm,vy,vym))
    for step in range(1,UPDATES+1):
        ix=draws[step-1];x=np.array(arrays['decoded'][ix],copy=True);mask=np.array(arrays['coverage'][ix],copy=True)
        tx,tm,ty,ym=map(torch.from_numpy,(x,mask,np.array(arrays['labels'][ix],copy=True),np.array(arrays['label_observed'][ix],dtype=bool)))
        optimizer.zero_grad(set_to_none=True);p=net(tx,tm);loss=complete_pair_loss(p,ty,ym,tx,tm)
        if not torch.isfinite(loss): raise ArithmeticError('nonfinite source fitting loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
        if step%200==0:
            net.eval()
            with torch.no_grad(): val=float(complete_pair_loss(net(tvx,tvm),tvy,tvym,tvx,tvm))
            trace.append({'update':step,'batch_train_complete_pair_nmae':float(loss.detach()),'held_time_complete_pair_nmae':val})
            if val<best: best=val;selected=step;saved={k:t.detach().numpy().copy() for k,t in net.state_dict().items()}
            net.train();print(json.dumps({'job':tag,**trace[-1]}),flush=True)
    fit_wall=time.perf_counter()-started;fit_cpu=time.process_time()-cpu
    np.savez_compressed(checkpoint,**saved)
    reloaded=model_factory(adapter)
    with np.load(checkpoint,allow_pickle=False) as z: reloaded.load_state_dict({k:torch.from_numpy(z[k].copy()) for k in z.files})
    reloaded.eval()
    with torch.no_grad(): checked=float(complete_pair_loss(reloaded(tvx,tvm),tvy,tvym,tvx,tvm))
    if abs(checked-best)>2e-7*max(1,abs(best)): raise AssertionError('saved checkpoint validation mismatch')
    write_json(report,{'status':'SOURCE_FIT_COMPLETE_NO_CALIBRATION_PREDICTION','adapter':adapter,'history':history,'seed':seed,
        'updates':UPDATES,'parameter_count':sum(p.numel() for p in net.parameters()),
        'prediction_active_parameter_count':PREDICTION_ACTIVE_PARAMETERS,'selected_update':selected,
        'best_source_held_time_complete_pair_nmae':best,'checkpoint_reload_source_nmae':checked,'trace':trace,
        'fit_wall_seconds':fit_wall,'fit_process_cpu_seconds':fit_cpu,'checkpoint_bytes':checkpoint.stat().st_size,
        'checkpoint_sha256':sha(checkpoint),'script_sha256':sha(__file__),
        'source_cache_manifest_sha256':sha(cache/'manifest.json'),'draws_sha256':sha(cache/f'draws_seed{seed}.npy'),
        'protocol_sha256':sha(OUT/'proposed_protocol.json'),'calibration_or_confirmation_inference_performed':False})
    print(json.dumps({'source_fit_complete':tag,'fit_wall_seconds':fit_wall,'fit_process_cpu_seconds':fit_cpu}),flush=True)


def train_all():
    if not (OUT/'source_cache/manifest.json').exists(): prepare_sources()
    verify_source_cache()
    jobs=[]
    # Prespecified seed-dependent adapter shuffle and alternating history order.
    for si,seed in enumerate(SEEDS):
        order=np.random.default_rng(93000+seed).permutation(ADAPTERS).tolist()
        for ai,adapter in enumerate(order):
            hs=HISTORIES if (si+ai)%2==0 else tuple(reversed(HISTORIES))
            for history in hs:
                p=OUT/'source_fits'/f'{adapter}_{history}_seed{seed}.json'
                if not p.exists(): fit_one(adapter,history,seed)
                row=load_json(p)
                if row['script_sha256']!=sha(__file__) or row['status']!='SOURCE_FIT_COMPLETE_NO_CALIBRATION_PREDICTION': raise ValueError('inconsistent source fit')
                jobs.append({'adapter':adapter,'history':history,'seed':seed,'report_path':str(p),'report_sha256':sha(p),
                    'checkpoint_sha256':row['checkpoint_sha256']})
    if len(jobs)!=36: raise ValueError('source family incomplete')
    write_json(OUT/'source_fit_completion.json',{'status':'ALL36_SOURCE_FITS_COMPLETE_CALIBRATION_GATED','jobs':jobs,
        'script_sha256':sha(__file__),'protocol_sha256':sha(OUT/'proposed_protocol.json'),
        'confirmation_outcomes_scored':False,'calibration_predictions_created':False})


def calibration_predict():
    verify_source_cache()
    gate=load_json(OUT/'root_calibration_gate.json')
    if gate.get('status')!='ROOT_APPROVED_CALIBRATION_INFERENCE': raise PermissionError('root calibration gate closed')
    for key,path in [('script_sha256',Path(__file__)),('protocol_sha256',OUT/'proposed_protocol.json'),
                     ('source_completion_sha256',OUT/'source_fit_completion.json'),
                     ('data_manifest_sha256',DATA/'dataset_manifest_v1.json')]:
        if gate.get(key)!=sha(path): raise ValueError('root gate binding mismatch '+key)
    if not gate.get('near_neighbor_gate_complete'): raise PermissionError('near-neighbor review not complete')
    m=load_json(DATA/'dataset_manifest_v1.json');ids=np.load(DATA/'client_ids.npy',allow_pickle=False).tolist()
    labels=np.load(DATA/'provider_labels.npy',allow_pickle=False)
    v=np.load(DATA/'panel_kw.npy',mmap_mode='r');valid=np.load(DATA/'valid.npy',mmap_mode='r')
    cs=m['customer_groups']['calibration'];cols=[ids.index(i) for i in cs]
    start=int(np.searchsorted(labels,np.datetime64('2013-07-01')));end=int(np.searchsorted(labels,np.datetime64('2014-01-01')))
    origins=np.arange(start,end,96,dtype=np.int64)
    if len(origins)!=184 or cols!=list(range(32,48)): raise ValueError('fixed calibration scope mismatch')
    # Evaluate support and input histories only after the explicit gate opens.
    support=[]
    for c in cols:
        y=np.asarray(v[start-HISTORY:start,c]);ob=np.asarray(valid[start-HISTORY:start,c],bool)
        support.append(max(float(y[ob].mean(dtype=np.float64)) if ob.any() else 0.,1.))
    raw=np.stack([v[o-HISTORY:o,c] for c in cols for o in origins]).astype(np.float32)
    observed=np.stack([valid[o-HISTORY:o,c] for c in cols for o in origins]).astype(np.uint8)
    prepared={h:history_representation(raw,observed,h) for h in HISTORIES}
    reports=[];dest=OUT/'predictions';dest.mkdir(exist_ok=True)
    for seed in SEEDS:
        torch=setup(seed)
        for adapter in ADAPTERS:
            for history in HISTORIES:
                tag=f'{adapter}_{history}_seed{seed}';fit=load_json(OUT/'source_fits'/f'{tag}.json')
                cp=OUT/'source_fits'/f'{tag}.npz'
                if sha(cp)!=fit['checkpoint_sha256']: raise ValueError('checkpoint changed')
                net=model_factory(adapter)
                with np.load(cp,allow_pickle=False) as z: net.load_state_dict({k:torch.from_numpy(z[k].copy()) for k in z.files})
                net.eval();x,mask=prepared[history];parts=[];wall=time.perf_counter();cpu=time.process_time()
                with torch.no_grad():
                    for b in range(0,len(x),128):
                        parts.append(net(torch.from_numpy(x[b:b+128]),torch.from_numpy(mask[b:b+128])).numpy().copy())
                pred=np.concatenate(parts).reshape(len(cols),len(origins),HORIZON).astype(np.float32)
                if not np.isfinite(pred).all(): raise ValueError('nonfinite forecast')
                p=dest/f'{tag}.npz'
                if p.exists(): raise FileExistsError(p)
                np.savez_compressed(p,customer_ids=np.asarray(cs),forecast_dates=labels[origins].astype('datetime64[D]'),
                    origin_indices=origins,predictions_kw=pred,support_scales_kw=np.asarray(support,np.float64))
                r={'status':'COMPLETE_CALIBRATION_FORECAST_UNSCORED','adapter':adapter,'history':history,'seed':seed,
                    'shape':list(pred.shape),'parameter_count':PARAMETERS,
                    'prediction_active_parameter_count':PREDICTION_ACTIVE_PARAMETERS,'checkpoint_sha256':sha(cp),
                    'source_fit_report_sha256':sha(OUT/'source_fits'/f'{tag}.json'),'forecast_sha256':sha(p),
                    'script_sha256':sha(__file__),'root_gate_sha256':sha(OUT/'root_calibration_gate.json'),
                    'prediction_wall_seconds':time.perf_counter()-wall,'prediction_cpu_seconds':time.process_time()-cpu,
                    'confirmation_values_used':False,'future_calibration_truth_scored_by_predictor':False}
                write_json(dest/f'{tag}.json',r);reports.append({'path':str(p),'sha256':sha(p),'report_sha256':sha(dest/f'{tag}.json')})
    references=[];fine_x=prepared['fine'][0]
    for name in QUALITY_REFERENCES:
        wall=time.perf_counter();cpu=time.process_time()
        ref=(np.repeat(fine_x[:,-1:],HORIZON,axis=1) if name=='persistence' else
             fine_x[:,-HORIZON:].copy() if name=='daily_naive' else fine_x[:,:HORIZON].copy())
        ref=ref.reshape(len(cols),len(origins),HORIZON).astype(np.float32)
        p=dest/f'reference_{name}.npz'
        if p.exists(): raise FileExistsError(p)
        np.savez_compressed(p,customer_ids=np.asarray(cs),forecast_dates=labels[origins].astype('datetime64[D]'),
            origin_indices=origins,predictions_kw=ref,support_scales_kw=np.asarray(support,np.float64))
        r={'status':'COMPLETE_CALIBRATION_QUALITY_REFERENCE_UNSCORED','reference':name,
            'shape':list(ref.shape),'fit_count':0,'forecast_sha256':sha(p),'script_sha256':sha(__file__),
            'root_gate_sha256':sha(OUT/'root_calibration_gate.json'),
            'construction_wall_seconds':time.perf_counter()-wall,'construction_cpu_seconds':time.process_time()-cpu,
            'confirmation_values_used':False,'future_calibration_truth_scored_by_predictor':False}
        write_json(dest/f'reference_{name}.json',r)
        references.append({'name':name,'path':str(p),'sha256':sha(p),'report_sha256':sha(dest/f'reference_{name}.json')})
    write_json(OUT/'prediction_completion.json',{'status':'ALL36_CALIBRATION_FORECASTS_COMPLETE_UNSCORED','forecasts':reports,
        'quality_references':references,'formal_fit_count':36,'quality_reference_count':3,
        'script_sha256':sha(__file__),'root_gate_sha256':sha(OUT/'root_calibration_gate.json'),
        'source_completion_sha256':sha(OUT/'source_fit_completion.json'),'data_manifest_sha256':sha(DATA/'dataset_manifest_v1.json'),
        'all_future_target_scoring_independent':True,'confirmation_2014_used':False})


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('preflight','prepare','train','predict'));a=p.parse_args()
    {'preflight':preflight,'prepare':prepare_sources,'train':train_all,'predict':calibration_predict}[a.stage]()


if __name__=='__main__': main()
