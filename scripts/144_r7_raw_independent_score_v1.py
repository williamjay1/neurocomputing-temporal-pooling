"""Independent post-review R7 raw-network numerical audit and complete scoring.

No producer, fitting factory, or metric helper is imported. Both external
periods/customers were known in R6; these results are exploratory sensitivity,
not another independent confirmation. Existing files are never overwritten.
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, json, math, os, platform, sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
sys.dont_write_bytecode=True
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

ROOT=Path('D:/MLWork/Neurocomputing_TailTransfer')
OUT=ROOT/'results/r7_raw_independent_v1'
DATA=ROOT/'datasets/upgrade5_uci_v1'
CACHE=ROOT/'results/r6_access_v1/source_cache'
CODE=Path(__file__).resolve()
CORE=('direct','max_repeat','avg_repeat','haar_absolute','haar_signed','max_blur_repeat')
EXTRA=('mixed','gated','attention','stride_conv')
ADAPTERS=CORE+EXTRA
HISTORIES=('fine','coarse')
SEEDS=(20261001,20261002,20261003)
METRICS=('endpoint_mae_kw','endpoint_nmae','endpoint_mse_kw2','mean_mse_kw2','detail_mse_kw2')
BOOTSTRAP_REPS,BOOTSTRAP_SEED=5000,13320261002
FINE_PAIRS=(('max_repeat','haar_absolute'),('haar_absolute','haar_signed'),('haar_signed','direct'),('max_blur_repeat','max_repeat'))
CHECKS=[]

def check(condition,name):
    if not bool(condition):raise AssertionError(name)
    CHECKS.append(name)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def loadj(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def savej(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def savecsv(path,rows):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    if not rows:raise ValueError('Empty CSV '+str(p))
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def bind(path,expected,label):
    actual=sha(path);check(actual==expected,label+' hash');return actual

def tag(a,h,s):return f'{a}_{h}_seed{s}'

def own_history(raw,observed,history):
    """Observed-only adjacent means; causal last observation, initially zero."""
    x=np.asarray(raw,dtype=np.float32).copy();ob=np.asarray(observed,dtype=bool)
    if history=='coarse':
        pairs=x.reshape(len(x),336,2);live=ob.reshape(len(x),336,2)
        count=live.sum(-1)
        total=np.where(live,pairs,0.).astype(np.float64).sum(-1)
        mean=np.zeros_like(total)
        np.divide(total,count,out=mean,where=count>0)
        x=np.repeat(mean.astype(np.float32),2,axis=-1)
        ob=np.repeat(count>0,2,axis=-1)
    carry=np.zeros(len(x),np.float32)
    for i in range(672):
        good=ob[:,i];carry[good]=x[good,i];x[:,i]=carry
    check(np.isfinite(x).all(),'independent '+history+' history finite')
    return x

def tensor_shapes(adapter):
    shapes={'layers.0.weight':(64,672),'layers.0.bias':(64,),
            'layers.2.weight':(64,64),'layers.2.bias':(64,),
            'layers.4.weight':(96,64),'layers.4.bias':(96,),
            'center':(),'scale':()}
    if adapter in ('mixed','attention'):shapes['adapter.theta']=()
    if adapter in ('gated','stride_conv'):shapes['adapter.coefficients']=(3,)
    return shapes

def own_forward_torch(x,weights,adapter,require_grad=False):
    """Explicit saved-tensor forward; no producer class/factory is used."""
    import torch
    import torch.nn.functional as F
    tensors={k:torch.tensor(v,requires_grad=require_grad and k not in ('center','scale')) for k,v in weights.items()}
    z=(torch.as_tensor(x,dtype=torch.float32)-tensors['center'])/tensors['scale']
    if adapter=='max_blur_repeat':
        # Dense max over current/next; replicated last edge, then blur/decimate.
        dense=torch.maximum(z,torch.cat((z[:,1:],z[:,-1:]),dim=-1))
        z=F.conv1d(F.pad(dense[:,None,:],(1,1),mode='replicate'),
                   torch.tensor([.25,.5,.25]).reshape(1,1,3),stride=2).squeeze(1).repeat_interleave(2,-1)
    elif adapter!='direct':
        p=z.reshape(len(z),336,2)
        if adapter=='max_repeat':pooled=p.amax(-1)
        elif adapter=='avg_repeat':pooled=p.mean(-1)
        elif adapter in ('haar_absolute','haar_signed'):
            c=(p[...,0]+p[...,1])/math.sqrt(2)
            d=(p[...,0]-p[...,1])/math.sqrt(2)
            z=torch.stack((c,d.abs() if adapter=='haar_absolute' else d),-1).flatten(-2)
            pooled=None
        elif adapter=='mixed':
            q=tensors['adapter.theta'].sigmoid();pooled=q*p.amax(-1)+(1-q)*p.mean(-1)
        elif adapter=='gated':
            q=tensors['adapter.coefficients'];g=(q[0]*p[...,0]+q[1]*p[...,1]+q[2]).sigmoid()
            pooled=g*p.amax(-1)+(1-g)*p.mean(-1)
        elif adapter=='attention':
            pooled=(torch.softmax(p*tensors['adapter.theta'],dim=-1)*p).sum(-1)
        elif adapter=='stride_conv':
            q=tensors['adapter.coefficients'];pooled=q[0]*p[...,0]+q[1]*p[...,1]+q[2]
        else:raise ValueError(adapter)
        if pooled is not None:z=pooled.repeat_interleave(2,-1)
    for i in (0,2,4):
        z=F.linear(z,tensors[f'layers.{i}.weight'],tensors[f'layers.{i}.bias'])
        if i!=4:z=F.relu(z)
    out=z*tensors['scale']+tensors['center']
    if require_grad:
        out.mean().backward()
        trainable={k:v for k,v in tensors.items() if v.requires_grad}
        check(all(v.grad is not None and torch.isfinite(v.grad).all() for v in trainable.values()),'all learned tensors forecast-reachable '+adapter)
        return out.detach().numpy(),sum(v.numel() for v in trainable.values())
    return out.detach().numpy()

def own_forward_numpy64(x,w,adapter):
    """Second implementation of selected rows, in double precision."""
    z=(np.asarray(x,np.float64)-float(w['center']))/float(w['scale'])
    if adapter=='max_blur_repeat':
        dense=np.maximum(z,np.concatenate((z[:,1:],z[:,-1:]),axis=1))
        padded=np.pad(dense,((0,0),(1,1)),mode='edge')
        z=(.25*padded[:,:-2]+.5*padded[:,1:-1]+.25*padded[:,2:])[:,::2].repeat(2,axis=1)
    elif adapter!='direct':
        p=z.reshape(len(z),336,2);mean=p.mean(-1);maximum=p.max(-1)
        if adapter=='max_repeat':pooled=maximum
        elif adapter=='avg_repeat':pooled=mean
        elif adapter in ('haar_absolute','haar_signed'):
            c=(p[...,0]+p[...,1])/math.sqrt(2);d=(p[...,0]-p[...,1])/math.sqrt(2)
            z=np.stack((c,np.abs(d) if adapter=='haar_absolute' else d),-1).reshape(len(z),672);pooled=None
        elif adapter=='mixed':
            g=1/(1+np.exp(-float(w['adapter.theta'])));pooled=g*maximum+(1-g)*mean
        elif adapter=='gated':
            q=w['adapter.coefficients'].astype(np.float64)
            logit=q[0]*p[...,0]+q[1]*p[...,1]+q[2]
            g=np.exp(-np.logaddexp(0.,-logit));pooled=g*maximum+(1-g)*mean
        elif adapter=='attention':
            score=p*float(w['adapter.theta']);score-=score.max(-1,keepdims=True)
            mass=np.exp(score);mass/=mass.sum(-1,keepdims=True);pooled=(mass*p).sum(-1)
        elif adapter=='stride_conv':
            q=w['adapter.coefficients'].astype(np.float64);pooled=q[0]*p[...,0]+q[1]*p[...,1]+q[2]
        else:raise ValueError(adapter)
        if pooled is not None:z=pooled.repeat(2,axis=1)
    for i in (0,2,4):
        z=z@w[f'layers.{i}.weight'].astype(np.float64).T+w[f'layers.{i}.bias'].astype(np.float64)
        if i!=4:z=np.maximum(z,0.)
    return z*float(w['scale'])+float(w['center'])

def read_weights(path,adapter,center,scale):
    with np.load(path,allow_pickle=False) as f:weights={k:f[k].copy() for k in f.files}
    check({k:v.shape for k,v in weights.items()}==tensor_shapes(adapter),'exact keys/shapes, no trainable padding '+path.stem)
    check(all(v.dtype==np.float32 and np.isfinite(v).all() for v in weights.values()),'finite float32 tensors '+path.stem)
    check(weights['center']==np.float32(center) and weights['scale']==np.float32(scale),'identical global buffers '+path.stem)
    count=sum(v.size for k,v in weights.items() if k not in ('center','scale'))
    expected=53472+(1 if adapter in ('mixed','attention') else 3 if adapter in ('gated','stride_conv') else 0)
    check(count==expected,'forecast-active coefficient count '+path.stem)
    return weights,count

def source_gate(family):
    protocol=loadj(family/'protocol.json');complete=loadj(family/'training_completion.json')
    check(protocol['core_adapters']==list(CORE) and protocol['known_nonmatched_parameter_references']==list(EXTRA),'full ten-operation family')
    check(protocol['histories']==list(HISTORIES) and protocol['seeds']==list(SEEDS),'histories and seeds frozen')
    check(protocol['formal_fit_count']==complete['formal_fit_count']==60,'all60 source fits completed')
    check(protocol['source_training_rows']==11952 and protocol['source_validation_rows']==104,'frozen source sample sizes')
    check(protocol['updates']==2000 and protocol['batch']==64 and protocol['learning_rate']==.001,'common source update budget')
    check(protocol['per_window_mean_std_as_inputs'] is False and protocol['last_value_residual_forecast_bypass'] is False,'no local statistic or residual bypass declared')
    bind(family/'protocol.json',complete['protocol_sha256'],'completed source protocol')
    # Binding code is a read/hash action, never a Python import.
    producers=[ROOT/'scripts/138_r7_raw_access_experiment_v1.py',ROOT/'scripts/145_r7_raw_robust_source_experiment_v1.py']
    code_matches=[p for p in producers if p.exists() and sha(p)==protocol['script_sha256']]
    if not code_matches:
        code_matches=[p for p in (ROOT/'scripts').glob('145*.py') if sha(p)==protocol['script_sha256']]
    check(bool(code_matches),'declared producer code frozen (not imported)')
    check(complete['script_sha256']==protocol['script_sha256'],'source completion code binding')
    bind(CACHE/'manifest.json',protocol['source_cache_manifest_sha256'],'original source cache')
    cachemanifest=loadj(CACHE/'manifest.json')
    for n,r in cachemanifest['files'].items():bind(CACHE/n,r['sha256'],'source array '+n)
    dataset=loadj(DATA/'dataset_manifest_v1.json')
    for n in ('panel_kw.npy','valid.npy','provider_labels.npy','client_ids.npy'):bind(DATA/n,dataset['artifacts'][n]['sha256'],'frozen external '+n)
    bind(DATA/'dataset_manifest_v1.json',cachemanifest['data_manifest_sha256'],'source-bound data manifest')
    expected={tag(a,h,s) for a in ADAPTERS for h in HISTORIES for s in SEEDS}
    check({Path(r['checkpoint']).stem for r in complete['fits']}==expected,'complete60 checkpoint identities')
    reports={}
    for b in complete['fits']:
        cp=ROOT/b['checkpoint'];rp=cp.with_suffix('.json');name=cp.stem
        bind(cp,b['sha256'],'selected checkpoint '+name);bind(rp,b['report_sha256'],'fit report '+name)
        r=loadj(rp);reports[name]=r
        check(name==tag(r['adapter'],r['history'],r['seed']),'fit identity '+name)
        check(r['external_truth_scored'] is False and r['additional_source_row_scaler_as_network_input'] is False,'source-only selection '+name)
        check(r['checkpoint_sha256']==b['sha256'] and r['protocol_sha256']==complete['protocol_sha256'] and r['script_sha256']==protocol['script_sha256'],'fit provenance chain '+name)
        trace=r['validation_trace'];check([t['update'] for t in trace]==list(range(200,2001,200)),'ten source validation points '+name)
        selected=min(trace,key=lambda t:t['common_weight_source_nmae'])
        check(selected['update']==r['selected_update'] and selected['common_weight_source_nmae']==r['source_validation_nmae'],'earliest minimum source checkpoint '+name)
        _,count=read_weights(cp,r['adapter'],protocol['fixed_global_center_kw'],protocol['fixed_global_scale_kw'])
        check(count==r['registered_and_forecast_active_parameters'],'reported parameters '+name)
    return dataset,cachemanifest,protocol,complete,reports

def source_arrays_audit():
    labels=np.load(DATA/'provider_labels.npy',allow_pickle=False)
    panel=np.load(DATA/'panel_kw.npy',mmap_mode='r');valid=np.load(DATA/'valid.npy',mmap_mode='r')
    index=lambda day:int(np.searchsorted(labels,np.datetime64(day)))
    keys_by_split={
      'train':[(c,o) for c in range(24) for o in range(index('2012-01-08'),index('2012-09-15')-96,48)
               if np.mean(valid[o:o+96,c])>=.95 and np.mean(valid[o-672:o,c])>=.95],
      'validation':[(c,o) for c in range(8) for o in range(index('2012-09-16'),index('2012-09-30')-96,96)
                    if np.mean(valid[o:o+96,c])>=.95]}
    for split,keys in keys_by_split.items():
        check(len(keys)==(11952 if split=='train' else 104),'independent '+split+' count')
        check(np.array_equal(np.load(CACHE/f'{split}_row_keys.npy'),np.asarray(keys,np.int64)),'independent '+split+' key selection')
        arrays={n:np.load(CACHE/f'{split}_{n}.npy',mmap_mode='r') for n in ('values','observed','labels','label_observed','fine_decoded','coarse_decoded')}
        for b in range(0,len(keys),256):
            part=keys[b:b+256];sl=slice(b,b+len(part))
            check(all(o+96<index('2012-10-01') for _,o in part),'actual source chronology '+split+str(b))
            x=np.stack([panel[o-672:o,c] for c,o in part]).astype(np.float32)
            ob=np.stack([valid[o-672:o,c] for c,o in part]).astype(np.uint8)
            y=np.stack([panel[o:o+96,c] for c,o in part]).astype(np.float32)
            ym=np.stack([valid[o:o+96,c] for c,o in part]).astype(np.uint8)
            for n,v in (('values',x),('observed',ob),('labels',np.nan_to_num(y)),('label_observed',ym)):
                check(np.array_equal(arrays[n][sl],v,equal_nan=True),'source cache independent '+split+'/'+n+'/'+str(b))
            for h in HISTORIES:check(np.array_equal(own_history(x,ob,h),arrays[h+'_decoded'][sl]),'independent source decode '+split+'/'+h+'/'+str(b))
    for s in SEEDS:
        expected=np.random.default_rng(s).integers(11952,size=(2000,64),dtype=np.int32)
        check(np.array_equal(np.load(CACHE/f'draws_seed{s}.npy'),expected),'independent frozen draw reconstruction '+str(s))

def source(family,branch):
    import torch
    torch.set_num_threads(2)
    dest=OUT/branch;dest.mkdir(parents=True,exist_ok=True)
    _,cm,p,complete,reports=source_gate(family)
    source_arrays_audit()
    raw=np.load(CACHE/'train_values.npy');ob=np.load(CACHE/'train_observed.npy').astype(bool)
    measured=raw[ob].astype(np.float64)
    center=float(measured.mean());scale=max(float(measured.std()),1.)
    median=float(np.median(measured));mad=max(float(1.4826*np.median(np.abs(measured-median))),1.)
    if np.isclose(p['fixed_global_center_kw'],center,rtol=0,atol=1e-10) and np.isclose(p['fixed_global_scale_kw'],scale,rtol=0,atol=1e-10):normalizer='source observed arithmetic mean / population std'
    elif np.isclose(p['fixed_global_center_kw'],median,rtol=0,atol=1e-10) and np.isclose(p['fixed_global_scale_kw'],mad,rtol=0,atol=1e-10):normalizer='source observed median / 1.4826 MAD floored1'
    else:raise AssertionError('Source normalizer does not match independently reconstructed mean/std or median/MAD')
    vxraw=np.load(CACHE/'validation_values.npy');vxob=np.load(CACHE/'validation_observed.npy').astype(bool)
    vx={h:own_history(vxraw,vxob,h) for h in HISTORIES}
    truth=np.load(CACHE/'validation_labels.npy').astype(np.float64)
    observed=np.load(CACHE/'validation_label_observed.npy').astype(bool)
    live=observed.reshape(104,48,2).all(-1)
    counts=np.maximum(live.sum(1),1)
    weights=np.maximum(np.where(vxob,vxraw,0.).sum(1,dtype=np.float64)/np.maximum(vxob.sum(1),1),1.)
    rows=[];numpy_checks=[];gradient_checked=set()
    for name,r in sorted(reports.items()):
        cp=family/'fits'/f'{name}.npz';w,_=read_weights(cp,r['adapter'],p['fixed_global_center_kw'],p['fixed_global_scale_kw'])
        pred=own_forward_torch(vx[r['history']],w,r['adapter'])
        er=np.where(np.repeat(live,2,1),pred.astype(np.float64)-truth,0.).reshape(104,48,2)
        own=float(np.mean(np.abs(er).mean(2).sum(1)/counts/weights))
        delta=abs(own-r['source_validation_nmae'])
        check(delta<4e-6*max(1,abs(own)),'104row selected loss independently reproduced '+name)
        npred=own_forward_numpy64(vx[r['history']][:8],w,r['adapter'])
        maxdiff=float(np.abs(npred-pred[:8]).max())
        numpy_checks.append({'adapter':r['adapter'],'history':r['history'],'seed':r['seed'],'rows':8,'max_kw_numpy64_vs_torch32':maxdiff})
        if r['adapter'] not in gradient_checked:
            fixture=np.random.default_rng(14420261002).normal(p['fixed_global_center_kw'],p['fixed_global_scale_kw'],size=(7,672)).astype(np.float32)
            _,active=own_forward_torch(fixture,w,r['adapter'],True)
            check(active==r['registered_and_forecast_active_parameters'],'no padding/reachable coefficient count '+r['adapter'])
            gradient_checked.add(r['adapter'])
        rows.append({'adapter':r['adapter'],'history':r['history'],'seed':r['seed'],'selected_update':r['selected_update'],
                     'independent_source_validation_nmae':own,'producer_source_validation_nmae':r['source_validation_nmae'],
                     'absolute_metric_float64_vs_float32_disagreement':delta,'parameters':r['registered_and_forecast_active_parameters']})
    refs=[]
    for n in ('persistence','daily_naive','weekly_naive'):
        x=vx['fine'];pred=np.repeat(x[:,-1:],96,1) if n=='persistence' else x[:,-96:] if n=='daily_naive' else x[:,:96]
        err=np.where(np.repeat(live,2,1),pred.astype(np.float64)-truth,0.).reshape(104,48,2)
        loss=float(np.mean(np.abs(err).mean(2).sum(1)/counts/weights))
        refs.append({'reference':n,'source_validation_nmae':loss,'rows':104,'same_source_pair_mask':True})
    # R6 source values already independently restored by script133, loaded only as a numerical comparator.
    r6=[]
    with (ROOT/'results/r6_independent_v1/source_checkpoint_validation_independent.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):r6.append({'adapter':r['adapter'],'history':r['history'],'seed':int(r['seed']),'source_validation_nmae':float(r['independent_source_validation_nmae'])})
    savecsv(dest/'source_checkpoint_validation.csv',rows)
    savecsv(dest/'source_numpy64_forward_tolerance.csv',numpy_checks)
    savecsv(dest/'source_quality_references.csv',refs)
    savecsv(dest/'r6_source_competence_comparison.csv',r6)
    report={'status':'PASS_ALL60_SOURCE_CHECKPOINTS','branch':branch,'family_path':str(family),'scorer_sha256':sha(CODE),
       'protocol_sha256':sha(family/'protocol.json'),'training_completion_sha256':sha(family/'training_completion.json'),
       'source_cache_manifest_sha256':sha(CACHE/'manifest.json'),'data_manifest_sha256':sha(DATA/'dataset_manifest_v1.json'),
       'normalizer':normalizer,'center_kw':p['fixed_global_center_kw'],'scale_kw':p['fixed_global_scale_kw'],
       'source_training_rows':11952,'source_validation_rows':104,'checkpoints':60,
       'max_metric_float64_vs_float32_disagreement':max(r['absolute_metric_float64_vs_float32_disagreement'] for r in rows),
       'max_numpy64_vs_torch32_source_forecast_difference_kw':max(r['max_kw_numpy64_vs_torch32'] for r in numpy_checks),
       'nmae_range':[min(r['independent_source_validation_nmae'] for r in rows),max(r['independent_source_validation_nmae'] for r in rows)],
       'source_naive_references':refs,'r6_independent_source_nmae_range':[min(r['source_validation_nmae'] for r in r6),max(r['source_validation_nmae'] for r in r6)],
       'no_producer_or_fitter_or_metric_imported':True,'external_target_scores_in_this_stage':False,
       'checks':list(CHECKS),'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__}
    savej(dest/'source_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}),flush=True)

def metrics(pred,truth,live,scales):
    # Invalid truths removed before subtraction; complete-pair mask shared.
    live2=np.repeat(live,2,2)
    safe=np.where(live2,truth,0.)
    err=np.where(live2,np.asarray(pred,np.float64)-safe,0.).reshape(*live.shape,2)
    mae=np.abs(err).mean(-1);mse=np.square(err).mean(-1)
    mean=err.mean(-1);detail=(err[...,1]-err[...,0])/2
    discrepancy=float(np.max(np.abs(mse-mean**2-detail**2)))
    check(discrepancy<2e-12*max(1.,float(mse.max())),'per-pair squared-loss decomposition')
    return np.stack((mae,mae/scales[:,None,None],mse,mean**2,detail**2))

def bootstrap(sums,counts):
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    nc,nw=counts.shape;reps=np.empty((BOOTSTRAP_REPS,sums.shape[0],sums.shape[1]),np.float64)
    for b in range(BOOTSTRAP_REPS):
        c=rng.integers(nc,size=nc);w=rng.integers(nw,size=nw)
        denominator=counts[np.ix_(c,w)].sum(1)
        check(np.all(denominator>0),'bootstrap positive availability '+str(b))
        reps[b]=(sums[:,:,c][:,:,:,w].sum(3)/denominator[None,None,:]).mean(2)
    return reps

def forecast_gate(family):
    _,_,p,tc,reports=source_gate(family)
    complete=loadj(family/'forecast_completion.json')
    check(complete['formal_fit_count']==60 and complete['external_data_previously_scored'] is True and complete['not_independent_confirmation'] is True,'post-review exploratory complete forecast family')
    bind(family/'training_completion.json',complete['training_completion_sha256'],'forecast follows source completion')
    bind(family/'protocol.json',complete['protocol_sha256'],'forecast protocol')
    for phase in ('calibration','confirmation'):
        entries=complete['phases'][phase]
        check(len(entries)==60,'all60 '+phase+' files retained')
        expected={tag(a,h,s) for a in ADAPTERS for h in HISTORIES for s in SEEDS}
        check({Path(r['path']).stem for r in entries}==expected,'forecast complete identities '+phase)
        for b in entries:
            fp=ROOT/b['path'];rp=fp.with_suffix('.json');r=loadj(rp)
            bind(fp,b['sha256'],'forecast '+phase+'/'+fp.stem);bind(rp,b['report_sha256'],'forecast report '+phase+'/'+fp.stem)
            check(r['forecast_sha256']==b['sha256'] and r['checkpoint_sha256']==reports[fp.stem]['checkpoint_sha256'],'forecast-source checkpoint binding '+phase+'/'+fp.stem)
            check(r['future_labels_scored_in_producer'] is False and r['post_review_exploratory'] is True,'prediction unscored declaration '+phase+'/'+fp.stem)
    return p,reports,complete

def score_phase(family,branch,phase,p,reports):
    import torch
    torch.set_num_threads(2)
    dest=OUT/branch;nc,nd=(16,184) if phase=='calibration' else (32,365)
    manifest=loadj(DATA/'dataset_manifest_v1.json');clients=manifest['customer_groups'][phase]
    ids=np.load(DATA/'client_ids.npy');cols=[ids.tolist().index(c) for c in clients]
    check(cols==list(range(32,48) if phase=='calibration' else range(48,80)),'frozen customer columns '+phase)
    labels=np.load(DATA/'provider_labels.npy');index=lambda d:int(np.searchsorted(labels,np.datetime64(d)))
    start,stop=(index('2013-07-01'),index('2014-01-01')) if phase=='calibration' else (index('2014-01-01'),index('2015-01-01'))
    origins=np.arange(start,stop,96,dtype=np.int64);dates=labels[origins].astype('datetime64[D]')
    check(len(origins)==nd,'full fixed calendar '+phase)
    panel=np.load(DATA/'panel_kw.npy',mmap_mode='r');valid=np.load(DATA/'valid.npy',mmap_mode='r')
    truth=np.stack([np.stack([panel[o:o+96,c] for o in origins]) for c in cols]).astype(np.float64)
    observed=np.stack([np.stack([valid[o:o+96,c] for o in origins]) for c in cols]).astype(bool)
    live=observed.reshape(nc,nd,48,2).all(-1)
    check(np.isfinite(truth.reshape(nc,nd,48,2)[live]).all(),'complete-pair truth finite '+phase)
    scales=[]
    for c in cols:
        ob=valid[start-672:start,c].astype(bool);v=panel[start-672:start,c].astype(np.float64)
        scales.append(max(float(v[ob].mean()) if ob.any() else 0.,1.))
    scales=np.asarray(scales,np.float64)
    raw=np.stack([panel[o-672:o,c] for c in cols for o in origins]).astype(np.float32)
    ob=np.stack([valid[o-672:o,c] for c in cols for o in origins]).astype(np.uint8)
    prepared={h:own_history(raw,ob,h) for h in HISTORIES}
    daycounts=live.sum(-1);customer_counts=daycounts.sum(1)
    check(np.all(customer_counts>0),'every customer retained '+phase)
    weekdates=dates-((dates.astype(np.int64)+3)%7).astype('timedelta64[D]')
    weeks,weekindex=np.unique(weekdates,return_inverse=True)
    counts_week=np.stack([daycounts[:,weekindex==w].sum(1) for w in range(len(weeks))],axis=1)
    cells=[(a,h) for a in ADAPTERS for h in HISTORIES]
    sums_week=np.zeros((5,20,nc,len(weeks)),np.float64)
    seed_scores=[];customer_rows=[];date_rows=[];forward_checks=[]
    coverage=[{'customer_id':c,'column':cols[j],'complete_pairs':int(customer_counts[j]),'all_possible_pairs':nd*48,
               'observed_endpoints':int(observed[j].sum()),'past_support_scale_kw':float(scales[j])} for j,c in enumerate(clients)]
    fullpath=dest/f'{phase}_metrics_by_customer_date.csv.gz'
    if fullpath.exists():raise FileExistsError(fullpath)
    with gzip.open(fullpath,'wt',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['adapter','history','seed','customer_id','forecast_date','complete_pairs']+list(METRICS));writer.writeheader()
        for ci,(a,h) in enumerate(cells):
            for seed in SEEDS:
                name=tag(a,h,seed);fp=family/phase/f'{name}.npz'
                with np.load(fp,allow_pickle=False) as z:
                    check(set(z.files)=={'customer_ids','forecast_dates','origin_indices','predictions_kw','support_scales_kw'},'forecast schema '+phase+'/'+name)
                    check(z['customer_ids'].tolist()==clients and np.array_equal(z['forecast_dates'],dates) and np.array_equal(z['origin_indices'],origins),'customer date origin alignment '+phase+'/'+name)
                    check(np.array_equal(z['support_scales_kw'],scales),'past scale exact '+phase+'/'+name)
                    pred=z['predictions_kw'].copy()
                check(pred.shape==(nc,nd,96) and pred.dtype==np.float32 and np.isfinite(pred).all(),'finite full forecast '+phase+'/'+name)
                # Reconstruct original producer-sized first and last blocks.
                w,_=read_weights(family/'fits'/f'{name}.npz',a,p['fixed_global_center_kw'],p['fixed_global_scale_kw'])
                flat=pred.reshape(-1,96);maxdiff=0.;maxrelative=0.;rows_rebuilt=0
                for b in sorted({0,((len(flat)-1)//256)*256}):
                    restored=own_forward_torch(prepared[h][b:b+256],w,a)
                    target=flat[b:b+256]
                    difference=np.abs(restored.astype(np.float64)-target.astype(np.float64))
                    maxdiff=max(maxdiff,float(difference.max()));maxrelative=max(maxrelative,float((difference/(1+np.abs(target))).max()))
                    check(np.allclose(restored,target,atol=.02,rtol=2e-5),'independent saved-weight external forward '+phase+'/'+name+'/'+str(b))
                    rows_rebuilt+=len(target)
                forward_checks.append({'adapter':a,'history':h,'seed':seed,'rows_reconstructed':rows_rebuilt,
                                       'max_absolute_forecast_difference_kw':maxdiff,'max_difference_div_1plus_abs_forecast':maxrelative})
                errors=metrics(pred,truth,live,scales);daily=errors.sum(-1)
                sums=np.stack([daily[:,:,weekindex==wi].sum(-1) for wi in range(len(weeks))],axis=-1)
                sums_week[:,ci]+=sums/3.
                percustomer=daily.sum(-1)/customer_counts[None,:];macro=percustomer.mean(-1)
                r=reports[name]
                seed_scores.append({'adapter':a,'history':h,'seed':seed,'customers':nc,'calendar_dates':nd,'complete_pairs':int(customer_counts.sum()),
                    **{m:float(macro[k]) for k,m in enumerate(METRICS)},'selected_source_update':r['selected_update'],
                    'source_validation_nmae':r['source_validation_nmae'],'parameters':r['registered_and_forecast_active_parameters']})
                for j,c in enumerate(clients):
                    customer_rows.append({'adapter':a,'history':h,'seed':seed,'customer_id':c,'complete_pairs':int(customer_counts[j]),**{m:float(percustomer[k,j]) for k,m in enumerate(METRICS)}})
                    for di,date in enumerate(dates):
                        n=int(daycounts[j,di]);writer.writerow({'adapter':a,'history':h,'seed':seed,'customer_id':c,'forecast_date':str(date),'complete_pairs':n,
                            **{m:float(daily[k,j,di]/n) if n else '' for k,m in enumerate(METRICS)}})
                for di,date in enumerate(dates):
                    n=daycounts[:,di];good=n>0
                    date_rows.append({'adapter':a,'history':h,'seed':seed,'forecast_date':str(date),'customers':int(good.sum()),'complete_pairs':int(n.sum()),
                        **{m:float(np.mean(daily[k,good,di]/n[good])) for k,m in enumerate(METRICS)}})
    macro=(sums_week.sum(-1)/customer_counts[None,None,:]).mean(-1)
    summaries=[]
    for ci,(a,h) in enumerate(cells):
        group=[r for r in seed_scores if (r['adapter'],r['history'])==(a,h)]
        summaries.append({'adapter':a,'history':h,'fits':3,**{m+'_three_seed_mean':float(macro[k,ci]) for k,m in enumerate(METRICS)},
           'endpoint_rmse_sqrt_mean_mse_kw':float(math.sqrt(macro[2,ci])),
           'nmae_seed_min':min(r['endpoint_nmae'] for r in group),'nmae_seed_max':max(r['endpoint_nmae'] for r in group),
           'source_validation_nmae_three_seed_mean':float(np.mean([r['source_validation_nmae'] for r in group]))})
    def vector(terms):
        v=np.zeros(20)
        for a,h,n in terms:v[cells.index((a,h))]+=n
        return v
    contrasts={a+'_fine_minus_coarse':vector([(a,'fine',1),(a,'coarse',-1)]) for a in ADAPTERS}
    contrasts['primary_max_minus_avg_history_interaction']=vector([('max_repeat','fine',1),('max_repeat','coarse',-1),('avg_repeat','fine',-1),('avg_repeat','coarse',1)])
    for h in HISTORIES:
        for a,b in FINE_PAIRS:contrasts[a+'_minus_'+b+'_'+h]=vector([(a,h,1),(b,h,-1)])
    reps=bootstrap(sums_week,counts_week);contrast_rows=[];seed_contrast_rows=[]
    fine_names=[a+'_minus_'+b+'_fine' for a,b in FINE_PAIRS]
    fine_boot=np.stack([reps@contrasts[n] for n in fine_names],axis=-1)
    fine_point=np.stack([macro@contrasts[n] for n in fine_names],axis=-1)
    fine_se=fine_boot.std(0,ddof=1)
    sup=np.max(np.abs((fine_boot-fine_point[None,:,:])/np.maximum(fine_se[None,:,:],1e-20)),axis=-1)
    critical=np.quantile(sup,.95,axis=0)
    for n,v in contrasts.items():
        boot=reps@v;point=macro@v
        for k,m in enumerate(METRICS):
            lo,hi=np.quantile(boot[:,k],[.025,.975])
            b_lo,b_hi=np.quantile(boot[:,k],[.00625,.99375]) if n in fine_names else (np.nan,np.nan)
            fi=fine_names.index(n) if n in fine_names else None
            contrast_rows.append({'contrast':n,'metric':m,'three_seed_mean_difference':float(point[k]),'conditional_ci95_low':float(lo),'conditional_ci95_high':float(hi),
                'bootstrap_se':float(boot[:,k].std(ddof=1)),'bonferroni_four_fine_ci95_low':float(b_lo) if fi is not None else '',
                'bonferroni_four_fine_ci95_high':float(b_hi) if fi is not None else '',
                'centered_max_t_four_fine_ci95_low':float(point[k]-critical[k]*fine_se[k,fi]) if fi is not None else '',
                'centered_max_t_four_fine_ci95_high':float(point[k]+critical[k]*fine_se[k,fi]) if fi is not None else '',
                'bootstrap_reps':BOOTSTRAP_REPS,'bootstrap_seed':BOOTSTRAP_SEED,'customers':nc,'monday_week_blocks':len(weeks),
                'scope':'conditional on three fixed fits; post-review exploratory known external data'})
            for seed in SEEDS:
                vals=np.asarray([next(r[m] for r in seed_scores if (r['adapter'],r['history'],r['seed'])==(a,h,seed)) for a,h in cells])
                seed_contrast_rows.append({'contrast':n,'metric':m,'seed':seed,'difference':float(vals@v)})
    refrows=[]
    oldfolder=ROOT/('results/r6_access_v1/predictions' if phase=='calibration' else 'results/r6_confirmation_v1/predictions')
    original_score=ROOT/('results/r6_independent_v1/uci_quality_references.csv' if phase=='calibration' else 'results/r6_independent_v1/confirmation_quality_references.csv')
    with original_score.open(encoding='utf-8-sig',newline='') as f:old_scores={r['reference']:r for r in csv.DictReader(f)}
    for n in ('persistence','daily_naive','weekly_naive'):
        x=prepared['fine'];pred=(np.repeat(x[:,-1:],96,1) if n=='persistence' else x[:,-96:] if n=='daily_naive' else x[:,:96]).reshape(nc,nd,96)
        with np.load(oldfolder/f'reference_{n}.npz',allow_pickle=False) as z:
            check(np.array_equal(pred,z['predictions_kw']) and z['customer_ids'].tolist()==clients and np.array_equal(z['forecast_dates'],dates) and np.array_equal(z['origin_indices'],origins) and np.array_equal(z['support_scales_kw'],scales),'original naive exact reconstruction '+phase+'/'+n)
        er=metrics(pred,truth,live,scales)
        own=(er.sum((-1,-2))/customer_counts[None,:]).mean(-1)
        largest=max(abs(own[k]-float(old_scores[n][m])) for k,m in enumerate(METRICS))
        check(all(np.isclose(own[k],float(old_scores[n][m]),rtol=2e-12,atol=1e-9) for k,m in enumerate(METRICS)),'original naive same-mask scores '+phase+'/'+n)
        refrows.append({'reference':n,'fit_count':0,'customers':nc,'complete_pairs':int(customer_counts.sum()),**{m:float(own[k]) for k,m in enumerate(METRICS)},'largest_absolute_old_score_difference':float(largest)})
    tables={'metrics_by_seed':seed_scores,'metrics_summary':summaries,'metrics_by_customer':customer_rows,'metrics_by_date':date_rows,
            'coverage_and_scales':coverage,'contrasts_conditional_bootstrap':contrast_rows,'contrasts_by_seed':seed_contrast_rows,
            'quality_references':refrows,'saved_weight_forward_checks':forward_checks}
    for n,rows in tables.items():savecsv(dest/f'{phase}_{n}.csv',rows)
    statistic=dest/f'{phase}_paired_week_sufficient_statistics.npz'
    if statistic.exists():raise FileExistsError(statistic)
    np.savez_compressed(statistic,metric_names=np.asarray(METRICS),cell_adapter=np.asarray([a for a,h in cells]),cell_history=np.asarray([h for a,h in cells]),
      customer_ids=np.asarray(clients),calendar_week_start=weeks,observed_pair_counts=counts_week,three_seed_mean_error_sums=sums_week,
      bootstrap_cell_macro_losses=reps,bootstrap_seed=np.asarray(BOOTSTRAP_SEED),bootstrap_reps=np.asarray(BOOTSTRAP_REPS))
    files=list(tables);outputs={str(dest/f'{phase}_{n}.csv'):sha(dest/f'{phase}_{n}.csv') for n in files}
    outputs[str(fullpath)]=sha(fullpath);outputs[str(statistic)]=sha(statistic)
    report={'status':'PASS_ALL60_'+phase.upper()+'_POST_REVIEW_EXPLORATORY','branch':branch,'scorer_sha256':sha(CODE),
      'forecast_completion_sha256':sha(family/'forecast_completion.json'),'source_audit_sha256':sha(dest/'source_audit.json'),
      'customers':nc,'dates':nd,'complete_pairs':int(customer_counts.sum()),'monday_week_blocks':len(weeks),'fits':60,
      'bootstrap_reps':BOOTSTRAP_REPS,'bootstrap_seed':BOOTSTRAP_SEED,'primary':[r for r in contrast_rows if r['contrast']=='primary_max_minus_avg_history_interaction' and r['metric']=='endpoint_nmae'],
      'max_external_saved_weight_forward_difference_kw':max(r['max_absolute_forecast_difference_kw'] for r in forward_checks),
      'max_external_saved_weight_forward_scaled_difference':max(r['max_difference_div_1plus_abs_forecast'] for r in forward_checks),
      'forward_acceptance_tolerance':'float32 independent torch full-size blocks: atol .02kW, rtol2e-5; measured discrepancy separately retained',
      'metric_arithmetic':'float64 from saved float32 forecasts; complete adjacent observed pairs only; masked before arithmetic',
      'aggregation':'within customer over available pairs, equal-customer macro; mean of three seed losses, never ensemble prediction',
      'support_scale':'observed past672provider slots Jun24-30/Dec25-31; arithmetic mean floor1kW; same as R6',
      'inference':'5000 paired crossed resamples of customers and Monday-calendar weeks; identical draws across all cells and branches; conditional on fitted seeds',
      'scope':'previously scored external groups/periods; cannot be claimed new independent confirmation or component-level causal bypass isolation',
      'output_sha256':outputs,'checks':list(CHECKS)}
    savej(dest/f'{phase}_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('checks','output_sha256')}),flush=True)

def score(family,branch):
    sourceaudit=loadj(OUT/branch/'source_audit.json')
    bind(CODE,sourceaudit['scorer_sha256'],'same independent scorer for source and external')
    p,reports,complete=forecast_gate(family)
    for phase in ('calibration','confirmation'):score_phase(family,branch,phase,p,reports)

def report():
    branches=[]
    for folder in sorted(OUT.iterdir()):
        if folder.is_dir() and (folder/'calibration_audit.json').exists() and (folder/'confirmation_audit.json').exists():
            s=loadj(folder/'source_audit.json');phases={p:loadj(folder/f'{p}_audit.json') for p in ('calibration','confirmation')}
            branches.append({'branch':folder.name,'source':s,'phases':phases})
    check(bool(branches),'at least one fully audited raw branch')
    target=ROOT/'results/r7_article_v1';target.mkdir(exist_ok=True)
    summary={'status':'VERIFIED_COMPLETE_RAW_NETWORK_EXPLORATORY_FAMILIES','scorer_sha256':sha(CODE),'branches':branches,
             'interpretation_boundary':'architecture, capacity and normalization changed jointly from R6. Raw-network outcomes do not identify a causal effect of bypass removal. All external outcomes are post-review exploratory; source-only selection cannot turn already known external data into independent confirmation.',
             'all_conditions_retained':True,'bootstrap':'5000paired customer×Monday-calendar-week resamples; fixed seed13320261002; macro3seed losses, not ensemble forecasts'}
    savej(target/'raw_network_verified.json',summary)
    lines=['# Independently verified R7 raw-network extension','',summary['interpretation_boundary'],'',
      'Scores use complete adjacent observed endpoint pairs, fixed past support scales with 1 kW floor, equal-customer macros and averages of three fitted-seed losses. No producer, fitter or producer metric is imported. All 60 checkpoints and 120 external prediction files per normalization branch are retained.','']
    for b in branches:
        s=b['source'];lines+=['## '+b['branch'],'',f"The fixed source normalizer is {s['normalizer']} ({s['center_kw']:.9g} kW center, {s['scale_kw']:.9g} kW scale). Independently reproduced 104-row source NMAE ranges from {s['nmae_range'][0]:.6f} to {s['nmae_range'][1]:.6f}; original R6 source NMAE ranges from {s['r6_independent_source_nmae_range'][0]:.6f} to {s['r6_independent_source_nmae_range'][1]:.6f}. Source naive NMAEs: "+', '.join(f"{r['reference']} {r['source_validation_nmae']:.6f}" for r in s['source_naive_references'])+'.','',
          f"All 60 selected-checkpoint traces use the earliest minimum among ten source checkpoints; tensor schemas contain exactly 53,472 core parameters, +1 mixed/attention or +3 gated/stride, plus the common nontrainable center/scale buffers. Every learned tensor reaches the forecast. Maximum source metric float64-versus-reported-float32 discrepancy is {s['max_metric_float64_vs_float32_disagreement']:.3g}; independently implemented NumPy float64 versus Torch float32 source forecast discrepancy is {s['max_numpy64_vs_torch32_source_forecast_difference_kw']:.3g} kW.",'']
        for phase in ('calibration','confirmation'):
            a=b['phases'][phase];r=a['primary'][0]
            lines+=[f"{phase}: all {a['customers']} customers, {a['dates']} dates, {a['complete_pairs']:,} complete pairs and {a['monday_week_blocks']} Monday-week blocks. The primary NMAE interaction (max fine−coarse)−(average fine−coarse) is {r['three_seed_mean_difference']:.8f}, conditional 95% paired-bootstrap interval [{r['conditional_ci95_low']:.8f}, {r['conditional_ci95_high']:.8f}]. Independent saved-weight forecast reconstruction discrepancy is at most {a['max_external_saved_weight_forward_difference_kw']:.3g} kW. All three original naive forecast arrays and same-mask scores reproduce.",'']
        lines+=['The branch directory contains full cell/seed/customer/date scores, source validation reproduction, source competence comparisons, four fine-access diagnostics, all-history diagnostics, pointwise/Bonferroni/centered max-t intervals, seed contrasts and retained paired-week sufficient statistics/bootstrap draws. Simultaneous intervals apply separately to the four fine diagnostics within each metric/phase; they do not adjust cross-metric or cross-normalizer searching.','']
    path=target/'raw_network_verified.md'
    if path.exists():raise FileExistsError(path)
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':summary['status'],'branches':[b['branch'] for b in branches],'report':str(path)}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('source','score','report'))
    p.add_argument('--family',default=str(ROOT/'results/r7_raw_access_v1'));p.add_argument('--branch',default='mean_std')
    args=p.parse_args();OUT.mkdir(exist_ok=True)
    if args.stage=='report':report()
    else:globals()[args.stage](Path(args.family),args.branch)

if __name__=='__main__':main()
