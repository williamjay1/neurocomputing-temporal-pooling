"""Post-review exploratory whole-input access experiment on the frozen UCI panel.

New outputs only. No per-window normalization statistic, residual, or last-value
forecast bypass enters this two-hidden-layer MLP. Producer does not score
external future outcomes. All 60 fits precede every external forecast.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, math, sys, time
import numpy as np

sys.dont_write_bytecode=True
ROOT=Path('D:/MLWork/Neurocomputing_TailTransfer')
OUT=ROOT/'results/r7_raw_access_v1'
OLD=ROOT/'results/r6_access_v1'
DATA=ROOT/'datasets/upgrade5_uci_v1'
SEEDS=(20261001,20261002,20261003)
CORE=('direct','max_repeat','avg_repeat','haar_absolute','haar_signed','max_blur_repeat')
EXTRA=('mixed','gated','attention','stride_conv')
ADAPTERS=CORE+EXTRA
HISTORIES=('fine','coarse')
BASE_PARAMETERS=53472
HISTORY,HORIZON,UPDATES,BATCH=672,96,2000,64
EXTRA_PARAMETERS={'mixed':1,'gated':3,'attention':1,'stride_conv':3}
OLD_SCRIPT=ROOT/'scripts/130_r6_access_experiment_v1.py'
OLD_SHA='6b434b9d806e12fef001fc7fd8ceaaa73ac5c8badf992de2e947f7d8d2f581e4'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def source_module():
    if sha(OLD_SCRIPT)!=OLD_SHA:raise ValueError('Frozen source preparation changed')
    spec=importlib.util.spec_from_file_location('r7_old_preparation',OLD_SCRIPT)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def setup(seed):
    import torch
    torch.set_num_threads(2);torch.manual_seed(seed)
    return torch


def model_factory(adapter,center,scale):
    import torch
    from torch import nn
    from torch.nn import functional as F
    class Access(nn.Module):
        def __init__(self,kind):
            super().__init__();self.kind=kind
            self.register_buffer('blur',torch.tensor([.25,.5,.25]).reshape(1,1,3),persistent=False)
            if kind in ('mixed','attention'):
                self.theta=nn.Parameter(torch.tensor(0. if kind=='mixed' else 1.))
            if kind in ('gated','stride_conv'):
                self.coefficients=nn.Parameter(torch.tensor([0.,0.,0.] if kind=='gated' else [.5,.5,0.]))
        def forward(self,x):
            if self.kind=='direct':return x
            if self.kind=='max_blur_repeat':
                dense=F.max_pool1d(F.pad(x[:,None,:],(0,1),mode='replicate'),2,stride=1)
                return F.conv1d(F.pad(dense,(1,1),mode='replicate'),self.blur,stride=2).squeeze(1).repeat_interleave(2,-1)
            p=x.reshape(len(x),HISTORY//2,2)
            mean=p.mean(-1);maximum=p.max(-1).values
            if self.kind=='max_repeat':z=maximum
            elif self.kind=='avg_repeat':z=mean
            elif self.kind in ('haar_absolute','haar_signed'):
                c=(p[...,0]+p[...,1])/math.sqrt(2);a=(p[...,0]-p[...,1])/math.sqrt(2)
                return torch.stack((c,a.abs() if self.kind=='haar_absolute' else a),-1).flatten(-2)
            elif self.kind=='mixed':
                g=self.theta.sigmoid();z=g*maximum+(1-g)*mean
            elif self.kind=='gated':
                q=self.coefficients;g=(q[0]*p[...,0]+q[1]*p[...,1]+q[2]).sigmoid()
                z=g*maximum+(1-g)*mean
            elif self.kind=='attention':
                z=((p*self.theta).softmax(-1)*p).sum(-1)
            elif self.kind=='stride_conv':
                q=self.coefficients;z=q[0]*p[...,0]+q[1]*p[...,1]+q[2]
            else:raise ValueError(self.kind)
            return z.repeat_interleave(2,-1)
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            # Shared affine tensors are generated before additional pool parameters.
            self.layers=nn.Sequential(nn.Linear(HISTORY,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,HORIZON))
            self.adapter=Access(adapter)
            self.register_buffer('center',torch.tensor(center,dtype=torch.float32))
            self.register_buffer('scale',torch.tensor(scale,dtype=torch.float32))
            if adapter in ('haar_absolute','haar_signed'):
                with torch.no_grad():
                    w=self.layers[0].weight.reshape(64,HISTORY//2,2).clone()
                    self.layers[0].weight.copy_(torch.stack(((w[...,0]+w[...,1])/math.sqrt(2),
                        (w[...,0]-w[...,1])/math.sqrt(2)),-1).flatten(-2))
        def forward(self,x):
            return self.layers(self.adapter((x-self.center)/self.scale))*self.scale+self.center
    return Model()


def source_arrays():
    folder=OLD/'source_cache';m=load(folder/'manifest.json')
    for n,r in m['files'].items():
        if sha(folder/n)!=r['sha256']:raise ValueError('Changed source array '+n)
    arrays={n[:-4]:np.load(folder/n,allow_pickle=False) for n in m['files'] if n.endswith('.npy')}
    values=arrays['train_values'];observed=arrays['train_observed'].astype(bool)
    measured=values[observed].astype(np.float64)
    center=float(measured.mean());scale=max(float(measured.std()),1.)
    weights={}
    for split in ('train','validation'):
        v=arrays[split+'_values'];ob=arrays[split+'_observed'].astype(bool)
        weights[split]=np.maximum(np.where(ob,v,0.).sum(1,dtype=np.float64)/ob.sum(1).clip(1),1.).astype(np.float32)
    return arrays,weights,center,scale


def preflight():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('R7 protocol already exists')
    arrays,weights,center,scale=source_arrays()
    tests=[];initial={}
    for a in ADAPTERS:
        torch=setup(SEEDS[0]);net=model_factory(a,center,scale)
        count=sum(p.numel() for p in net.parameters())
        if count!=BASE_PARAMETERS+EXTRA_PARAMETERS.get(a,0):raise ValueError('Parameter count '+a)
        net.eval()
        with torch.no_grad():
            y=net(torch.from_numpy(arrays['validation_fine_decoded'][:8].copy())).numpy()
        if not np.isfinite(y).all():raise ValueError('Nonfinite preflight')
        initial[a]=y
        tests.append({'adapter':a,'registered_and_forecast_active_parameters':count})
    difference=float(np.max(np.abs(initial['direct']-initial['haar_signed'])))
    if difference>0.001:raise ValueError('Initial function not matched')
    protocol={
        'stage':'POST_REVIEW_EXPLORATORY_ALREADY_SCORED_EXTERNAL_CUSTOMERS_AND_PERIODS',
        'script_sha256':sha(__file__),'frozen_preparation_sha256':OLD_SHA,
        'core_adapters':CORE,'known_nonmatched_parameter_references':EXTRA,
        'histories':HISTORIES,'seeds':SEEDS,'formal_fit_count':60,
        'source_training_rows':len(arrays['train_values']),'source_validation_rows':len(arrays['validation_values']),
        'source_cache_manifest_sha256':sha(OLD/'source_cache/manifest.json'),
        'fixed_global_center_kw':center,'fixed_global_scale_kw':scale,
        'normalization_source':'all observed fine-history entries in source training rows only; shared constant affine',
        'hidden_layers':[64,64],'horizon':96,'history':672,
        'loss':'complete-pair endpoint MAE divided by common source-row observed-history mean floored at1kW',
        'optimizer':'Adam','learning_rate':.001,'updates':2000,'batch':64,'gradient_norm_cap':1.,
        'schedule':None,'validation':'104 source rows; every200updates; earliest minimum common-weight NMAE',
        'per_window_mean_std_as_inputs':False,'last_value_residual_forecast_bypass':False,
        'clipping':False,'mask_feature':False,
        'core_initial_compensated_signed_max_abs_prediction_difference_kw':difference,
        'tests':tests,'batch_draws':'frozen R6 source draws per seed; identical all20 conditions',
        'external_protocol':{'calibration':'all16 customers,184 days July–December2013,Jun24–30 prior scale',
          'confirmation':'all32 customers,365 days2014,Dec25–31 2013 prior scale',
          'all_six_core_and_four_reference_results_retained':True,
          'future_truth_not_scored_in_producer':True,'not_new_independent_validation':True},
        'boundary':'architecture changes normalization and bypasses jointly; difference cannot causally isolate bypass removal',
    }
    save(OUT/'protocol.json',protocol)
    print(json.dumps({'status':'PREFLIGHT_PASS','parameters':tests,'center':center,'scale':scale,'initial_difference_kw':difference}),flush=True)


def train():
    protocol=load(OUT/'protocol.json')
    if protocol['script_sha256']!=sha(__file__):raise ValueError('Code changed after freeze')
    arrays,weights,center,scale=source_arrays()
    if center!=protocol['fixed_global_center_kw'] or scale!=protocol['fixed_global_scale_kw']:raise ValueError('Scaler changed')
    dest=OUT/'fits';dest.mkdir(exist_ok=True);records=[]
    for seed in SEEDS:
        draws=arrays[f'draws_seed{seed}']
        for adapter in ADAPTERS:
            for history in HISTORIES:
                tag=f'{adapter}_{history}_seed{seed}';cp=dest/f'{tag}.npz';rp=dest/f'{tag}.json'
                if cp.exists() or rp.exists():raise FileExistsError(tag)
                torch=setup(seed);net=model_factory(adapter,center,scale)
                opt=torch.optim.Adam(net.parameters(),lr=.001)
                x=torch.from_numpy(arrays[f'train_{history}_decoded'].copy())
                vx=torch.from_numpy(arrays[f'validation_{history}_decoded'].copy())
                y=torch.from_numpy(np.nan_to_num(arrays['train_labels']).copy())
                vy=torch.from_numpy(np.nan_to_num(arrays['validation_labels']).copy())
                pair=arrays['train_label_observed'].reshape(-1,48,2).all(-1)
                vpair=arrays['validation_label_observed'].reshape(-1,48,2).all(-1)
                mask=torch.from_numpy(np.repeat(pair,2,-1).astype(np.float32))
                vmask=torch.from_numpy(np.repeat(vpair,2,-1).astype(np.float32))
                weight=torch.from_numpy(weights['train']);vweight=torch.from_numpy(weights['validation'])
                best=float('inf');beststep=None;beststate=None;validations=[];wall=time.perf_counter();cpu=time.process_time()
                for step,indices in enumerate(draws,1):
                    net.train();opt.zero_grad(set_to_none=True)
                    pred=net(x[indices]);loss=(((pred-y[indices]).abs()*mask[indices]).sum(-1)/mask[indices].sum(-1).clamp(min=1)/weight[indices]).mean()
                    if not torch.isfinite(loss):raise ValueError('Nonfinite fit '+tag)
                    loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
                    if step%200==0:
                        net.eval()
                        with torch.no_grad():
                            val=float((((net(vx)-vy).abs()*vmask).sum(-1)/vmask.sum(-1).clamp(min=1)/vweight).mean())
                        validations.append({'update':step,'common_weight_source_nmae':val})
                        if val<best:
                            best=val;beststep=step;beststate={k:v.detach().cpu().numpy().copy() for k,v in net.state_dict().items()}
                np.savez_compressed(cp,**beststate)
                report={'status':'SOURCE_FIT_COMPLETE_EXTERNAL_UNSCORED_BY_PRODUCER','adapter':adapter,'history':history,'seed':seed,
                    'registered_and_forecast_active_parameters':BASE_PARAMETERS+EXTRA_PARAMETERS.get(adapter,0),
                    'selected_update':beststep,'source_validation_nmae':best,'validation_trace':validations,
                    'checkpoint_sha256':sha(cp),'script_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),
                    'source_cache_manifest_sha256':sha(OLD/'source_cache/manifest.json'),
                    'fit_plus_validation_wall_seconds':time.perf_counter()-wall,'fit_plus_validation_cpu_seconds':time.process_time()-cpu,
                    'additional_source_row_scaler_as_network_input':False,'external_truth_scored':False}
                save(rp,report);records.append({'checkpoint':str(cp.relative_to(ROOT)),'sha256':sha(cp),'report_sha256':sha(rp)})
                print(json.dumps({'fit':tag,'step':beststep,'source_loss':best,'seconds':report['fit_plus_validation_wall_seconds']}),flush=True)
                del x,vx,y,vy,mask,vmask,weight,vweight,opt,net
    save(OUT/'training_completion.json',{'status':'ALL60_SOURCE_FITS_COMPLETE','formal_fit_count':60,'fits':records,
        'script_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),'source_cache_manifest_sha256':sha(OLD/'source_cache/manifest.json'),
        'external_scoring_in_this_producer':False,'post_review_exploratory':True})


def forecast():
    p=load(OUT/'protocol.json');complete=load(OUT/'training_completion.json')
    if p['script_sha256']!=sha(__file__) or complete['formal_fit_count']!=60:raise ValueError('Unfrozen family')
    for r in complete['fits']:
        if sha(ROOT/r['checkpoint'])!=r['sha256']:raise ValueError('Checkpoint changed')
    base=source_module();m=load(DATA/'dataset_manifest_v1.json')
    ids=np.load(DATA/'client_ids.npy',allow_pickle=False).tolist()
    values=np.load(DATA/'panel_kw.npy',mmap_mode='r',allow_pickle=False)
    valid=np.load(DATA/'valid.npy',mmap_mode='r',allow_pickle=False)
    labels=np.load(DATA/'provider_labels.npy',allow_pickle=False)
    def index(date):return int(np.searchsorted(labels,np.datetime64(date)))
    allrecords={}
    for phase,start,end,group,supportstart,supportend in (
        ('calibration','2013-07-01','2014-01-01','calibration','2013-06-24','2013-07-01'),
        ('confirmation','2014-01-01','2015-01-01','confirmation','2013-12-25','2014-01-01')):
        clients=m['customer_groups'][group];cols=[ids.index(c) for c in clients]
        origins=np.arange(index(start),index(end),96,dtype=np.int64)
        raw=np.stack([values[o-672:o,c] for c in cols for o in origins]).astype(np.float32)
        observed=np.stack([valid[o-672:o,c] for c in cols for o in origins]).astype(np.uint8)
        prepared={h:base.history_representation(raw,observed,h)[0] for h in HISTORIES}
        scales=[]
        for c in cols:
            vv=values[index(supportstart):index(supportend),c];ob=valid[index(supportstart):index(supportend),c].astype(bool)
            scales.append(max(float(vv[ob].mean(dtype=np.float64)),1.))
        dest=OUT/phase;dest.mkdir(exist_ok=True);reports=[]
        for seed in SEEDS:
            for adapter in ADAPTERS:
                for history in HISTORIES:
                    torch=setup(seed);net=model_factory(adapter,p['fixed_global_center_kw'],p['fixed_global_scale_kw'])
                    tag=f'{adapter}_{history}_seed{seed}';cp=OUT/'fits'/f'{tag}.npz'
                    with np.load(cp,allow_pickle=False) as z:net.load_state_dict({k:torch.from_numpy(z[k].copy()) for k in z.files})
                    net.eval();parts=[];wall=time.perf_counter()
                    with torch.no_grad():
                        for b in range(0,len(raw),256):parts.append(net(torch.from_numpy(prepared[history][b:b+256].copy())).numpy())
                    pred=np.concatenate(parts).reshape(len(cols),len(origins),96).astype(np.float32)
                    if not np.isfinite(pred).all():raise ValueError('Nonfinite prediction')
                    path=dest/f'{tag}.npz'
                    if path.exists():raise FileExistsError(path)
                    np.savez_compressed(path,customer_ids=np.asarray(clients),forecast_dates=labels[origins].astype('datetime64[D]'),
                        origin_indices=origins,predictions_kw=pred,support_scales_kw=np.asarray(scales,np.float64))
                    r={'adapter':adapter,'history':history,'seed':seed,'shape':list(pred.shape),
                        'checkpoint_sha256':sha(cp),'forecast_sha256':sha(path),'protocol_sha256':sha(OUT/'protocol.json'),
                        'script_sha256':sha(__file__),'forecast_and_write_wall_seconds':time.perf_counter()-wall,
                        'future_labels_scored_in_producer':False,'post_review_exploratory':True}
                    save(path.with_suffix('.json'),r);reports.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'report_sha256':sha(path.with_suffix('.json'))})
        allrecords[phase]=reports
        print(json.dumps({'phase':phase,'forecasts':len(reports),'customers':len(cols),'days':len(origins)}),flush=True)
    save(OUT/'forecast_completion.json',{'status':'ALL120_EXTERNAL_FORECASTS_COMPLETE_UNSCORED_BY_PRODUCER',
        'phases':allrecords,'formal_fit_count':60,'protocol_sha256':sha(OUT/'protocol.json'),
        'training_completion_sha256':sha(OUT/'training_completion.json'),'script_sha256':sha(__file__),
        'external_data_previously_scored':True,'not_independent_confirmation':True})


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('preflight','train','forecast'));a=p.parse_args()
    {'preflight':preflight,'train':train,'forecast':forecast}[a.stage]()


if __name__=='__main__':main()
