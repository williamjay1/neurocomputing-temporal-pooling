"""R6 frozen-learner reuse: source access diagnostic and gated2014 inference.

No fitting or scoring entry point. Confirmation truth is never loaded here.
The source-only diagnostic perturbs only final-block complete-coverage pairs.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path('D:/MLWork/Neurocomputing_TailTransfer')
LEARNER=ROOT/'scripts/130_r6_access_experiment_v1.py'
LEARNER_SHA='6b434b9d806e12fef001fc7fd8ceaaa73ac5c8badf992de2e947f7d8d2f581e4'
OLD=ROOT/'results/r6_access_v1'
OUT=ROOT/'results/r6_confirmation_v1'
DATA=ROOT/'datasets/upgrade5_uci_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''): h.update(block)
    return h.hexdigest()


def write_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def load_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def load_learner():
    if sha(LEARNER)!=LEARNER_SHA: raise ValueError('frozen130 learner mismatch')
    spec=importlib.util.spec_from_file_location('r6_frozen130',LEARNER)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def protocol():
    return {
        'stage':'BEFORE_CONFIRMATION_INFERENCE_SOURCE_DIAGNOSTIC_ONLY_UNTIL_ROOT_GATE',
        'learner_sha256':LEARNER_SHA,
        'learner_protocol_sha256':sha(OLD/'proposed_protocol.json'),
        'learner_preflight_sha256':sha(OLD/'preflight.json'),
        'source_cache_manifest_sha256':sha(OLD/'source_cache/manifest.json'),
        'data_manifest_sha256':sha(DATA/'dataset_manifest_v1.json'),
        'confirmation_scope':'all32 fixed confirmation IDs, columns48:80;2014-01-01 through2014-12-31;365 daily origins',
        'confirmation_model_policy':'all36 existing source-only checkpoints, all six adapters/two histories/three seeds; no new fitting',
        'advancement_policy':'execute full confirmation after source completion and independent engineering/data checks; independent of calibration directions or scores',
        'confirmation_support':'2013-12-25 through2013-12-31 observed mean floored1kW per customer',
        'history':'same672 slots ending strictly before each origin; future outcomes not loaded for prediction',
        'prediction_clipping':'none, identical to frozen130/R5',
        'quality_references':'all three persistence/daily_naive/weekly_naive fine-history no-fit references from frozen130 protocol',
        'prediction_schema':{'customer_ids':'[32] string','forecast_dates':'[365] datetime64[D]',
            'origin_indices':'[365] int64','predictions_kw':'[32,365,96] float32','support_scales_kw':'[32] float64'},
        'root_gate_status':'ROOT_APPROVED_CONFIRMATION_INFERENCE',
        'root_gate_sha_fields':['script_sha256','confirmation_protocol_sha256','learner_script_sha256',
            'learner_protocol_sha256','source_completion_sha256','data_manifest_sha256'],
        'root_gate_bool_fields':['near_neighbor_gate_complete','independent_engineering_and_data_checks_passed',
            'all36_confirmation_precommitted_independent_of_calibration_scores'],
        'source_diagnostic':{
            'scope':'all36 selected checkpoints on all104 fixed source-held-time windows',
            'location':'final block immediately before its input adapter; earlier blocks, normalization and baseline path frozen',
            'eligible_pairs':'both reversed-history coverage entries exactly1; other residual pairs unchanged',
            'transform':'m=(r0+r1)/2,d=(r0-r1)/2;replace d by s*alpha*d; preserve m',
            'alpha':[0,.5,1,2],'sign':[-1,1],'reference':'unmodified forward alpha1/sign+1',
            'reported_metrics':['physical forecast RMS change kW','physical forecast mean-absolute change kW'],
            'no_source_label_accuracy_or_checkpoint_selection':True,
            'scope_boundary':'local trained final-residual access response, not whole-network/input invariance, Adam dynamics, or London ranking explanation',
            'local_sign_invariance':'max-repeat,avg-repeat,absolute-Haar invariant to complete pair sign swaps; overlapping max-blur not generally invariant'},
        'confirmation_truth_scoring':'external independent scorer only after all36 forecasts and three references complete'
    }


def preflight():
    OUT.mkdir(parents=True,exist_ok=True);base=load_learner()
    if (OUT/'confirmation_protocol.json').exists(): raise FileExistsError('131 protocol already sealed')
    rng=np.random.default_rng(7411);raw=rng.uniform(10,20,(3,672)).astype(np.float32)
    observed=np.ones_like(raw,np.uint8);observed[0,:4]=0;raw[observed==0]=np.nan
    poisoned=raw.copy();poisoned[observed==0]=-1e20
    torch=base.setup(20261001)
    for history in base.HISTORIES:
        x,m=base.history_representation(raw,observed,history)
        xp,mp=base.history_representation(poisoned,observed,history)
        assert np.array_equal(x,xp) and np.array_equal(m,mp)
        for adapter in base.ADAPTERS:
            net=base.model_factory(adapter)
            with torch.no_grad(): p=net(torch.from_numpy(x),torch.from_numpy(m))
            assert p.shape==(3,96) and torch.isfinite(p).all()
    # Metadata-only calendar/roster checks; neither2013 nor2014 numeric arrays loaded.
    manifest=load_json(DATA/'dataset_manifest_v1.json');ids=np.load(DATA/'client_ids.npy',allow_pickle=False).tolist()
    cols=[ids.index(i) for i in manifest['customer_groups']['confirmation']]
    labels=np.load(DATA/'provider_labels.npy',allow_pickle=False)
    start=int(np.searchsorted(labels,np.datetime64('2014-01-01')))
    end=int(np.searchsorted(labels,np.datetime64('2015-01-01')))
    origins=np.arange(start,end,96,dtype=np.int64)
    assert cols==list(range(48,80)) and len(origins)==365
    assert labels[start-672]==np.datetime64('2013-12-25T00:00')
    write_json(OUT/'confirmation_protocol.json',protocol())
    write_json(OUT/'preflight.json',{'status':'PASS','script_sha256':sha(__file__),
        'confirmation_protocol_sha256':sha(OUT/'confirmation_protocol.json'),'synthetic_only':True,
        'no_calibration_or_confirmation_numeric_values_read':True,'confirmation_columns':cols,
        'confirmation_start_index':start,'confirmation_end_exclusive':end,'confirmation_origin_count':len(origins),
        'learner_model_and_history_reuse_pass':True,'missing_value_poison_pass':True})
    print(json.dumps({'preflight':'PASS','script_sha256':sha(__file__),
        'confirmation_protocol_sha256':sha(OUT/'confirmation_protocol.json')}),flush=True)


def verify_inputs():
    base=load_learner();proof=load_json(OUT/'preflight.json');plan=load_json(OUT/'confirmation_protocol.json')
    if proof.get('status')!='PASS' or proof['script_sha256']!=sha(__file__): raise ValueError('131 preflight/code mismatch')
    if proof['confirmation_protocol_sha256']!=sha(OUT/'confirmation_protocol.json'): raise ValueError('131 protocol changed')
    for key,path in [('learner_protocol_sha256',OLD/'proposed_protocol.json'),
                     ('learner_preflight_sha256',OLD/'preflight.json'),
                     ('source_cache_manifest_sha256',OLD/'source_cache/manifest.json'),
                     ('data_manifest_sha256',DATA/'dataset_manifest_v1.json')]:
        if plan[key]!=sha(path): raise ValueError('131 input mismatch '+key)
    base.verify_source_cache()
    completion=load_json(OLD/'source_fit_completion.json')
    if completion['status']!='ALL36_SOURCE_FITS_COMPLETE_CALIBRATION_GATED' or len(completion['jobs'])!=36:
        raise ValueError('source family incomplete')
    if completion['script_sha256']!=LEARNER_SHA: raise ValueError('source completion wrong code')
    if completion['protocol_sha256']!=sha(OLD/'proposed_protocol.json'): raise ValueError('source completion wrong protocol')
    expected={(a,h,s) for a in base.ADAPTERS for h in base.HISTORIES for s in base.SEEDS}
    actual=set()
    for row in completion['jobs']:
        key=(row['adapter'],row['history'],row['seed']);actual.add(key)
        p=Path(row['report_path'])
        if sha(p)!=row['report_sha256']: raise ValueError('source report changed')
        report=load_json(p);cp=p.with_suffix('.npz')
        if report['checkpoint_sha256']!=sha(cp) or row['checkpoint_sha256']!=sha(cp): raise ValueError('source checkpoint changed')
    if actual!=expected: raise ValueError('source job keys mismatch')
    return base,completion


def restore(base,adapter,history,seed):
    torch=base.setup(seed);net=base.model_factory(adapter)
    cp=OLD/'source_fits'/f'{adapter}_{history}_seed{seed}.npz'
    with np.load(cp,allow_pickle=False) as z: net.load_state_dict({k:torch.from_numpy(z[k].copy()) for k in z.files})
    net.eval();return torch,net,cp


def source_diagnostic():
    base,completion=verify_inputs();folder=OLD/'source_cache';rows=[]
    for seed in base.SEEDS:
        for adapter in base.ADAPTERS:
            for history in base.HISTORIES:
                x=np.load(folder/f'validation_{history}_decoded.npy')
                mask=np.load(folder/f'validation_{history}_coverage.npy')
                torch,net,cp=restore(base,adapter,history,seed)
                tx,tm=torch.from_numpy(x),torch.from_numpy(mask)
                live=(tm.flip(-1).reshape(len(x),1,336,2)==1).all(-1)
                with torch.no_grad(): reference=net(tx,tm).numpy().astype(np.float64)
                for alpha in (0,.5,1,2):
                    for sign in (-1,1):
                        def perturb(module,args):
                            residual=args[0]
                            if alpha==1 and sign==1: return None
                            pair=residual.reshape(len(x),1,336,2)
                            if alpha==1 and sign==-1: changed=pair.flip(-1)
                            else:
                                m=pair.mean(-1);d=(pair[...,0]-pair[...,1])/2
                                changed=torch.stack((m+sign*alpha*d,m-sign*alpha*d),-1)
                            return (torch.where(live.unsqueeze(-1),changed,pair).flatten(-2),)
                        hook=net.blocks[-1].pooling_layer.register_forward_pre_hook(perturb)
                        try:
                            with torch.no_grad(): forecast=net(tx,tm).numpy().astype(np.float64)
                        finally: hook.remove()
                        response=forecast-reference
                        ma=float(np.abs(response).mean());rms=float(np.sqrt(np.square(response).mean()))
                        if alpha==1 and sign==-1 and adapter in ('max_repeat','avg_repeat','haar_absolute'):
                            if not np.array_equal(forecast,reference): raise AssertionError('local sign invariance failed')
                        rows.append({'adapter':adapter,'history':history,'seed':seed,'alpha':alpha,'sign':sign,
                            'forecast_mean_absolute_change_kw':ma,'forecast_rms_change_kw':rms,
                            'source_windows':len(x),'eligible_complete_residual_pairs':int(live.sum()),
                            'checkpoint_sha256':sha(cp)})
    if len(rows)!=288: raise ValueError('diagnostic family incomplete')
    import csv
    p=OUT/'source_residual_diagnostic.csv'
    if p.exists(): raise FileExistsError(p)
    with p.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(OUT/'source_residual_diagnostic.json',{'status':'ALL36_SOURCE_ONLY_DIAGNOSTICS_COMPLETE',
        'rows':len(rows),'csv_sha256':sha(p),'script_sha256':sha(__file__),
        'confirmation_protocol_sha256':sha(OUT/'confirmation_protocol.json'),
        'source_completion_sha256':sha(OLD/'source_fit_completion.json'),
        'confirmation_or_calibration_inputs_read':False,'source_labels_scored':False,
        'local_pair_sign_invariance_checks':'PASS for max-repeat/avg-repeat/absolute-Haar; not assumed for other adapters'})
    print(json.dumps({'source_diagnostic':'PASS','rows':len(rows),'csv_sha256':sha(p)}),flush=True)


def predict():
    gate=load_json(OUT/'root_confirmation_gate.json')
    if gate.get('status')!='ROOT_APPROVED_CONFIRMATION_INFERENCE': raise PermissionError('confirmation gate closed')
    for key in ('near_neighbor_gate_complete','independent_engineering_and_data_checks_passed',
                'all36_confirmation_precommitted_independent_of_calibration_scores'):
        if gate.get(key) is not True: raise PermissionError('confirmation gate missing '+key)
    for key,path in [('script_sha256',Path(__file__)),('confirmation_protocol_sha256',OUT/'confirmation_protocol.json'),
        ('learner_script_sha256',LEARNER),('learner_protocol_sha256',OLD/'proposed_protocol.json'),
        ('source_completion_sha256',OLD/'source_fit_completion.json'),('data_manifest_sha256',DATA/'dataset_manifest_v1.json')]:
        if gate.get(key)!=sha(path): raise ValueError('confirmation gate binding mismatch '+key)
    base,completion=verify_inputs();plan=load_json(OUT/'confirmation_protocol.json')
    manifest=load_json(DATA/'dataset_manifest_v1.json');ids=np.load(DATA/'client_ids.npy',allow_pickle=False).tolist()
    cs=manifest['customer_groups']['confirmation'];cols=[ids.index(i) for i in cs]
    labels=np.load(DATA/'provider_labels.npy',allow_pickle=False)
    start=int(np.searchsorted(labels,np.datetime64('2014-01-01')));end=len(labels)
    origins=np.arange(start,end,96,dtype=np.int64)
    if cols!=list(range(48,80)) or len(origins)!=365: raise ValueError('confirmation scope mismatch')
    # First numerical access to confirmation/support histories occurs only here.
    values=np.load(DATA/'panel_kw.npy',mmap_mode='r');valid=np.load(DATA/'valid.npy',mmap_mode='r')
    support=[]
    for c in cols:
        y=np.asarray(values[start-672:start,c]);ob=np.asarray(valid[start-672:start,c],bool)
        support.append(max(float(y[ob].mean(dtype=np.float64)) if ob.any() else 0.,1.))
    raw=np.stack([values[o-672:o,c] for c in cols for o in origins]).astype(np.float32)
    observed=np.stack([valid[o-672:o,c] for c in cols for o in origins]).astype(np.uint8)
    prepared={h:base.history_representation(raw,observed,h) for h in base.HISTORIES}
    dest=OUT/'predictions';dest.mkdir(exist_ok=True);reports=[]
    def save(tag,pred,report):
        p=dest/f'{tag}.npz'
        if p.exists(): raise FileExistsError(p)
        if pred.shape!=(32,365,96) or not np.isfinite(pred).all(): raise ValueError('invalid forecast')
        np.savez_compressed(p,customer_ids=np.asarray(cs),forecast_dates=labels[origins].astype('datetime64[D]'),
            origin_indices=origins,predictions_kw=pred.astype(np.float32),support_scales_kw=np.asarray(support,np.float64))
        report.update({'forecast_sha256':sha(p),'script_sha256':sha(__file__),
            'confirmation_protocol_sha256':sha(OUT/'confirmation_protocol.json'),
            'root_gate_sha256':sha(OUT/'root_confirmation_gate.json'),'shape':list(pred.shape),
            'confirmation_truth_scored_by_predictor':False})
        write_json(dest/f'{tag}.json',report)
        return {'path':str(p),'sha256':sha(p),'report_sha256':sha(dest/f'{tag}.json')}
    for seed in base.SEEDS:
        for adapter in base.ADAPTERS:
            for history in base.HISTORIES:
                torch,net,cp=restore(base,adapter,history,seed);x,mask=prepared[history]
                wall=time.perf_counter();cpu=time.process_time();parts=[]
                with torch.no_grad():
                    for b in range(0,len(x),128):
                        parts.append(net(torch.from_numpy(x[b:b+128]),torch.from_numpy(mask[b:b+128])).numpy().copy())
                pred=np.concatenate(parts).reshape(32,365,96)
                report={'status':'COMPLETE_CONFIRMATION_FORECAST_UNSCORED','adapter':adapter,'history':history,'seed':seed,
                    'parameter_count':base.PARAMETERS,'prediction_active_parameter_count':base.PREDICTION_ACTIVE_PARAMETERS,
                    'checkpoint_sha256':sha(cp),'source_fit_report_sha256':sha(cp.with_suffix('.json')),
                    'prediction_wall_seconds':time.perf_counter()-wall,'prediction_cpu_seconds':time.process_time()-cpu}
                reports.append(save(f'{adapter}_{history}_seed{seed}',pred,report))
    references=[];fine_x=prepared['fine'][0]
    for name in base.QUALITY_REFERENCES:
        ref=(np.repeat(fine_x[:,-1:],96,axis=1) if name=='persistence' else
             fine_x[:,-96:].copy() if name=='daily_naive' else fine_x[:,:96].copy()).reshape(32,365,96)
        record=save(f'reference_{name}',ref,{'status':'COMPLETE_CONFIRMATION_QUALITY_REFERENCE_UNSCORED','reference':name,'fit_count':0})
        record['name']=name;references.append(record)
    write_json(OUT/'prediction_completion.json',{'status':'ALL36_CONFIRMATION_FORECASTS_COMPLETE_UNSCORED',
        'forecasts':reports,'quality_references':references,'formal_fit_count':36,'quality_reference_count':3,
        'script_sha256':sha(__file__),'learner_script_sha256':LEARNER_SHA,
        'confirmation_protocol_sha256':sha(OUT/'confirmation_protocol.json'),
        'source_completion_sha256':sha(OLD/'source_fit_completion.json'),
        'root_gate_sha256':sha(OUT/'root_confirmation_gate.json'),'data_manifest_sha256':sha(DATA/'dataset_manifest_v1.json'),
        'confirmation_truth_scored_by_predictor':False,'all36_conditions_retained':True})
    print(json.dumps({'confirmation_forecasts':'COMPLETE_UNSCORED','formal':len(reports),'references':len(references)}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('preflight','source-diagnostic','predict'));a=p.parse_args()
    {'preflight':preflight,'source-diagnostic':source_diagnostic,'predict':predict}[a.stage]()


if __name__=='__main__': main()
