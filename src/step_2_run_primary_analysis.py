"""Primary analysis. Run from this directory: python step_2_run_primary_analysis.py --jobs 6"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse, hashlib, json, shutil, time, itertools, importlib.metadata, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, ParameterSampler
from sklearn.metrics import roc_auc_score,brier_score_loss
from step_1_modeling_utils import *

ROOT=Path(__file__).resolve().parents[1]

def evaluate_trial(name,params,paths,y):
    scores=[];ids=[];raw=[];warnings_count=0
    for path in paths:
        b=joblib.load(path)
        _,p,_,con=fit_members(b,y[b['train_idx']],name,params)
        scores.append(roc_auc_score(y[b['valid_idx']],p))
        ids.extend(b['valid_idx'].tolist());raw.extend(p.tolist());warnings_count+=sum(con)
    return dict(params=params,mean_auc=float(np.mean(scores)),fold_auc=scores,ids=ids,raw=raw,convergence_warnings=warnings_count)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',type=int,default=6);ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    for d in ['data','cache','models','results','imputations','figures','reports']:(ROOT/d).mkdir(exist_ok=True)
    tr=pd.read_excel(ROOT/'data'/'training.xlsx',sheet_name=0)
    ex=pd.read_excel(ROOT/'data'/'external.xlsx',sheet_name=0)
    assert set(tr.dNCR)=={0,1} and set(ex.dNCR)=={0,1}
    assert list(tr.columns)==list(ex.columns)
    y=tr.dNCR.to_numpy(int)
    cols=[c for c in tr if c!='dNCR' and c not in CONFIG['exclude']]
    X=tr[cols]
    print(f'Candidate predictors ({len(cols)}): {cols}',flush=True)
    fingerprint=hashlib.sha256((tr.to_csv(index=False)+ex.to_csv(index=False)+json.dumps(CONFIG,sort_keys=True)).encode()).hexdigest()
    marker=ROOT/'cache'/'fingerprint.txt'
    if marker.exists() and marker.read_text()!=fingerprint:raise RuntimeError('Input/config changed. Move the old cache directory before a fresh analysis.')
    marker.write_text(fingerprint)
    jsonwrite(ROOT/'results'/'config.json',CONFIG)
    jsonwrite(ROOT/'results'/'input_audit.json',dict(training_n=len(tr),external_n=len(ex),training_events=int(y.sum()),external_events=int(ex.dNCR.sum()),duplicates_train=int(tr.duplicated().sum()),duplicates_external=int(ex.duplicated().sum()),input_sha256=fingerprint,excluded=CONFIG['exclude'],outcome='dNCR=1 treated as event; semantic confirmation pending',notes='ASA retained as a candidate predictor; training and external coding harmonized to 0/1 by the user. H kept as source label; units not supplied. Original values retained. Continuous variables with >10 unique values winsorized at fold-training 1st/99th percentiles.'))
    missing=pd.DataFrame({'Variable':tr.columns,'Training_missing':tr.isna().sum().values,'External_missing':ex.isna().sum().values})
    missing.to_csv(ROOT/'results'/'missingness.csv',index=False)
    flags=[]
    for cohort,d in [('Training',tr),('External',ex)]:
        for c,mask,reason in [('Na',(d.Na<100)|(d.Na>180),'Verify value/unit; retained raw, fold-wise winsorization only'),('H',d.H<50,'Identity/unit unclear; verify source'),('Neostigmine',d.Neostigmine>5,'Verify dose/unit; no guessed correction')]:
            for idx in d.index[mask]:flags.append(dict(Cohort=cohort,Excel_row=int(idx+2),Variable=c,Value=float(d.loc[idx,c]),Action=reason))
    pd.DataFrame(flags).to_csv(ROOT/'results'/'data_queries.csv',index=False)
    assignments=[];contexts=[];outer=[]
    cv=StratifiedKFold(CONFIG['outer_folds'],shuffle=True,random_state=CONFIG['seed'])
    for f,(a,b) in enumerate(cv.split(X,y),1):
        tag=f'outer{f}';contexts.append((a,b,tag,f*100));inner=[]
        for j,(ia,ib) in enumerate(StratifiedKFold(CONFIG['inner_folds'],shuffle=True,random_state=seed(f,40)).split(a,y[a]),1):
            itag=f'{tag}_inner{j}';contexts.append((a[ia],a[ib],itag,f*100+j));inner.append(itag)
        outer.append(dict(fold=f,train=a,valid=b,tag=tag,inner=inner))
    for j,(a,b) in enumerate(StratifiedKFold(CONFIG['inner_folds'],shuffle=True,random_state=seed(999,40)).split(X,y),1):contexts.append((a,b,f'final_inner{j}',900+j))
    contexts.append((np.arange(len(tr)),np.array([],dtype=int),'final',999))
    for a,b,tag,cid in contexts:
        for role,ix in [('fit',a),('validation',b)]:
            assignments.extend(dict(context=tag,role=role,training_row=int(i+2),row_id=f'TR{i+1:04d}',context_seed_id=cid) for i in ix)
    pd.DataFrame(assignments).to_csv(ROOT/'results'/'fold_assignments.csv',index=False)
    print('Preparing 24 isolated fold pipelines; external outcomes are not used.',flush=True)
    Parallel(n_jobs=args.jobs)(delayed(prepare)(X,y,a,b,tag,cid,ROOT) for a,b,tag,cid in contexts)
    if args.prepare_only:return
    candidates={n:list(ParameterSampler(SPACES[n],n_iter=CONFIG['trials_per_model'],random_state=seed(50,NAMES.index(n)))) for n in NAMES}
    jsonwrite(ROOT/'results'/'search_candidates.json',candidates)
    oof={n:np.full(len(tr),np.nan) for n in NAMES};oof_t={n:np.full(len(tr),np.nan) for n in NAMES};folds=np.zeros(len(tr),int)
    final_info={};tuning=[];foldsummary=[]
    loops=outer+[dict(fold=0,train=np.arange(len(tr)),valid=np.array([],int),tag='final',inner=[f'final_inner{j}' for j in range(1,4)])]
    for loop in loops:
        b=joblib.load(ROOT/'cache'/f"{loop['tag']}.joblib")
        for name in NAMES:
            result_path=ROOT/'cache'/f"fit_{loop['tag']}_{name}.joblib"
            if result_path.exists():saved=joblib.load(result_path)
            else:
                paths=[ROOT/'cache'/f'{tag}.joblib' for tag in loop['inner']]
                trials=Parallel(n_jobs=args.jobs)(delayed(evaluate_trial)(name,p,paths,y) for p in candidates[name])
                # First candidate is the deterministic tie-break; do not examine external metrics.
                best=max(range(len(trials)),key=lambda k:trials[k]['mean_auc'])
                selected=trials[best];ids=np.asarray(selected['ids']);raw=np.asarray(selected['raw'])
                calibrator=fit_calibrator(y[ids],raw,name)
                cp=calibrated(calibrator,raw,name);threshold=choose_threshold(y[ids],cp)
                fitted,p,ptr,con=fit_members(b,y[b['train_idx']],name,selected['params'])
                saved=dict(params=selected['params'],calibrator=calibrator,threshold=threshold,inner_ids=ids,inner_raw=raw,inner_calibrated=cp,trials=trials,fitted=fitted,valid_probability=calibrated(calibrator,p,name) if len(p) else p,train_probability=calibrated(calibrator,ptr,name),convergence_warnings=con,best_index=best)
                joblib.dump(saved,result_path,compress=3)
            for k,t in enumerate(saved['trials']):tuning.append(dict(context=loop['tag'],model=name,candidate=k+1,mean_auc=t['mean_auc'],fold_auc=json.dumps(t['fold_auc']),parameters=json.dumps(t['params']),convergence_warnings=t['convergence_warnings'],selected=k==saved['best_index']))
            if loop['fold']:
                ix=loop['valid'];oof[name][ix]=saved['valid_probability'];oof_t[name][ix]=saved['threshold'];folds[ix]=loop['fold']
                auc=roc_auc_score(y[ix],saved['valid_probability'])
                foldsummary.append(dict(Fold=loop['fold'],Model=name,AUC=auc,Threshold=saved['threshold'],Features=';'.join(b['selected'])))
                print(f"FIT {loop['tag']} {name}: AUC={auc:.3f}, threshold={saved['threshold']:.3f}",flush=True)
            else:
                artifact=dict(name=name,preprocess=b,fitted=saved['fitted'],calibrator=saved['calibrator'],threshold=saved['threshold'],parameters=saved['params'],config=CONFIG,input_sha256=fingerprint)
                # Only a training-based decision is made at this point.
                modelpath=ROOT/'models'/f'{name}_pipeline.joblib'
                temporary=modelpath.with_suffix('.tmp')
                # Standard protocol 5 avoids compressed joblib array aliasing failures.
                with temporary.open('wb') as f:pickle.dump(artifact,f,protocol=5)
                check=joblib.load(temporary)
                assert len(check['fitted'])==CONFIG['imputations']
                temporary.replace(modelpath)
                final_info[name]=dict(params=saved['params'],threshold=saved['threshold'],inner_auc=saved['trials'][saved['best_index']]['mean_auc'])
                for m,member in enumerate(saved['fitted'],1):
                    mdl=member['model']
                    if name=='ANN':np.savez(ROOT/'models'/f'ANN_weights_imputation{m}.npz',**{f'W{i}':v for i,v in enumerate(mdl.coefs_)},**{f'b{i}':v for i,v in enumerate(mdl.intercepts_)})
                    if name=='XGB':mdl.save_model(ROOT/'models'/f'XGB_imputation{m}.json')
                    if name=='LR':pd.DataFrame({'Feature':member['feature_names']+['Intercept'],'Standardized_coefficient':list(mdl.coef_[0])+list(mdl.intercept_)}).to_csv(ROOT/'models'/f'LR_coefficients_imputation{m}.csv',index=False)
                print(f"FINAL {name}: inner AUC={final_info[name]['inner_auc']:.3f}, threshold={saved['threshold']:.3f}",flush=True)
    for name in NAMES:assert np.isfinite(oof[name]).all()
    # Model family is selected with development data only, before external prediction.
    ranking=sorted(NAMES,key=lambda n:(-roc_auc_score(y,oof[n]),brier_score_loss(y,oof[n]),NAMES.index(n)))
    best=ranking[0]
    jsonwrite(ROOT/'results'/'model_lock.json',dict(best_model=best,selection='Highest pooled outer-fold OOF AUC, then lower Brier, then fixed name order',ranking=ranking,final=final_info,features=joblib.load(ROOT/'cache'/'final.joblib')['selected'],external_outcomes_used_for_selection=False))
    pd.DataFrame(tuning).to_csv(ROOT/'results'/'hyperparameter_trials.csv',index=False)
    pd.DataFrame(foldsummary).to_csv(ROOT/'results'/'outer_fold_metrics.csv',index=False)
    pred=pd.DataFrame({'row_id':[f'TR{i+1:04d}' for i in range(len(tr))],'Excel_row':np.arange(len(tr))+2,'dNCR':y,'outer_fold':folds})
    apparent=pd.DataFrame({'row_id':pred.row_id,'dNCR':y})
    ep=pd.DataFrame({'row_id':[f'EX{i+1:04d}' for i in range(len(ex))],'Excel_row':np.arange(len(ex))+2,'dNCR':ex.dNCR})
    for name in NAMES:
        pred[name]=oof[name];pred[name+'_threshold']=oof_t[name]
        artifact=joblib.load(ROOT/'models'/f'{name}_pipeline.joblib')
        ep[name]=predict_saved(artifact,ex);ep[name+'_threshold']=artifact['threshold']
        saved=joblib.load(ROOT/'cache'/f'fit_final_{name}.joblib')
        apparent[name]=saved['train_probability'];apparent[name+'_threshold']=artifact['threshold']
    pred.to_csv(ROOT/'results'/'internal_oof_predictions.csv',index=False)
    ep.to_csv(ROOT/'results'/'external_predictions.csv',index=False)
    apparent.to_csv(ROOT/'results'/'apparent_training_predictions.csv',index=False)
    final=joblib.load(ROOT/'cache'/'final.joblib')
    for m,member in enumerate(final['members'],1):
        a=pd.DataFrame(member['train'],columns=cols);a.insert(0,'row_id',pred.row_id);a['dNCR']=y
        a.to_csv(ROOT/'imputations'/f'training_imputation{m}.csv',index=False)
        z=impute_new(member['imputer'],clip_values(ex[cols],final['lower'],final['upper']),member['transform_seed'])
        a=pd.DataFrame(z,columns=cols);a.insert(0,'row_id',ep.row_id);a['dNCR']=ex.dNCR.to_numpy()
        a.to_csv(ROOT/'imputations'/f'external_imputation{m}.csv',index=False)
    # Save each fold's actual imputed train/validation data, with membership only.
    for _,_,tag,_ in contexts:
        b=joblib.load(ROOT/'cache'/f'{tag}.joblib')
        for m,member in enumerate(b['members'],1):
            for role,key,ix in [('train','train',b['train_idx']),('validation','valid',b['valid_idx'])]:
                if not len(ix):continue
                a=pd.DataFrame(member[key],columns=cols);a.insert(0,'row_id',[f'TR{i+1:04d}' for i in ix]);a['dNCR']=y[ix]
                folder=ROOT/'imputations'/'folds';folder.mkdir(exist_ok=True)
                a.to_csv(folder/f'{tag}_{role}_imputation{m}.csv',index=False)
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','xgboost','Boruta','shap','statsmodels','matplotlib','joblib','openpyxl','python-docx']}
    (ROOT/'requirements.txt').write_text('\n'.join(f'{p}=={v}' for p,v in versions.items())+'\n')
    jsonwrite(ROOT/'results'/'environment.json',dict(python=os.sys.version,packages=versions))
    print('LOCKED BEST MODEL',best,'FEATURES',final['selected'],flush=True)

if __name__=='__main__':main()
