"""Leakage-aware multiple-imputation and Boruta prediction utilities."""
from __future__ import annotations
import copy, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.exceptions import ConvergenceWarning
from boruta import BorutaPy
from xgboost import XGBClassifier

CONFIG = dict(seed=20261017, outer_folds=5, inner_folds=3, imputations=5,
              boruta_seeds=2, boruta_iterations=50, boruta_trees=128,
              boruta_depth=5, boruta_percentile=100, stability_cutoff=0.8,
              trials_per_model=12, target_sensitivity=0.85,
              winsor_quantiles=[0.01,0.99], exclude=[],
              categorical=['Surgery_Type'],
              pair_priority=[['Operation_Time','Anesthesia_Time'],['MOCA_Score','Education']],
              bootstrap=2000, extra_boruta_seeds=10)
NAMES = ['LR','RF','XGB','ANN','SVM']
SPACES = {
 'LR':dict(C=[0.001,0.01,0.1,1,10,100],l1_ratio=[0.,0.5,1.],class_weight=[None,'balanced']),
 'RF':dict(n_estimators=[200,400],max_depth=[3,5,8,None],min_samples_leaf=[3,5,10],max_features=['sqrt',0.8],class_weight=[None,'balanced']),
 'XGB':dict(n_estimators=[100,200,300],max_depth=[1,2,3],learning_rate=[0.01,0.03,0.05],subsample=[0.7,0.9],colsample_bytree=[0.7,0.9],min_child_weight=[3,5,10],reg_alpha=[0.,0.5,1.],reg_lambda=[2.,5.,10.],scale_pos_weight=[1.,1.5,2.]),
 'ANN':dict(hidden_layer_sizes=[(8,),(16,),(16,8)],activation=['relu','tanh'],alpha=[0.001,0.01,0.1,1.],learning_rate_init=[0.001,0.003],batch_size=[16,32],class_weight=[None,'balanced']),
 'SVM':dict(C=[0.01,0.1,1,10,100],gamma=['scale',0.001,0.01,0.1],class_weight=[None,'balanced'])}

def seed(*parts):
    return int(np.random.SeedSequence([CONFIG['seed'],*map(int,parts)]).generate_state(1)[0])

def jsonwrite(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)),encoding='utf-8')

def encoded(a,cols):
    out=[]; groups=[]; labels=[]
    for j,c in enumerate(cols):
        if c=='Surgery_Type':
            for v in [1,2,3]:
                out.append((a[:,j]==v).astype(float));groups.append(c);labels.append(f'Surgery_Type_{v}')
        else:
            out.append(a[:,j]);groups.append(c);labels.append(c)
    return np.asarray(out).T,groups,labels

def clip_values(frame,lo,hi):
    return np.clip(frame.to_numpy(dtype=float),lo,hi)

def impute_new(imp,a,random_seed):
    # Copy so predictions never mutate a saved imputer's posterior RNG.
    obj=copy.deepcopy(imp)
    obj.random_state_=np.random.RandomState(random_seed)
    return obj.transform(a)

def boruta_once(a,y,cols,random_seed):
    z,groups,labels=encoded(a,cols)
    forest=RandomForestClassifier(n_estimators=CONFIG['boruta_trees'],max_depth=CONFIG['boruta_depth'],min_samples_leaf=3,class_weight='balanced',n_jobs=1,random_state=random_seed)
    selector=BorutaPy(forest,n_estimators=CONFIG['boruta_trees'],perc=CONFIG['boruta_percentile'],alpha=0.05,two_step=True,max_iter=CONFIG['boruta_iterations'],random_state=random_seed,verbose=0)
    selector.fit(z,y)
    return {c:dict(confirmed=bool(np.any(selector.support_[np.array(groups)==c])),tentative=bool(np.any(selector.support_weak_[np.array(groups)==c]))) for c in cols}

def priority(selected):
    selected=list(selected)
    for keep,drop in CONFIG['pair_priority']:
        if keep in selected and drop in selected:selected.remove(drop)
    return selected

def prepare(X,y,train_idx,valid_idx,tag,context_id,outdir):
    """Everything in this function is fitted only on train_idx."""
    import joblib
    outdir=Path(outdir);cache=outdir/'cache'/f'{tag}.joblib'
    if cache.exists():return str(cache)
    tr=X.iloc[train_idx];va=X.iloc[valid_idx];cols=list(X.columns)
    lo=tr.min().to_numpy(float);hi=tr.max().to_numpy(float)
    for j,c in enumerate(cols):
        if tr[c].nunique()>10 and c not in CONFIG['categorical']:
            lo[j],hi[j]=tr[c].quantile(CONFIG['winsor_quantiles']).to_numpy()
    a=clip_values(tr,lo,hi);b=clip_values(va,lo,hi)
    members=[];votes=[]
    for m in range(CONFIG['imputations']):
        rs=seed(context_id,10,m)
        imp=IterativeImputer(max_iter=10,sample_posterior=True,random_state=rs,min_value=lo,max_value=hi,skip_complete=False,initial_strategy='median')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',ConvergenceWarning)
            imputed=imp.fit_transform(a)
        new=impute_new(imp,b,seed(context_id,11,m)) if len(b) else np.empty((0,len(cols)))
        members.append(dict(imputer=imp,train=imputed,valid=new,imputation_seed=rs,transform_seed=seed(context_id,11,m)))
        for r in range(CONFIG['boruta_seeds']):
            bs=seed(context_id,20,r)
            res=boruta_once(imputed,y[train_idx],cols,bs)
            for c,v in res.items():votes.append(dict(context=tag,imputation=m+1,boruta_run=r+1,imputation_seed=rs,boruta_seed=bs,variable=c,**v))
    vote=pd.DataFrame(votes)
    freq=vote.groupby('variable').confirmed.mean().reindex(cols)
    selected=priority(freq[freq>=CONFIG['stability_cutoff']].index.tolist())
    bundle=dict(tag=tag,context_id=context_id,columns=cols,lower=lo,upper=hi,members=members,selected=selected,frequencies=freq,train_idx=np.asarray(train_idx),valid_idx=np.asarray(valid_idx))
    joblib.dump(bundle,cache,compress=3)
    vote.to_csv(outdir/'results'/f'boruta_{tag}.csv',index=False)
    print(f'PREP {tag}: n={len(train_idx)}, selected={selected}',flush=True)
    return str(cache)

def matrix(a,cols,selected):
    z,groups,labels=encoded(a,cols)
    mask=np.array([g in selected for g in groups])
    return z[:,mask],[l for l,k in zip(labels,mask) if k]

def model_factory(name,p,rs):
    p=p.copy();cw=p.pop('class_weight',None)
    if name=='LR':return LogisticRegression(solver='saga',max_iter=3000,tol=1e-4,random_state=rs,class_weight=cw,**p),None
    if name=='RF':return RandomForestClassifier(n_jobs=1,random_state=rs,class_weight=cw,**p),None
    if name=='XGB':return XGBClassifier(objective='binary:logistic',eval_metric='logloss',tree_method='hist',n_jobs=1,random_state=rs,**p),None
    if name=='ANN':return MLPClassifier(solver='adam',max_iter=600,early_stopping=True,validation_fraction=0.15,n_iter_no_change=30,tol=1e-4,random_state=rs,**p),cw
    if name=='SVM':return SVC(kernel='rbf',probability=False,random_state=rs,class_weight=cw,**p),None
    raise ValueError(name)

def model_score(model,z,name):
    if name=='SVM' and not isinstance(model,DummyClassifier):return model.decision_function(z)
    return model.predict_proba(z)[:,1]

def fit_members(bundle,y,name,params):
    models=[];preds=[];trainpreds=[];conv=[]
    for m,member in enumerate(bundle['members']):
        a,labels=matrix(member['train'],bundle['columns'],bundle['selected'])
        b,_=matrix(member['valid'],bundle['columns'],bundle['selected'])
        scaler=None
        if a.shape[1]==0:
            a=np.zeros((len(a),1));b=np.zeros((len(b),1));model=DummyClassifier(strategy='prior');cw=None
        else:
            if name in ['LR','ANN','SVM']:
                scaler=StandardScaler().fit(a);a=scaler.transform(a);b=scaler.transform(b) if len(b) else b
            model,cw=model_factory(name,params,seed(bundle['context_id'],30,NAMES.index(name),m))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            weight=compute_sample_weight('balanced',y) if cw=='balanced' else None
            if weight is None:model.fit(a,y)
            else:model.fit(a,y,sample_weight=weight)
        conv.append(any(issubclass(w.category,ConvergenceWarning) for w in caught))
        models.append(dict(model=model,scaler=scaler,feature_names=labels))
        preds.append(model_score(model,b,name) if len(b) else np.array([]))
        trainpreds.append(model_score(model,a,name))
    return models,np.mean(preds,axis=0),np.mean(trainpreds,axis=0),conv

def calibration_input(raw,name):
    a=np.asarray(raw)
    return a.reshape(-1,1) if name=='SVM' else logit(np.clip(a,1e-6,1-1e-6)).reshape(-1,1)

def fit_calibrator(y,raw,name):
    c=LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000)
    c.fit(calibration_input(raw,name),y)
    return c

def calibrated(c,raw,name):return c.predict_proba(calibration_input(raw,name))[:,1]

def choose_threshold(y,p,target=None):
    target=CONFIG['target_sensitivity'] if target is None else target
    fpr,tpr,thresholds=roc_curve(y,p,drop_intermediate=False)
    valid=np.flatnonzero((tpr>=target)&np.isfinite(thresholds))
    return float(thresholds[valid[0]])

def predict_saved(artifact,frame):
    bundle=artifact['preprocess'];name=artifact['name']
    a=clip_values(frame[bundle['columns']],bundle['lower'],bundle['upper'])
    pp=[]
    for m,(member,fitted) in enumerate(zip(bundle['members'],artifact['fitted'])):
        values=impute_new(member['imputer'],a,member['transform_seed'])
        z,_=matrix(values,bundle['columns'],bundle['selected'])
        if z.shape[1]==0:z=np.zeros((len(z),1))
        if fitted['scaler'] is not None:z=fitted['scaler'].transform(z)
        pp.append(model_score(fitted['model'],z,name))
    return calibrated(artifact['calibrator'],np.mean(pp,axis=0),name)

def predict_complete_selected(artifact,frame):
    """Equivalent inference when every selected predictor is observed.

    Auxiliary imputation columns are unnecessary in this specific case.
    """
    b=artifact['preprocess'];selected=b['selected'];cols=b['columns'];name=artifact['name']
    if frame[selected].isna().any().any():raise ValueError('Selected predictors contain missing values; use the full candidate-input pipeline.')
    a=np.zeros((len(frame),len(cols)),dtype=float)
    for c in selected:a[:,cols.index(c)]=frame[c].to_numpy(float)
    a=np.clip(a,b['lower'],b['upper']);z,_=matrix(a,cols,selected);pp=[]
    for member in artifact['fitted']:
        zz=member['scaler'].transform(z) if member['scaler'] is not None else z
        pp.append(model_score(member['model'],zz,name))
    return calibrated(artifact['calibrator'],np.mean(pp,axis=0),name)
