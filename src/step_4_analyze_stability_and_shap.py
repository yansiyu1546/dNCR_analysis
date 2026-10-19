"""Prespecified stability and selection-rule sensitivity analyses; no external retuning."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import copy,itertools,json,argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel,delayed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_auc_score,confusion_matrix,brier_score_loss
from step_1_modeling_utils import *
from step_3_evaluate_model_performance import savefig,COLORS

ROOT=Path(__file__).resolve().parents[1]

def extra_run(m,r,b,y):
    res=boruta_once(b['members'][m]['train'],y,b['columns'],seed(b['context_id'],20,r))
    return [dict(imputation=m+1,boruta_run=r+1,imputation_seed=b['members'][m]['imputation_seed'],boruta_seed=seed(b['context_id'],20,r),variable=c,**v) for c,v in res.items()]

def subset(bundle,rule):
    b=copy.copy(bundle)
    if rule=='Original 5/5 intersection':
        v=pd.read_csv(ROOT/'results'/f"boruta_{b['tag']}.csv")
        f=v[v.boruta_run==1].groupby('variable').confirmed.mean()
        b['selected']=priority([c for c in b['columns'] if f[c]>=1])
    else:
        cutoff=float(rule.split('%')[0])/100
        b['selected']=priority([c for c in b['columns'] if b['frequencies'][c]>=cutoff])
    return b

def sensitivity_rule(rule,name,y,X,ex):
    oof=np.zeros(len(y));thresholds=np.zeros(len(y));featurelists=[];final_saved=None
    for f in [1,2,3,4,5,0]:
        tag=f'outer{f}' if f else 'final'
        saved=joblib.load(ROOT/'cache'/f'fit_{tag}_{name}.joblib')
        ids=[];rr=[]
        for j in range(1,4):
            b=subset(joblib.load(ROOT/'cache'/f'{tag}_inner{j}.joblib'),rule)
            _,raw,_,_=fit_members(b,y[b['train_idx']],name,saved['params'])
            ids.extend(b['valid_idx']);rr.extend(raw)
        ids=np.array(ids);rr=np.array(rr)
        cal=fit_calibrator(y[ids],rr,name);pp=calibrated(cal,rr,name);t=choose_threshold(y[ids],pp)
        b=subset(joblib.load(ROOT/'cache'/f'{tag}.joblib'),rule)
        fitted,p,_,_=fit_members(b,y[b['train_idx']],name,saved['params'])
        if f:
            oof[b['valid_idx']]=calibrated(cal,p,name);thresholds[b['valid_idx']]=t
            featurelists.append(dict(Rule=rule,Fold=f,Variables=';'.join(b['selected'])))
        else:
            final_saved=dict(name=name,preprocess=b,fitted=fitted,calibrator=cal,threshold=t)
    external=predict_saved(final_saved,ex);rows=[]
    for cohort,yy,p,t in [('Internal OOF',y,oof,thresholds),('Temporal external',ex.dNCR.to_numpy(),external,final_saved['threshold'])]:
        tn,fp,fn,tp=confusion_matrix(yy,p>=t,labels=[0,1]).ravel()
        rows.append(dict(Rule=rule,Model=name,Cohort=cohort,AUC=roc_auc_score(yy,p),Sensitivity=tp/(tp+fn),Specificity=tn/(tn+fp),Brier=brier_score_loss(yy,p),Final_n_features=len(final_saved['preprocess']['selected']),Final_features=';'.join(final_saved['preprocess']['selected']),Final_threshold=final_saved['threshold'],Note='Main-analysis hyperparameters held fixed; imputation, selection-rule application, calibration and thresholds refitted within folds.'))
    return rows,featurelists

def shap_values(lock,b):
    name=lock['best_model'];artifact=joblib.load(ROOT/'models'/f'{name}_pipeline.joblib')
    selected=b['selected'];columns=b['columns']
    # Explain the five-member calibrated probability function on complete, post-imputation values.
    values=np.mean([m['train'] for m in b['members']],axis=0)
    x=values[:,[columns.index(c) for c in selected]]
    def fun(z):
        full=np.tile(np.median(values,axis=0),(len(z),1))
        for j,c in enumerate(selected):full[:,columns.index(c)]=z[:,j]
        scores=[]
        for member in artifact['fitted']:
            enc,_=matrix(full,columns,selected)
            if member['scaler'] is not None:enc=member['scaler'].transform(enc)
            scores.append(model_score(member['model'],enc,name))
        return calibrated(artifact['calibrator'],np.mean(scores,axis=0),name)
    rng=np.random.default_rng(seed(81));background=x[rng.choice(len(x),min(60,len(x)),replace=False)]
    labels=[c.replace('_',' ').replace('MOCA Score','MoCA score') for c in selected]
    exp=shap.Explainer(fun,shap.maskers.Independent(background,max_samples=60),algorithm='permutation',feature_names=labels,seed=seed(82))
    sv=exp(x,max_evals=max(2*len(selected)+1,(2*len(selected)+1)*20),batch_size=100,silent=True)
    np.savez(ROOT/'results'/'SHAP_values.npz',values=sv.values,base_values=sv.base_values,data=x,features=np.array(selected))
    err=float(np.max(np.abs(sv.base_values+sv.values.sum(axis=1)-fun(x))))
    imp=np.mean(abs(sv.values),axis=0);order=np.argsort(imp)
    pd.DataFrame({'Variable':selected,'Mean_absolute_SHAP':imp}).sort_values('Mean_absolute_SHAP',ascending=False).to_csv(ROOT/'results'/'shap_importance.csv',index=False)
    fig,ax=plt.subplots(figsize=(8.5,max(4.5,.52*len(selected)+1.5)))
    ax.barh(np.array(labels)[order],imp[order],color=COLORS[name],height=.67)
    ax.set(xlabel='Mean absolute SHAP value (probability)',title=f'(a) Variable importance ranking: {name}')
    ax.grid(axis='x',alpha=.18);fig.tight_layout();savefig(fig,'最优模型_SHAP_变量重要性')
    plt.figure(figsize=(9,max(5,.52*len(selected)+1.5)))
    shap.summary_plot(sv.values,x,feature_names=labels,max_display=len(selected),show=False,plot_size=None)
    plt.xlabel('SHAP value (effect on predicted probability)');plt.title(f'(b) SHAP beeswarm: {name}',fontsize=16,pad=15);plt.tight_layout();savefig(plt.gcf(),'最优模型_SHAP_蜂群图')
    jsonwrite(ROOT/'results'/'shap_notes.json',dict(model=name,rows=len(x),background_rows=len(background),method='Permutation SHAP for calibrated five-member probability ensemble',reference='Development data only, mean of five completed predictor matrices',scope='Post-imputation predictor explanation; does not attribute uncertainty to missingness/imputation. Complete surrogate inputs can differ from saved raw-data predictions on missing selected predictors.',max_additivity_error=err,seed=seed(82),causal_interpretation=False))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--jobs',type=int,default=6);parser.add_argument('--stability-only',action='store_true');args=parser.parse_args()
    tr=pd.read_excel(ROOT/'data'/'training.xlsx');ex=pd.read_excel(ROOT/'data'/'external.xlsx');y=tr.dNCR.to_numpy()
    b=joblib.load(ROOT/'cache'/'final.joblib')
    p=ROOT/'results'/'boruta_50_runs.csv'
    if p.exists():v=pd.read_csv(p)
    else:
        original=pd.read_csv(ROOT/'results'/'boruta_final.csv').drop(columns='context')
        outputs=Parallel(n_jobs=args.jobs)(delayed(extra_run)(m,r,b,y) for m in range(5) for r in range(2,10))
        v=pd.concat([original,pd.DataFrame([x for rr in outputs for x in rr])],ignore_index=True)
        v.to_csv(p,index=False)
    table=v.pivot_table(index=['imputation','boruta_run'],columns='variable',values='confirmed').astype(int)
    frequency=table.mean().sort_values(ascending=False)
    mainfreq=pd.read_csv(ROOT/'results'/'boruta_final.csv').groupby('variable').confirmed.mean()
    stability=pd.DataFrame({'Variable':frequency.index,'Frequency_50':frequency.values,'Primary_frequency_10':mainfreq.reindex(frequency.index).values,'Final_selected':[x in b['selected'] for x in frequency.index]})
    foldfreq=[]
    for f in range(1,6):
        bb=joblib.load(ROOT/'cache'/f'outer{f}.joblib');foldfreq.append({c:int(c in bb['selected']) for c in b['columns']})
    folds=pd.DataFrame(foldfreq,index=range(1,6))
    stability['Outer_fold_selection_frequency']=folds.mean().reindex(stability.Variable).to_numpy()
    stability.to_csv(ROOT/'results'/'feature_stability_summary.csv',index=False)
    jac=[]
    for i,j in itertools.combinations(range(len(table)),2):
        a=table.iloc[i].to_numpy(bool);c=table.iloc[j].to_numpy(bool)
        ji=(a&c).sum()/(a|c).sum() if (a|c).sum() else 1.
        mi,si=table.index[i];mj,sj=table.index[j]
        jac.append(dict(Imputation1=int(mi),Seed_run1=int(si),Imputation2=int(mj),Seed_run2=int(sj),Jaccard=float(ji),Comparison='Within imputation: Boruta seed' if mi==mj else ('Across imputations: same Boruta seed' if si==sj else 'Both differ')))
    jd=pd.DataFrame(jac);jd.to_csv(ROOT/'results'/'jaccard_pairs.csv',index=False)
    jd.groupby('Comparison').Jaccard.agg(['mean','median','min','max','count']).to_csv(ROOT/'results'/'jaccard_summary.csv')
    shown=stability.head(22).iloc[::-1];fig,ax=plt.subplots(figsize=(10,9))
    ax.barh(shown.Variable.str.replace('_',' '),shown.Frequency_50,color=[COLORS['LR'] if x else '#B8C5D3' for x in shown.Final_selected])
    ax.axvline(.8,color='#A33A39',ls='--',lw=1.5);ax.set(xlim=(0,1.06),xlabel='Confirmed selection frequency (5 imputations x 10 seeds)',title='Boruta selection stability');ax.grid(axis='x',alpha=.18);fig.tight_layout();savefig(fig,'补充_Boruta入选频率')
    keep=stability[stability.Frequency_50>0].Variable.tolist()
    fig,ax=plt.subplots(figsize=(11,max(5,.35*len(keep)+2)))
    im=ax.imshow(table[keep].to_numpy().T,aspect='auto',cmap='Blues',vmin=0,vmax=1,interpolation='nearest')
    ax.set_yticks(range(len(keep)),[x.replace('_',' ') for x in keep]);ax.set_xticks([4.5,14.5,24.5,34.5,44.5],[f'Imputation {i}' for i in range(1,6)]);ax.set(title='Confirmed Boruta features across 50 runs',xlabel='Ten fixed Boruta seeds within each imputation');fig.colorbar(im,ax=ax,ticks=[0,1],label='Confirmed');fig.tight_layout();savefig(fig,'补充_Boruta稳定性热图')
    if args.stability_only:return
    lock=json.loads((ROOT/'results'/'model_lock.json').read_text())
    rules=['60% consensus','100% consensus','Original 5/5 intersection']
    outputs=Parallel(n_jobs=min(4,args.jobs))(delayed(sensitivity_rule)(rule,lock['best_model'],y,tr[b['columns']],ex) for rule in rules)
    rr=[r for result,_ in outputs for r in result];ff=[r for _,f in outputs for r in f]
    pd.DataFrame(rr).to_csv(ROOT/'results'/'selection_rule_sensitivity.csv',index=False)
    pd.DataFrame(ff).to_csv(ROOT/'results'/'sensitivity_fold_features.csv',index=False)
    shap_values(lock,b)
    print('Stability, selection-rule sensitivity and SHAP completed.',flush=True)

if __name__=='__main__':main()
