"""Create transparent evaluation tables and publication figures from saved predictions."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import itertools,json,io,os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logit
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,precision_recall_curve,confusion_matrix,brier_score_loss
from sklearn.calibration import calibration_curve
from step_1_modeling_utils import CONFIG,NAMES,SPACES,seed,jsonwrite

ROOT=Path(__file__).resolve().parents[1]
COLORS=dict(LR='#2166AC',RF='#009E73',XGB='#D55E00',ANN='#8B5AA9',SVM='#B38B22')
LABELS=dict(LR='LR',RF='RF',XGB='XGBoost',ANN='ANN',SVM='SVM')
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],'font.size':12,'axes.titlesize':16,'axes.labelsize':14,'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':10,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.1,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})

def ci_auc(y,p,reps=None):
    rng=np.random.default_rng(seed(60));pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    au=[]
    for _ in range(CONFIG['bootstrap'] if reps is None else reps):
        ix=np.r_[rng.choice(pos,len(pos)),rng.choice(neg,len(neg))]
        au.append(roc_auc_score(y[ix],p[ix]))
    return np.quantile(au,[.025,.975])

def metrics(y,p,t):
    yy=(p>=t).astype(int);tn,fp,fn,tp=confusion_matrix(y,yy,labels=[0,1]).ravel()
    sensitivity=tp/(tp+fn);specificity=tn/(tn+fp)
    se=proportion_confint(tp,tp+fn,alpha=.05,method='wilson');sp=proportion_confint(tn,tn+fp,alpha=.05,method='wilson')
    lo,hi=ci_auc(y,p)
    return dict(AUC=roc_auc_score(y,p),AUC_low=lo,AUC_high=hi,Accuracy=(tp+tn)/len(y),Sensitivity=sensitivity,Sensitivity_low=se[0],Sensitivity_high=se[1],Specificity=specificity,Specificity_low=sp[0],Specificity_high=sp[1],PPV=tp/(tp+fp) if tp+fp else 0,NPV=tn/(tn+fn) if tn+fn else 0,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,Brier=brier_score_loss(y,p),AP=average_precision_score(y,p),TP=int(tp),FN=int(fn),TN=int(tn),FP=int(fp),Threshold=float(np.mean(t)),Threshold_min=float(np.min(t)),Threshold_max=float(np.max(t)),N=len(y),Events=int(sum(y)))

def calibration_stats(y,p):
    z=logit(np.clip(p,1e-6,1-1e-6))
    model=sm.GLM(y,sm.add_constant(z),family=sm.families.Binomial()).fit()
    ci=model.conf_int()
    citl=sm.GLM(y,np.ones((len(y),1)),offset=z,family=sm.families.Binomial()).fit()
    g=pd.qcut(p,q=10,duplicates='drop')
    d=pd.DataFrame(dict(y=y,p=p,g=g)).groupby('g',observed=True).agg(N=('y','size'),O=('y','sum'),E=('p','sum'))
    den=d.E*(1-d.E/d.N)
    hl=float(np.sum((d.O-d.E)**2/den));df=len(d)-2
    return dict(Intercept=float(model.params[0]),Intercept_low=float(ci[0,0]),Intercept_high=float(ci[0,1]),Slope=float(model.params[1]),Slope_low=float(ci[1,0]),Slope_high=float(ci[1,1]),CITL=float(citl.params[0]),HL_chi2=hl,HL_df=df,HL_P=float(stats.chi2.sf(hl,df)) if df>0 else np.nan,HL_groups=len(d),Small_expected_bins=int(((d.E<5)|((d.N-d.E)<5)).sum()),Brier=brier_score_loss(y,p)),d

def delong_pair(y,a,b):
    vp=[];vn=[];means=[]
    for p in [a,b]:
        pos=p[y==1];neg=p[y==0]
        v=(pos[:,None]>neg[None,:]).astype(float)+.5*(pos[:,None]==neg[None,:])
        vp.append(v.mean(axis=1));vn.append(v.mean(axis=0));means.append(v.mean())
    cov=np.cov(vp)/sum(y==1)+np.cov(vn)/sum(y==0)
    variance=cov[0,0]+cov[1,1]-2*cov[0,1]
    z=abs(means[0]-means[1])/np.sqrt(variance) if variance>0 else (0. if means[0]==means[1] else np.inf)
    return dict(AUC1=means[0],AUC2=means[1],Difference=means[0]-means[1],Z=float(z),P=float(2*stats.norm.sf(z)))

def netbenefit(y,p,t):
    return np.array([np.mean((p>=x)&(y==1))-np.mean((p>=x)&(y==0))*x/(1-x) for x in t])

def savefig(fig,stem):
    for ext in ['png','pdf','tiff']:
        kw={'pil_kwargs':{'compression':'tiff_lzw'}} if ext=='tiff' else {}
        buffer=io.BytesIO()
        fig.savefig(buffer,format=ext,dpi=300,bbox_inches='tight',**kw)
        payload=buffer.getvalue();path=ROOT/'figures'/f'{stem}.{ext}';temporary=path.with_suffix(path.suffix+'.tmp')
        with temporary.open('wb') as output:
            output.write(payload);output.flush();os.fsync(output.fileno())
        assert temporary.read_bytes()==payload
        temporary.replace(path)
    plt.close(fig)

def panel(ax,kind,d,rows):
    y=d.dNCR.to_numpy();pt=np.linspace(.01,.99,99)
    for n in NAMES:
        p=d[n].to_numpy();rr=rows[n];c=COLORS[n]
        if kind=='ROC':
            x,yy,_=roc_curve(y,p);ax.plot(x,yy,color=c,lw=2.1,label=f'{LABELS[n]}: {rr["AUC"]:.3f} ({rr["AUC_low"]:.3f}-{rr["AUC_high"]:.3f})')
        elif kind=='Calibration':
            yy,x=calibration_curve(y,p,n_bins=6,strategy='quantile');ax.plot(x,yy,'o-',color=c,lw=1.8,ms=4,label=f'{LABELS[n]}: Brier={rr["Brier"]:.3f}')
        elif kind=='DCA':ax.plot(pt,netbenefit(y,p,pt),lw=2,color=c,label=LABELS[n])
        else:
            precision,recall,_=precision_recall_curve(y,p);ax.plot(recall,precision,color=c,lw=2,label=f'{LABELS[n]}: AP={rr["AP"]:.3f}')
    if kind in ['ROC','Calibration']:
        ax.plot([0,1],[0,1],'--',color='#777777',lw=1.2);ax.set(xlim=(0,1),ylim=(0,1),xlabel='1 - Specificity' if kind=='ROC' else 'Predicted probability',ylabel='Sensitivity' if kind=='ROC' else 'Observed event proportion')
    elif kind=='DCA':
        prev=y.mean();ax.plot(pt,prev-(1-prev)*pt/(1-pt),'--',color='#555555',lw=1.3,label='Treat all');ax.axhline(0,color='#999999',ls=':',label='Treat none');ax.set(xlim=(0,0.8),ylim=(-.07,max(.38,prev+.035)),xlabel='Threshold probability',ylabel='Net benefit')
    else:
        ax.axhline(y.mean(),ls='--',lw=1,color='#888888',label=f'Prevalence={y.mean():.3f}');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Recall (sensitivity)',ylabel='Precision (PPV)')
    ax.grid(alpha=.18,lw=.6);ax.legend(loc='lower right' if kind=='ROC' else 'best',frameon=False)

def main():
    cohorts=[('Internal OOF','internal_oof_predictions.csv','训练组_内部验证','Training cohort: nested cross-validation'),('Temporal external','external_predictions.csv','时间外部验证组','Temporal external validation'),('Apparent training','apparent_training_predictions.csv','补充_训练组拟合','Training cohort: apparent performance')]
    allrows=[];calrows=[];delong=[];thresholdrows=[]
    for cohort,file,stem,title in cohorts:
        d=pd.read_csv(ROOT/'results'/file);y=d.dNCR.to_numpy();rows={}
        for n in NAMES:
            p=d[n].to_numpy();t=d[n+'_threshold'].to_numpy();m=metrics(y,p,t);m.update(Cohort=cohort,Model=n);allrows.append(m);rows[n]=m
            c,bins=calibration_stats(y,p);c.update(Cohort=cohort,Model=n);calrows.append(c)
            bins.to_csv(ROOT/'results'/f'HL_bins_{file[:-4]}_{n}.csv',index=True)
            for rule,tt in [('Sensitivity-oriented',t),('Default 0.5',np.full(len(y),.5))]:
                yy=(p>=tt);tn,fp,fn,tp=confusion_matrix(y,yy,labels=[0,1]).ravel()
                thresholdrows.append(dict(Cohort=cohort,Model=n,Rule=rule,Threshold_min=min(tt),Threshold_max=max(tt),Sensitivity=tp/(tp+fn),Specificity=tn/(tn+fp),PPV=tp/(tp+fp) if tp+fp else 0,TP=tp,FN=fn,FP=fp,TN=tn))
        if cohort!='Apparent training':
            rr=[dict(Cohort=cohort,Model1=a,Model2=b,**delong_pair(y,d[a].to_numpy(),d[b].to_numpy())) for a,b in itertools.combinations(NAMES,2)]
            adjusted=multipletests([x['P'] for x in rr],method='holm')[1]
            for row,adj in zip(rr,adjusted):row['P_Holm']=adj
            delong.extend(rr)
        fig,axes=plt.subplots(2,2,figsize=(13.8,11.5))
        for ax,kind,letter in zip(axes.flat,['ROC','Calibration','DCA','PR'],'abcd'):
            panel(ax,kind,d,rows);ax.set_title(f'({letter}) '+{'ROC':'ROC curves','Calibration':'Calibration plots','DCA':'Decision curve analysis','PR':'Precision-recall curves'}[kind],loc='left',fontweight='bold',pad=12)
        fig.suptitle(title,fontsize=19,fontweight='bold',y=.995);fig.tight_layout(rect=(0,0,1,.98),h_pad=2.5,w_pad=2.5)
        savefig(fig,stem+'_四联图')
        if cohort!='Apparent training':
            for kind in ['ROC','Calibration','DCA','PR']:
                fig,ax=plt.subplots(figsize=(7,6.5));panel(ax,kind,d,rows);ax.set_title(title,fontsize=14,pad=14);fig.tight_layout();savefig(fig,stem+'_'+kind)
    performance=pd.DataFrame(allrows);performance.to_csv(ROOT/'results'/'table_performance.csv',index=False)
    pd.DataFrame(calrows).to_csv(ROOT/'results'/'table_calibration.csv',index=False)
    pd.DataFrame(delong).to_csv(ROOT/'results'/'table_delong.csv',index=False)
    pd.DataFrame(thresholdrows).to_csv(ROOT/'results'/'threshold_comparison.csv',index=False)
    lock=json.loads((ROOT/'results'/'model_lock.json').read_text())
    hyper=[]
    for n in NAMES:
        for k,v in SPACES[n].items():hyper.append(dict(Model=n,Hyperparameter=k,Search_range=json.dumps(v),Best=json.dumps(lock['final'][n]['params'][k])))
        fixed={'LR':dict(solver='saga',max_iter=3000),'RF':dict(n_jobs=1),'XGB':dict(objective='binary:logistic',tree_method='hist'),'ANN':dict(solver='adam',max_iter=600,early_stopping=True,validation_fraction=.15),'SVM':dict(kernel='rbf',probability=False)}[n]
        for k,v in fixed.items():hyper.append(dict(Model=n,Hyperparameter=k,Search_range='Fixed',Best=json.dumps(v)))
    pd.DataFrame(hyper).to_csv(ROOT/'results'/'table_hyperparameters.csv',index=False)
    for cohort,dfile,_,_ in cohorts:
        d=pd.read_csv(ROOT/'results'/dfile);y=d.dNCR.to_numpy();rr=[]
        for n in NAMES:
            for t,nb in zip(np.linspace(.01,.99,99),netbenefit(y,d[n].to_numpy(),np.linspace(.01,.99,99))):rr.append(dict(Model=n,Threshold=t,Net_benefit=nb))
        pd.DataFrame(rr).to_csv(ROOT/'results'/f'DCA_points_{dfile}',index=False)
    jsonwrite(ROOT/'results'/'evaluation_notes.json',dict(AUC_CI='Stratified bootstrap, 2000 resamples of fixed patient predictions, conditional on the fitted pipeline; not full-pipeline uncertainty.',sensitivity_CI='Wilson 95% confidence interval',calibration='Joint logistic recalibration intercept and slope with Wald CI; CITL also supplied',HL='Ten quantile groups, duplicates dropped; actual groups and g-2 df reported. Interpret cautiously for OOF predictions and small expected cells.',DeLong='Paired DeLong; 10 tests per cohort with Holm correction. Internal OOF inference is exploratory because folds share training observations.',threshold='Outer-fold thresholds derived from inner-CV predictions. Final external threshold derived from development-only final inner-CV predictions. Target sensitivity 0.85.',best_model=lock['best_model']))
    print(performance[['Cohort','Model','AUC','Sensitivity','Specificity','Threshold']].to_string(index=False),flush=True)

if __name__=='__main__':main()
