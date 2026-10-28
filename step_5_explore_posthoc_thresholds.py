"""Post-hoc operating-point exploration after the primary external analysis.

These results must not be presented as a fresh independent validation.
Models and calibration stay unchanged. Thresholds use development data only.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from step_1_modeling_utils import CONFIG,NAMES,choose_threshold,jsonwrite
from step_1_modeling_utils import require_current_analysis
from step_3_evaluate_model_performance import metrics,savefig,COLORS
import matplotlib.pyplot as plt

from step_1_modeling_utils import project_root
ROOT=project_root(__file__)

def main():
    tr,_,audit=require_current_analysis(ROOT,CONFIG)
    internal=pd.read_csv(ROOT/'results'/'internal_oof_predictions.csv');external=pd.read_csv(ROOT/'results'/'external_predictions.csv');rows=[];thresholds={}
    for name in NAMES:
        saved=joblib.load(ROOT/'cache'/f'fit_final_{name}.joblib');thresholds[name]={}
        for target in [.90,.95]:
            final_t=choose_threshold(tr.dNCR.to_numpy()[saved['inner_ids']],saved['inner_calibrated'],target)
            thresholds[name][str(target)]=final_t
            ooft=np.zeros(len(tr))
            for f in range(1,6):
                s=joblib.load(ROOT/'cache'/f'fit_outer{f}_{name}.joblib')
                t=choose_threshold(tr.dNCR.to_numpy()[s['inner_ids']],s['inner_calibrated'],target)
                ooft[internal.outer_fold==f]=t
            internal[name+f'_threshold_{int(100*target)}']=ooft
            external[name+f'_threshold_{int(100*target)}']=final_t
            for cohort,d,t in [('Internal OOF',internal,ooft),('Temporal external',external,np.full(len(external),final_t))]:
                m=metrics(d.dNCR.to_numpy(),d[name].to_numpy(),t)
                rows.append(dict(Cohort=cohort,Model=name,Training_sensitivity_target=target,Analysis_status='Post-hoc exploratory; external cohort reused',**m))
    pd.DataFrame(rows).to_csv(ROOT/'results'/'posthoc_high_sensitivity_metrics.csv',index=False)
    internal.to_csv(ROOT/'results'/'internal_exploratory_thresholds.csv',index=False)
    external.to_csv(ROOT/'results'/'external_exploratory_thresholds.csv',index=False)
    jsonwrite(ROOT/'models'/'exploratory_thresholds.json',dict(status='Post-hoc exploratory operating points. Require prospective validation.',thresholds=thresholds,input_sha256=audit['input_sha256']))
    table=pd.DataFrame(rows);primary=pd.read_csv(ROOT/'results'/'table_performance.csv');fig,axes=plt.subplots(1,2,figsize=(13.5,5.2))
    for name in NAMES:
        p=primary[(primary.Cohort=='Temporal external')&(primary.Model==name)].iloc[0]
        ss=table[(table.Cohort=='Temporal external')&(table.Model==name)].sort_values('Training_sensitivity_target')
        xx=[.85,.90,.95]
        axes[0].plot(xx,[p.Sensitivity,*ss.Sensitivity],'-o',color=COLORS[name],label=name,lw=2)
        axes[1].plot(xx,[p.Specificity,*ss.Specificity],'-o',color=COLORS[name],label=name,lw=2)
    for ax,ylabel in zip(axes,['External sensitivity','External specificity']):
        ax.set(xlabel='Development-CV sensitivity target',ylabel=ylabel,ylim=(.35,1.0),xticks=[.85,.90,.95]);ax.grid(alpha=.2);ax.legend(frameon=False)
    axes[0].axhline(.7,color='#777777',ls='--',lw=1)
    fig.suptitle('Operating-point exploration after primary external evaluation',fontsize=16);fig.tight_layout();savefig(fig,'补充_事后阈值探索')
    print(table[['Cohort','Model','Training_sensitivity_target','Sensitivity','Specificity','Threshold']].to_string(index=False))

if __name__=='__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
