# dNCR prediction models and analysis code

本目录可直接作为GitHub仓库内容：5个分析脚本、requirements、README和model里的5个模型。数据读取与预测功能已合并到step_1，不依赖包外的core.py、data_io.py或evaluate.py。无需另传核验记录、metadata或重复的权重文件。

This repository contains the scientific analysis code, deterministic random seeds and five trained model ensembles. Study datasets are not included. The distributed model artifacts have their explicit full training/validation matrices and preprocessing membership arrays removed. **RBF SVM support vectors are retained**, because they are required for its original prediction function. The artifacts are not claimed to be anonymous.

## Files

```text
README.md
requirements.txt
step_1_modeling_utils.py
step_2_run_primary_analysis.py
step_3_evaluate_model_performance.py
step_4_analyze_stability_and_shap.py
step_5_explore_posthoc_thresholds.py
model/
    README.md
    ANN_pipeline.joblib
    LR_pipeline.joblib
    RF_pipeline.joblib
    SVM_pipeline.joblib
    XGB_pipeline.joblib
```

The additional hidden `.gitignore` helps keep private data, caches and local analysis outputs out of Git. Preserve this structure when uploading. Each model file is below 25 MiB, gzip-compressed using standard pickle; load it normally with `joblib.load`.

## Environment

Validated with Python 3.12.10 and the pinned versions in `requirements.txt`.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe step_1_modeling_utils.py --help
```

Use the corresponding virtual-environment Python path on other operating systems.

## Predict without study datasets

Create your own CSV, for example the following **fictional** record:

```csv
Age,MOCA_Score,Depression,GFR,Operation_Time
70,24,0,90,90
```

Save it as `new.csv`, then run:

```powershell
.\.venv\Scripts\python.exe step_1_modeling_utils.py --input new.csv --output predictions.csv --model LR
```

Model choices are LR, RF, XGB, ANN and SVM. CSV and XLSX inputs are supported. Existing output files are not overwritten. The output contains calibrated `probability_dNCR`, the locked primary `threshold`, and `predicted_class` (probability >= threshold). A supplied row_id is retained.

When all five selected predictors are observed, only Age, MOCA_Score, Depression, GFR and Operation_Time are required. If a selected predictor is missing, supply all 38 candidate columns below, retaining blank cells for missing observations. The saved fitted imputers transform these inputs without refitting. Do not replace missing values with zero.

```text
Surgery_Type, Age, Education, MOCA_Score, Frailty, Depression, ASA,
Nutritional_Risk, Sleep_Disorder, Alcohol, ADL, NRS, Stroke_History,
Coronary_Heart_Disease, Hypertension, Diabetes, NLR, CRP, ALB, ESR,
ALT, AST, Cr, GFR, K, Na, H, Ca, Mg, P, Intraoperative_Blood_Loss,
Operation_Time, Anesthesia_Time, Sufentanil, Cisatracurium,
Neostigmine, Atropine, Propofol
```

Inputs must use the original study coding and units. Surgery_Type uses 1/2/3, Education and ADL use 0–4, and the binary fields use 0/1. Clinical meanings, particularly recoded ASA, must match the study codebook. Posterior imputation is reproducible for identical batch composition and row order; changing them can change imputed predictions. The original behavior is preserved.

## Full analysis with separately supplied research data

Full cohort statistics require the corresponding study datasets. Place authorized local files at `private_analysis/data/training.xlsx` and `private_analysis/data/external.xlsx`, each containing the candidate predictors and binary dNCR outcome, then run:

```powershell
.\.venv\Scripts\python.exe step_2_run_primary_analysis.py --jobs 6 --fresh
.\.venv\Scripts\python.exe step_3_evaluate_model_performance.py
.\.venv\Scripts\python.exe step_4_analyze_stability_and_shap.py --jobs 6
.\.venv\Scripts\python.exe step_5_explore_posthoc_thresholds.py
```

Use `DNCR_ANALYSIS_ROOT` to select another dedicated workspace. `--fresh` archives earlier outputs inside that workspace. Private imputed matrices, caches, newly fitted models and predictions are written there and do not overwrite the distributed `model` directory. Do not upload the private workspace or its newly fitted unfiltered models.

The stages implement fold-specific imputation and Boruta consensus, paired-variable priority rules, nested hyperparameter selection, calibration, threshold selection, performance/calibration/DeLong/DCA, 50-run Boruta stability, selection-rule sensitivity, SHAP and post-hoc threshold exploration. Full training was not repeated during packaging.

## Random seeds and model details

The master seed is **20261017**. All child seeds are derived as:

```python
int(np.random.SeedSequence([20261017, *integer_parts]).generate_state(1)[0])
```

The configuration and every seed call are included in step_1 through step_5. Outer CV uses the master seed; child-seed components identify inner CV, fold context, imputation, Boruta repetition, model member, search, bootstrap and SHAP. Fixed seeds enable repeatability but do not establish selection stability; the stability analyses are included separately.

Each model artifact contains five fitted members, clipping boundaries, fitted imputers, aggregate fill values, scaling where applicable, calibration and the primary threshold. The classifier parameters are under `artifact['fitted']`; ANN weights/biases are `model.coefs_` and `model.intercepts_`, LR coefficients/intercept are `model.coef_` and `model.intercept_`, RF trees are `model.estimators_`, XGBoost uses `model.get_booster()`, and SVM parameters include support vectors and dual coefficients.

The compact artifacts were compared with the previously verified exports using complete and missing artificial inputs; the checked probabilities were unchanged. The previous exports had also been compared with the original models on identical observed cohort inputs. This verifies prediction preservation, not a new clinical evaluation. No model was retrained, retuned or recalibrated for packaging.
