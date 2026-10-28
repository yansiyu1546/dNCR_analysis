# Trained prediction ensembles

The five files contain LR, RF, XGBoost, ANN and RBF SVM models, with five imputation-specific fitted members in each artifact. Use the supplied Python 3.12 environment and `joblib.load`; the gzip compression is detected automatically.

Explicit full training/validation matrices and preprocessing membership arrays have been removed. Fitted imputers, estimator weights, clipping/scaling statistics, aggregate auxiliary fill values, calibration and primary thresholds remain. **SVM support vectors remain as required by its original prediction function.** These coordinates are training-sample features; no claim of model anonymity is made.

Predict from the repository root:

```powershell
python step_1_modeling_utils.py --input new.csv --output predictions.csv --model LR
```

The five selected predictors are Age, MOCA_Score, Depression, GFR and Operation_Time. See the root README for incomplete-input requirements, units/coding and full-analysis commands. Study data are not included. Use these exported model files rather than the original unfiltered artifacts.
