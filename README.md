# dNCR Prediction Models and Analysis Code
## Overview
This repository contains the analysis code and trained prediction models used in our study of delayed neurocognitive recovery (dNCR) in elderly patients undergoing major orthopedic surgery.
Five prediction models were developed and evaluated: logistic regression (LR), random forest (RF), XGBoost, artificial neural network (ANN), and support vector machine (SVM).
## Repository Contents
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
The scripts cover data preprocessing, feature selection, model development, performance evaluation, Boruta stability analysis, and exploratory threshold analysis. The `model` folder contains the five trained prediction models.
## Environment
The code was validated using Python 3.12.10. Required dependencies and their versions are provided in `requirements.txt`.
## Reproducibility
The master random seed was fixed at **20261017**, with derived seeds used throughout the analysis.
Feature selection stability was assessed through 50 repeated Boruta runs. The analysis scripts include the procedures for nested cross-validation, model calibration, performance evaluation, and sensitivity analyses.
## Model Availability
The five trained models are provided as `.joblib` files and can be loaded using `joblib.load`.
The models can be used to generate predictions for new data with the required predictor variables and appropriate coding.
## Data Availability
The original patient-level datasets are not publicly available due to patient privacy and ethical restrictions. Therefore, reproducing the complete cohort-level analysis requires access to the corresponding study datasets.
However, the analysis scripts, random seed configurations, and trained model files are provided to support transparency and reproducibility.
