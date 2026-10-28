# Trained Prediction Models

This folder contains five trained prediction models: logistic regression (LR), random forest (RF), XGBoost, artificial neural network (ANN), and support vector machine (SVM). Each model contains five fitted members corresponding to the five imputed datasets.

The models are saved in `.joblib` format and can be loaded using `joblib.load` in the provided Python environment.

To generate predictions, run the following command from the repository root:

```
python step_1_modeling_utils.py --input new.csv --output predictions.csv --model LR
```

The five selected predictors are Age, MOCA_Score, Depression, GFR, and Operation_Time. Please refer to the main README for input requirements, variable coding, and instructions for running the analysis.

To protect patient privacy, the original study datasets and explicit training/validation matrices are not included. The exported models retain the fitted parameters and preprocessing components required for prediction. SVM support vectors are retained because they are necessary for its prediction function and may contain training-sample feature information.

Please use the exported model files provided in this folder.
