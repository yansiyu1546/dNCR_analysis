# Released fitted models

LR, RF, XGB, ANN, and SVM pipelines contain the fitted prediction components,
including five imputation-specific models, preprocessing, calibration, and thresholds.
Stored training/validation matrices, row membership indices, and input-file
fingerprints have been removed. Predictions were checked against the original
LR, RF, XGB, and ANN pipelines on the training cohort, external cohort, and
inputs with added missing values; outputs were identical. The cleaned SVM
pipeline was checked on the external cohort, with identical predictions.
No model was refitted.

The SVM binary retains support vectors containing standardized values of the
five selected predictors for 164-166 training observations per imputation.
The stored scalers allow these values to be converted back to the preprocessing
scale. These observations can include imputed or clipped values. Removing the
support vectors would change or break inference; the file therefore retains
this observation-level information.

These release pipelines support prediction; refitting and full analysis require
the original inputs and newly generated analysis artifacts. They are not a
replacement for the private cache or original analysis outputs.

Fitted models still contain learned statistics and parameters. Removing stored
patient matrices does not provide a formal privacy guarantee.
