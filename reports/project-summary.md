# Project summary

## Research question

Can a small set of physiological indicators support the classification of maternal-health observations into low-, mid- and high-risk groups?

## Scope

This MSc project investigated six routinely recorded measurements: age, systolic blood pressure, diastolic blood pressure, blood sugar, body temperature and heart rate. Exploratory analysis examined distributions, class overlap and correlations. The dissertation then compared Random Forest, XGBoost and an Artificial Neural Network.

## Reported findings

- XGBoost produced the strongest reported result: 73% accuracy, 84% precision and 90% recall.
- Random Forest achieved 72% accuracy and provided a comparatively interpretable baseline.
- The neural network achieved 66% accuracy, indicating that additional data and tuning would be required.
- Blood pressure and blood sugar showed meaningful associations with the assigned risk level.
- Mid-risk cases overlapped substantially with the other classes.

## Methodological improvements in this repository

The portfolio code places imputation, scaling and SMOTE inside an imbalanced-learn pipeline. The train/test split happens first, so synthetic observations from the test set cannot leak into training. Metrics are reported using macro averaging to avoid hiding weak performance on less frequent classes.

## Limitations

- The dataset is small and may contain repeated or highly similar observations.
- The labels and features do not capture full medical histories, social determinants or longitudinal change.
- Results from a single train/test split are not evidence of clinical generalisability.
- Feature importance indicates predictive contribution, not medical causation.
- No clinical validation, calibration or prospective evaluation was performed.

## Responsible interpretation

The project demonstrates analytical workflow and model evaluation. It does not provide medical advice and should not be used to make decisions about an individual's care.

