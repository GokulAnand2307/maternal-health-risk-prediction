# Literature review summary

This document condenses the literature-review foundation used in the MSc project **Predictive Modelling of Maternal Health Risks**.

## Research context

Maternal-health risk assessment often relies on physiological observations such as age, blood pressure, blood sugar, body temperature and heart rate. The project examined whether supervised machine-learning methods could complement traditional assessment by identifying patterns associated with low-, mid- and high-risk observations.

## Themes informing the project

### Feature engineering

Derived variables such as pulse pressure can express clinically meaningful relationships that are less obvious when systolic and diastolic blood pressure are considered independently. The dissertation also tested a risk-age interaction feature while recognising that engineered variables require external validation before clinical interpretation.

### Class imbalance

Healthcare classification often has fewer observations in the class of greatest interest. The project therefore explored SMOTE, while the reproducible repository pipeline applies resampling only to training folds to avoid information leaking into the test set.

### Model comparison

The dissertation compared interpretable and non-linear approaches, including logistic regression, Random Forest, gradient boosting, support-vector machines and an artificial neural network. The portfolio implementation concentrates on Random Forest and XGBoost and reports macro-averaged metrics alongside confusion matrices.

### Evaluation beyond accuracy

Accuracy alone can conceal weak performance for a minority or high-consequence class. Precision, recall, F1, ROC-AUC and class-specific errors were therefore considered. In a maternal-health context, missed high-risk observations may carry a different cost from false alerts.

### Ethics and limitations

The source dataset is small and cannot represent the demographic, socioeconomic, geographic and clinical diversity needed for deployment. Any real-world system would require external validation, calibration, subgroup bias assessment, governance, privacy controls and qualified medical oversight.

## Scope

This is an academic portfolio summary, not a systematic review or medical recommendation. The full dissertation remains the authoritative record of the submitted research.

## Dataset citation

Ahmed, M. (2020). *Maternal Health Risk* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DP5D (CC BY 4.0).

