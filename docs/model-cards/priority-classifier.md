# Model card: priority classifier

A scikit-learn pipeline combines TF-IDF text features, one-hot categorical features and scaled numeric features with logistic regression. It estimates low, medium or high triage priority.

Current held-out metrics and the confusion matrix are stored in `artifacts/reports/model-metrics.json`. The synthetic English-only data and template-based labels limit external validity. The model must not automate access or safety decisions.
