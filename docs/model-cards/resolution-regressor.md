# Model card: resolution-time regressor

A preprocessing pipeline feeds a random-forest regressor. It estimates resolution hours from decision-time context. Extreme target values are capped only in the training copy; raw values remain in the canonical data.

Metrics are stored in `artifacts/reports/model-metrics.json`. The estimate is not a service-level guarantee and omits staffing, queues and dependencies.
