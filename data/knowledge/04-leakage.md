# Data leakage

Leakage occurs when training uses information unavailable at prediction time or when preprocessing learns from the test set. Split data before fitting vectorizers, imputers or scalers, and exclude final outcomes from initial-triage features.
