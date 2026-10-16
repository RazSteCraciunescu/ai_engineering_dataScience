# Contributing

Keep the default workflow local and deterministic. Network integrations and heavyweight frameworks belong behind explicit optional adapters.

Before opening a pull request:

1. Run `python scripts/run_pipeline.py` when data or model behavior changed.
2. Run `pytest -q` and `python scripts/validate_repository.py`.
3. Update the data card, model card or decision note when an assumption changes.
4. Do not commit credentials, private data, downloaded model weights or local tracking databases.
