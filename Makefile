PYTHON ?= python

.PHONY: setup pipeline api test validate notebooks clean
setup:
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e .
pipeline:
	$(PYTHON) scripts/run_pipeline.py
api:
	uvicorn feedback_intelligence.api:app --host 127.0.0.1 --port 8000
test:
	$(PYTHON) -m pytest -q
validate:
	$(PYTHON) scripts/validate_repository.py
notebooks:
	$(PYTHON) scripts/build_notebooks.py
clean:
	$(PYTHON) scripts/clean_generated.py
