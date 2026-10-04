PY ?= python3

.PHONY: all install install-dev data pipeline test notebooks versions benchmark clean

# Full reproduction from raw install-dev:
	$(PY) -m pip install -r requirements-dev.txt

data: download, every stage, tests.
all: data pipeline test

install:
	$(PY) -m pip install -r requirements.txt

install-dev:
	$(PY) -m pip install -r requirements-dev.txt

data:
	$(PY) -m src.data.download

pipeline:
	$(PY) -m src.pipeline

test:
	$(PY) -m pytest -q tests

# Display-only notebooks, one per stage, executed with outputs saved (needs requirements-dev.txt
# and a completed `make all`).
notebooks:
	$(PY) -m src.reporting.notebooks
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/*.ipynb

# Optional evidence scripts (network access to GitHub; not needed to reproduce results).
# Their outputs (casdatasets_*.csv, shap_runtime_benchmark.csv) are committed.
versions:
	$(PY) -m src.data.versions

benchmark:
	$(PY) -m src.interactions.benchmark_shap

# Removes data and models only; reports are overwritten by the next `make all`.
clean:
	rm -rf data/raw data/processed
