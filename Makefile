PY ?= python3

.PHONY: all install data pipeline test versions benchmark clean

# Full reproduction from raw data: download, every stage, tests.
all: data pipeline test

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.data.download

pipeline:
	$(PY) -m src.pipeline

test:
	$(PY) -m pytest -q tests

# Optional evidence scripts (network access to GitHub; not needed to reproduce results).
# Their outputs (casdatasets_*.csv, shap_runtime_benchmark.csv) are committed.
versions:
	$(PY) -m src.data.versions

benchmark:
	$(PY) -m src.interactions.benchmark_shap

# Removes data and models only; reports are overwritten by the next `make all`.
clean:
	rm -rf data/raw data/processed
