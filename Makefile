PY ?= python3

.PHONY: all install data pipeline test versions benchmark clean

all: data pipeline test

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.data.download

pipeline:
	$(PY) -m src.pipeline

test:
	$(PY) -m pytest -q tests

# Optional evidence scripts (network access to GitHub; not needed to reproduce results)
versions:
	$(PY) -m src.data.versions

benchmark:
	$(PY) -m src.interactions.benchmark_shap

clean:
	rm -rf data/processed reports/figures/* 
