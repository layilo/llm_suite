PYTHON ?= python
PROFILE ?= configs/profiles/local-demo.yaml

.PHONY: install lint test demo smoke report baseline regress

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .[dev]

lint:
	$(PYTHON) -m ruff check src tests

test:
	$(PYTHON) -m pytest --cov=src/llm_benchmark_suite --cov-report=term-missing

demo:
	$(PYTHON) -m llm_benchmark_suite.cli run --config $(PROFILE)

smoke:
	$(PYTHON) -m llm_benchmark_suite.cli demo --output-dir artifacts/generated/demo

report:
	$(PYTHON) -m llm_benchmark_suite.cli report --input artifacts/sample_run/summary.json --format html --output-dir reports/generated

baseline:
	$(PYTHON) -m llm_benchmark_suite.cli export-baseline --input artifacts/sample_run/summary.json --output artifacts/generated/baseline.json

regress:
	$(PYTHON) -m llm_benchmark_suite.cli regress --current artifacts/sample_run/summary.json --baseline artifacts/generated/baseline.json --thresholds configs/thresholds/default.yaml
