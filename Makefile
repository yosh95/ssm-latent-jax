.PHONY: help install test test-fast test-cov lint typecheck fmt examples clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## install the package in editable mode with dev extras
	python -m pip install -e ".[dev]"

test:  ## run the whole test suite
	pytest -q

test-fast:  ## skip the slow meta-learning tests
	pytest -q -m "not slow"

test-cov:  ## run the test suite with a coverage gate
	pytest -q --cov=ssm_latent --cov-report=term-missing --cov-fail-under=90

lint:  ## static checks
	ruff check src tests examples
	ruff format --check src tests examples

typecheck:  ## static type checking
	mypy src

fmt:  ## auto-format
	ruff check --fix src tests examples
	ruff format src tests examples

examples:  ## run every example script
	@for f in examples/0*.py; do echo "--- $$f"; python "$$f"; done

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
