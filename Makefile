.PHONY: help install-scripts-test-deps test-pytest test-nf-unit test-nf-cpu test-nf-gpu test-all clean

PROFILE ?= milton

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'

install-scripts-test-deps: ## Install scripts/ pytest dependencies into the CURRENTLY ACTIVE Python env (e.g. after `micromamba activate <env>`). mkdssp is not pip-installable - see scripts/requirements-test.txt.
	pip install -r scripts/requirements-test.txt

test-pytest: ## Run the fast Python test suites (scripts/ needs an env with requirements-test.txt installed; bindsweeper is self-contained via uv)
	cd scripts && pytest
	cd bindsweeper && uv run pytest

test-nf-unit: ## Run fast nf-test Groovy/param unit tests (tests/unit, no containers/GPU)
	nf-test test tests/unit

test-nf-cpu: ## Run all CPU/python_tools nf-test module tests serially
	nf-test test tests/modules --tag cpu --profile $(PROFILE)

test-nf-gpu: ## Run all GPU-tagged nf-test module tests serially (slow - up to ~10 min for RunBC)
	nf-test test tests/modules --tag gpu --profile $(PROFILE)

test-all: test-pytest test-nf-unit test-nf-cpu test-nf-gpu ## Run everything serially: pytest -> nf-test unit -> nf-test cpu -> nf-test gpu

clean: ## Remove nf-test work directory and pytest caches
	rm -rf .nf-test
	rm -rf scripts/.pytest_cache scripts/htmlcov scripts/.coverage
	rm -rf bindsweeper/.pytest_cache bindsweeper/htmlcov bindsweeper/.coverage

.DEFAULT_GOAL := help
