# Canopy task runner. Works in WSL, Linux, macOS and Git Bash.
# PowerShell users: see tasks.ps1, which exposes the same targets.
#
# Everything runs through `uv run`, so no environment needs activating and the
# commands here are exactly what CI executes.

UV ?= uv
RUN := $(UV) run

.DEFAULT_GOAL := help
.PHONY: help setup setup-lean lock lint format typecheck test test-cov test-all \
        check fly fly-physics clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Create/refresh the environment from uv.lock
	$(UV) sync

setup-lean: ## Environment without the physics group (mirrors a Windows sync)
	$(UV) sync --no-group physics

lock: ## Re-resolve dependencies and update uv.lock
	$(UV) lock

lint: ## Lint and check formatting
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format: ## Auto-fix lint findings and format
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

typecheck: ## Strict type check
	$(RUN) mypy

test: ## Run the fast tests
	$(RUN) pytest

test-cov: ## Run tests with a coverage report
	$(RUN) pytest --cov --cov-report=term-missing --cov-report=xml

test-all: ## Include the slow/physics tests
	$(RUN) pytest -m "slow or physics or not slow"

check: lint typecheck test ## Everything CI runs

fly: ## Single-drone flight check, kinematic
	$(RUN) canopy-fly

fly-physics: ## Single-drone flight check, PyBullet GUI (Linux/WSL only)
	$(RUN) canopy-fly --dynamics pybullet --gui

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov coverage.xml .coverage
	find . -type d -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +
