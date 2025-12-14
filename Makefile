.PHONY: install lint format format-check typecheck test test-cov test-cov-html check ci docs docs-serve clean publish help

.DEFAULT_GOAL := help

##@ Development Setup

install: ## Install all dependencies with uv
	uv sync --all-extras

install-dev: ## Install with dev dependencies only
	uv sync --extra dev

##@ Code Quality

lint: ## Run ruff linter
	uv run ruff check src tests examples

format: ## Run ruff formatter
	uv run ruff format src tests examples

format-check: ## Check formatting without making changes
	uv run ruff format --check src tests examples

typecheck: ## Run ty type checker
	uv run ty check src

##@ Testing

test: ## Run pytest
	uv run pytest

test-cov: ## Run pytest with coverage
	uv run pytest --cov=src/textagents --cov-report=term-missing --cov-report=xml

test-cov-html: ## Run pytest with HTML coverage report
	uv run pytest --cov=src/textagents --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

##@ Combined Checks

check: lint typecheck test ## Run all checks (lint, typecheck, test)

ci: lint format-check typecheck test-cov ## Run all CI checks

##@ Documentation

docs: ## Build documentation
	uv run mkdocs build

docs-serve: ## Serve documentation locally with live reload
	uv run mkdocs serve

##@ Publishing

publish: ## Build and publish to PyPI
	rm -rf dist/
	uv build
	uv publish

##@ Cleanup

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf coverage.xml
	rm -rf site/
	rm -rf .ty_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

##@ Help

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[32m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[34m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)
