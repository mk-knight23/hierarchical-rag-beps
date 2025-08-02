# Makefile for Hierarchical RAG Document Processing Pipeline

.PHONY: help install install-dev test lint format clean build docs run-example

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install package and dependencies"
	@echo "  install-dev  - Install package with development dependencies"
	@echo "  test         - Run all tests"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build package"
	@echo "  docs         - Build documentation"
	@echo "  run-example  - Run basic usage example"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	python run_tests.py

test-unit:
	python -m pytest tests/test_pipeline.py -v

test-coverage:
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 src/ tests/ examples/
	mypy src/

format:
	black src/ tests/ examples/
	isort src/ tests/ examples/

format-check:
	black --check src/ tests/ examples/
	isort --check-only src/ tests/ examples/

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Build
build:
	python setup.py sdist bdist_wheel

# Documentation
docs:
	cd docs && make html

# Examples
run-example:
	python examples/basic_usage.py

run-batch-example:
	python examples/batch_processing.py

# Development workflow
dev-setup: install-dev
	pre-commit install

# Quick development check
check: format lint test

# Docker
docker-build:
	docker build -t hierarchical-rag-beps .

docker-run:
	docker run -it --rm -v $(PWD)/data:/app/data hierarchical-rag-beps

# Environment setup
env-setup:
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && make install-dev

# All-in-one setup for new developers
setup: env-setup check
	@echo "Development environment ready!"