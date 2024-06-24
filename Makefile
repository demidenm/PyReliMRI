# Makefile for running tests with pytest
# To run: make <type>
# Example: make brain_icc  <- runs tests for all in test_brainicc.py

# Default target
.PHONY: brain_icc
brain_icc:
	python -m pytest tests/test_brainicc.py

# Default target
.PHONY: conn_icc
conn_icc:
	python -m pytest tests/test_connicc.py


# Default target
.PHONY: timeseries
timeseries:
	python -m pytest tests/test_maskedtimeseries.py

# Default target
.PHONY: similarity_icc
similarity_icc:
	python -m pytest tests/test_similarity-icc.py

# Target to run all tests in the tests directory
.PHONY: test-all
test-all:
	python -m pytest -v tests/