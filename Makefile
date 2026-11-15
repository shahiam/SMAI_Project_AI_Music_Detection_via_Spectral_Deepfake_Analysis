# Use python3 in WSL/Linux when the `python` alias is unavailable.
PYTHON ?= $(shell command -v python 2>/dev/null || command -v python3 2>/dev/null || echo python3)

ifeq ($(OS),Windows_NT)
VENV_PYTHON := .venv/Scripts/python.exe
else
VENV_PYTHON := .venv/bin/python
endif

PIP := $(VENV_PYTHON) -m pip
STREAMLIT := $(VENV_PYTHON) -m streamlit

DEMO_MANIFEST := data/demo/manifest.csv
M6_ROOT ?= data/M6_database
M6_MANIFEST ?= data/m6_manifest.csv

.PHONY: help install test demo classical modern m6-manifest train dashboard all clean

help:
	@echo "SpectraGuard project commands:"
	@echo "  make install       Install Python dependencies and this package"
	@echo "  make test          Run the test suite"
	@echo "  make demo          Generate the small smoke-test dataset"
	@echo "  make classical     Train/evaluate the classical SVM on demo data"
	@echo "  make modern        Train/evaluate ResNet-18 on demo data"
	@echo "  make m6-manifest   Build an M6 manifest after extracting the archive"
	@echo "  make train         Run both demo training branches"
	@echo "  make dashboard     Launch the Streamlit dashboard"
	@echo "  make all           Install, test, and run both demo branches"
	@echo ""
	@echo "M6 example: make m6-manifest M6_ROOT=data/M6_database"

$(VENV_PYTHON):
	$(PYTHON) -m venv .venv

install: $(VENV_PYTHON)
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

test: install
	$(VENV_PYTHON) -m pytest -q

demo: install
	$(VENV_PYTHON) -m music_detection.cli make-demo-data --output data/demo

classical: demo
	$(VENV_PYTHON) -m music_detection.cli classical --manifest $(DEMO_MANIFEST) --output runs/classical

modern: demo
	$(VENV_PYTHON) -m music_detection.cli modern --manifest $(DEMO_MANIFEST) --output runs/modern --epochs 1 --no-pretrained

m6-manifest:
	$(VENV_PYTHON) -m music_detection.cli build-m6-manifest --root $(M6_ROOT) --output $(M6_MANIFEST) --include-duration

train: classical modern

dashboard: install
	$(STREAMLIT) run app.py

all: install test train

clean:
	$(VENV_PYTHON) -c "from pathlib import Path; [p.unlink() for p in Path('.').rglob('*.pyc')]; print('Removed Python bytecode')"
