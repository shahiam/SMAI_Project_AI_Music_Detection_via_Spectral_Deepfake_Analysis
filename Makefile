# Use python3 in WSL/Linux when the `python` alias is unavailable.
# Only used to create a new virtual environment; override with `make PYTHON=py` on Windows.
PYTHON ?= $(shell command -v python 2>/dev/null || command -v python3 2>/dev/null || echo python3)

# Optional per-machine settings (gitignored). Example local.mk:
#   VENV := ../../SMAI_env
-include local.mk

# Virtual environment used for every command. Defaults to a project-local .venv.
# Point it at an existing environment with `make VENV=../../SMAI_env <target>` or in local.mk.
VENV ?= .venv

ifeq ($(OS),Windows_NT)
VENV_PYTHON := $(VENV)/Scripts/python.exe
else
VENV_PYTHON := $(VENV)/bin/python
endif

# Paths are quoted so folders containing spaces work.
PY := "$(VENV_PYTHON)"
PIP := $(PY) -m pip
STREAMLIT := $(PY) -m streamlit

DEMO_MANIFEST := data/demo/manifest.csv
M6_ARCHIVE ?= data/M6_database.tar.gz
M6_ROOT ?= data/M6_database
M6_MANIFEST ?= data/m6_manifest.csv

# SONICS (MP3) pipeline. Override SONICS_ROOT in local.mk on other machines (e.g. D:/SONICS on Windows).
SONICS_ROOT ?= $(if $(HOME),$(HOME),$(USERPROFILE))/Datasets/SONICS
SONICS_MANIFEST ?= data/sonics_manifest.csv
SONICS_GB ?= 15
SONICS_JOBS ?= 8
SONICS_CLIP_RATE ?= 32000
SONICS_PREP := $(PY) -m music_detection.sonics.preprocessing

.PHONY: help venv install test demo classical modern extract-m6 m6-manifest m6-classical m6-modern m6-train m6-bl-classical m6-bl-modern m6-bl-ensemble m6-bl sr-baseline sonics-plan sonics-fakes sonics-reals sonics-status sonics-profile sonics-clips sonics-profile-clips sonics-sr-baseline sonics-classical sonics-modern sonics-ensemble sonics-train train dashboard all clean

help:
	@echo "Sarcospectus project commands:"
	@echo "  make install       Install Python dependencies and this package"
	@echo "  make test          Run the test suite"
	@echo "  make demo          Generate the small smoke-test dataset"
	@echo "  make classical     Train/evaluate the classical SVM on demo data"
	@echo "  make modern        Train/evaluate ResNet-18 on demo data"
	@echo "  make extract-m6    Extract data/M6_database.tar.gz into data/"
	@echo "  make m6-manifest   Build an M6 manifest after extracting the archive"
	@echo "  make m6-train      Train both branches on the M6 manifest (original 44.1 kHz pipeline)"
	@echo "  make m6-bl         Band-limited (16 kHz) classical + augmented CNN + validated ensemble"
	@echo "  make sr-baseline   Shortcut check: classify by native sample rate only"
	@echo ""
	@echo "SONICS pipeline (data in SONICS_ROOT=$(SONICS_ROOT)):"
	@echo "  make sonics-plan     Download metadata, choose a stratified balanced subset (SONICS_GB per class)"
	@echo "  make sonics-fakes    Download the selected AI songs (Hugging Face zip parts)"
	@echo "  make sonics-reals    Download the matching human songs from YouTube (SONICS_JOBS parallel)"
	@echo "  make sonics-status   Progress against the plan"
	@echo "  make sonics-profile  Sample rate / codec / bitrate / bandwidth per class (raw songs)"
	@echo "  make sonics-clips    Build identical-format clips + data/sonics_manifest.csv"
	@echo "  make sonics-train    Classical + modern + validated ensemble on the SONICS manifest"
	@echo "  make train         Run both demo training branches"
	@echo "  make dashboard     Launch the Streamlit dashboard"
	@echo "  make all           Install, test, and run both demo branches"
	@echo ""
	@echo "Virtual environment in use: $(VENV)"
	@echo "  Use an existing one:  make install VENV=../../SMAI_env  (or set VENV in local.mk)"
	@echo ""
	@echo "M6 example: make m6-manifest M6_ROOT=data/M6_database"

# Creates $(VENV) only if it does not already contain a virtual environment.
# (The Python fallback also handles absolute VENV paths containing spaces.)
venv:
ifneq ($(wildcard $(VENV)/pyvenv.cfg),)
	@echo "Using existing virtual environment at $(VENV)"
else
	@"$(PYTHON)" -c "import os, sys, venv; p = sys.argv[1]; exists = os.path.isfile(os.path.join(p, 'pyvenv.cfg')); print(('Using existing virtual environment at ' if exists else 'Creating virtual environment at ') + p); exists or venv.create(p, with_pip=True)" "$(VENV)"
endif

install: venv
	@echo "Installing project dependencies..."
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

test: install
	$(PY) -m pytest -q

demo: install
	$(PY) -m music_detection.cli make-demo-data --output "data/demo"

classical: demo
	$(PY) -m music_detection.cli classical --manifest "$(DEMO_MANIFEST)" --output "runs/classical"

modern: demo
	$(PY) -m music_detection.cli modern --manifest "$(DEMO_MANIFEST)" --output "runs/modern" --epochs 1 --no-pretrained

m6-manifest:
	$(PY) -m music_detection.cli build-m6-manifest --root "$(M6_ROOT)" --output "$(M6_MANIFEST)" --include-duration

m6-classical: install
	$(PY) -m music_detection.cli classical --manifest "$(M6_MANIFEST)" --output "runs/m6_classical"

m6-modern: install
	$(PY) -m music_detection.cli modern --manifest "$(M6_MANIFEST)" --output "runs/m6_modern" --epochs 10

m6-train: m6-classical m6-modern

# Band-limited experiment: every clip resampled to 16 kHz so AI and human audio share the
# same 0-8 kHz bandwidth (removes the native-sample-rate shortcut in M6).
m6-bl-classical: install
	$(PY) -m music_detection.cli classical --manifest "$(M6_MANIFEST)" --output "runs/m6_bl_classical" --sample-rate 16000

m6-bl-modern: install
	$(PY) -m music_detection.cli modern --manifest "$(M6_MANIFEST)" --output "runs/m6_bl_modern" --sample-rate 16000 --epochs 12 --augment --audio-augment

m6-bl-ensemble: install
	$(PY) -m music_detection.cli ensemble --classical "runs/m6_bl_classical" --modern "runs/m6_bl_modern" --output "runs/m6_bl_ensemble"

m6-bl: m6-bl-classical m6-bl-modern m6-bl-ensemble

sr-baseline: install
	$(PY) -m music_detection.cli sr-baseline --manifest "$(M6_MANIFEST)" --output "runs/sr_baseline.json"

# ---------------- SONICS (MP3) pipeline ----------------
# Downloads are long: on macOS prefix with caffeinate, e.g. `caffeinate -i make sonics-reals`.
sonics-plan:
	$(SONICS_PREP).fetch plan --root "$(SONICS_ROOT)" --gb-per-class $(SONICS_GB)

sonics-fakes:
	$(SONICS_PREP).fetch fakes --root "$(SONICS_ROOT)"

sonics-reals:
	$(SONICS_PREP).fetch reals --root "$(SONICS_ROOT)" --jobs $(SONICS_JOBS)

sonics-status:
	$(SONICS_PREP).fetch status --root "$(SONICS_ROOT)"

sonics-profile:
	$(SONICS_PREP).profile_audio --dir fake="$(SONICS_ROOT)/fake_songs" --dir real="$(SONICS_ROOT)/real_songs" --csv data/sonics_profile_raw.csv

sonics-clips:
	$(SONICS_PREP).prepare_clips --sonics-root "$(SONICS_ROOT)" --real-dir "$(SONICS_ROOT)/real_songs" --out-dir "$(SONICS_ROOT)/clips" --manifest "$(SONICS_MANIFEST)" --balance --workers $(SONICS_JOBS) --sample-rate $(SONICS_CLIP_RATE)

sonics-profile-clips:
	$(SONICS_PREP).profile_audio --dir fake="$(SONICS_ROOT)/clips/fake" --dir real="$(SONICS_ROOT)/clips/real" --csv data/sonics_profile_clips.csv

sonics-sr-baseline: install
	$(PY) -m music_detection.cli sr-baseline --manifest "$(SONICS_MANIFEST)" --output "runs/sonics_sr_baseline.json"

sonics-classical: install
	$(PY) -m music_detection.cli classical --manifest "$(SONICS_MANIFEST)" --output "runs/sonics_classical" --sample-rate 16000

sonics-modern: install
	$(PY) -m music_detection.cli modern --manifest "$(SONICS_MANIFEST)" --output "runs/sonics_modern" --sample-rate 16000 --epochs 12 --augment --audio-augment

sonics-ensemble: install
	$(PY) -m music_detection.cli ensemble --classical "runs/sonics_classical" --modern "runs/sonics_modern" --output "runs/sonics_ensemble"

sonics-train: sonics-classical sonics-modern sonics-ensemble

extract-m6:
	tar -xzf "$(M6_ARCHIVE)" -C data

train: classical modern

dashboard: install
	$(STREAMLIT) run app.py

all: install test train

clean:
	$(PY) -c "from pathlib import Path; [p.unlink() for p in Path('.').rglob('*.pyc')]; print('Removed Python bytecode')"
