"""
Shared project paths for the JSE cross-sectional ranking study.

Every research script imports paths from this module so that the pipeline
works regardless of the current terminal directory.

Project layout
--------------
project_root/
├── scripts/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── results/
├── outputs/
│   ├── audits/
│   ├── figures/
│   └── logs/
└── archive/
"""

from __future__ import annotations

from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# SCRIPT DIRECTORY
# ============================================================

SCRIPTS_DIRECTORY = PROJECT_ROOT / "scripts"


# ============================================================
# DATA DIRECTORIES
# ============================================================

DATA_DIRECTORY = PROJECT_ROOT / "data"

DATA_RAW = DATA_DIRECTORY / "raw"

DATA_INTERIM = DATA_DIRECTORY / "interim"
DATA_INTERIM_02_UNIVERSE = DATA_INTERIM / "02_universe"
DATA_INTERIM_03_TARGET = DATA_INTERIM / "03_target"

DATA_PROCESSED = DATA_DIRECTORY / "processed"
DATA_PROCESSED_05_FEATURES = DATA_PROCESSED / "05_features"
DATA_PROCESSED_06_VALIDATION = DATA_PROCESSED / "06_validation"


# ============================================================
# RESULT DIRECTORIES
# ============================================================

RESULTS_DIRECTORY = PROJECT_ROOT / "results"
RESULTS_07_OLS = RESULTS_DIRECTORY / "07_ols"
RESULTS_08_XGBOOST = RESULTS_DIRECTORY / "08_xgboost"
RESULTS_09_COMPARISON = RESULTS_DIRECTORY / "09_comparison"


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUTS_DIRECTORY = PROJECT_ROOT / "outputs"

AUDITS_DIRECTORY = OUTPUTS_DIRECTORY / "audits"
AUDITS_01_DATA = AUDITS_DIRECTORY / "01_data"
AUDITS_02_UNIVERSE = AUDITS_DIRECTORY / "02_universe"
AUDITS_03_TARGET = AUDITS_DIRECTORY / "03_target"
AUDITS_05_FEATURES = AUDITS_DIRECTORY / "05_features"
AUDITS_06_VALIDATION = AUDITS_DIRECTORY / "06_validation"
AUDITS_07_OLS = AUDITS_DIRECTORY / "07_ols"
AUDITS_08_XGBOOST = AUDITS_DIRECTORY / "08_xgboost"
AUDITS_09_COMPARISON = AUDITS_DIRECTORY / "09_comparison"

FIGURES_DIRECTORY = OUTPUTS_DIRECTORY / "figures"
FIGURES_CHAPTER_4 = FIGURES_DIRECTORY / "chapter_4"
FIGURES_CHAPTER_5 = FIGURES_DIRECTORY / "chapter_5"

LOGS_DIRECTORY = OUTPUTS_DIRECTORY / "logs"


# ============================================================
# ARCHIVE DIRECTORIES
# ============================================================

ARCHIVE_DIRECTORY = PROJECT_ROOT / "archive"
ARCHIVE_PRE_SCALE_CORRECTION = (
    ARCHIVE_DIRECTORY / "pre_scale_correction"
)
ARCHIVE_ECONOMIC_EXTENSION = (
    ARCHIVE_DIRECTORY / "economic_extension"
)


# ============================================================
# DIRECTORY CREATION
# ============================================================

DIRECTORIES_TO_CREATE = [
    DATA_RAW,
    DATA_INTERIM_02_UNIVERSE,
    DATA_INTERIM_03_TARGET,
    DATA_PROCESSED_05_FEATURES,
    DATA_PROCESSED_06_VALIDATION,
    RESULTS_07_OLS,
    RESULTS_08_XGBOOST,
    RESULTS_09_COMPARISON,
    AUDITS_01_DATA,
    AUDITS_02_UNIVERSE,
    AUDITS_03_TARGET,
    AUDITS_05_FEATURES,
    AUDITS_06_VALIDATION,
    AUDITS_07_OLS,
    AUDITS_08_XGBOOST,
    AUDITS_09_COMPARISON,
    FIGURES_CHAPTER_4,
    FIGURES_CHAPTER_5,
    LOGS_DIRECTORY,
]


def ensure_project_directories() -> None:
    """Create all active project directories when missing."""

    for directory in DIRECTORIES_TO_CREATE:
        directory.mkdir(parents=True, exist_ok=True)