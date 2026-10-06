"""
Script 01B: Isolated Price-Scale Error Audit
--------------------------------------------

Purpose
-------
Audits the corrected frozen broad JSE panel for isolated mechanical
price-scale errors.

This script does not modify the dataset.

Input
-----
data/raw/JSE_Broad_Master_Panel.csv

Output
------
outputs/audits/01_data/JSE_Price_Scale_Candidates.csv

Run from the project root
-------------------------
python .\scripts\01B_Price_Scale_Audit.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "JSE_Broad_Master_Panel.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "audits"
    / "01_data"
    / "JSE_Price_Scale_Candidates.csv"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

PRICE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
]

REQUIRED_COLUMNS = {
    "Date",
    "Ticker",
    "Volume",
    *PRICE_COLUMNS,
}

CANDIDATE_FACTORS = np.array(
    [0.01, 0.1, 10.0, 100.0]
)

FACTOR_RELATIVE_TOLERANCE = 0.15
MAX_NEIGHBOUR_RATIO = 2.0


# ============================================================
# 3. MAIN
# ============================================================

def main() -> None:
    print("=" * 78)
    print("SCRIPT 01B: ISOLATED PRICE-SCALE ERROR AUDIT")
    print("=" * 78)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Required input file not found:\n"
            f"{INPUT_FILE.resolve()}"
        )

    panel = pd.read_csv(
        INPUT_FILE
    )

    missing_columns = REQUIRED_COLUMNS.difference(
        panel.columns
    )

    if missing_columns:
        raise ValueError(
            "The broad panel is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    panel["Date"] = pd.to_datetime(
        panel["Date"],
        errors="raise",
    )

    for column in PRICE_COLUMNS + ["Volume"]:
        panel[column] = pd.to_numeric(
            panel[column],
            errors="coerce",
        )

    if panel.duplicated(
        ["Date", "Ticker"]
    ).any():
        raise ValueError(
            "Duplicate Date-Ticker observations were found."
        )

    if (
        panel["Adj Close"].isna().any()
        or panel["Adj Close"].le(0).any()
    ):
        raise ValueError(
            "Adjusted Close contains missing or non-positive values."
        )

    panel = (
        panel.sort_values(
            ["Ticker", "Date"]
        )
        .reset_index(
            drop=True
        )
    )

    grouped = panel.groupby(
        "Ticker",
        sort=False,
    )

    panel["Previous_Date"] = grouped[
        "Date"
    ].shift(1)

    panel["Next_Date"] = grouped[
        "Date"
    ].shift(-1)

    panel["Previous_Adj_Close"] = grouped[
        "Adj Close"
    ].shift(1)

    panel["Next_Adj_Close"] = grouped[
        "Adj Close"
    ].shift(-1)

    panel["Previous_Volume"] = grouped[
        "Volume"
    ].shift(1)

    panel["Next_Volume"] = grouped[
        "Volume"
    ].shift(-1)

    valid_neighbours = (
        panel["Previous_Adj_Close"].notna()
        & panel["Next_Adj_Close"].notna()
        & panel["Previous_Adj_Close"].gt(0)
        & panel["Next_Adj_Close"].gt(0)
    )

    audit = panel.loc[
        valid_neighbours
    ].copy()

    audit["Neighbour_Ratio"] = np.maximum(
        audit["Previous_Adj_Close"]
        / audit["Next_Adj_Close"],
        audit["Next_Adj_Close"]
        / audit["Previous_Adj_Close"],
    )

    audit = audit.loc[
        audit["Neighbour_Ratio"]
        <= MAX_NEIGHBOUR_RATIO
    ].copy()

    audit[
        "Implied_Factor_From_Previous"
    ] = (
        audit["Previous_Adj_Close"]
        / audit["Adj Close"]
    )

    audit[
        "Implied_Factor_From_Next"
    ] = (
        audit["Next_Adj_Close"]
        / audit["Adj Close"]
    )

    audit["Implied_Factor"] = audit[
        [
            "Implied_Factor_From_Previous",
            "Implied_Factor_From_Next",
        ]
    ].median(
        axis=1
    )

    factor_matrix = (
        np.abs(
            audit[
                "Implied_Factor"
            ].to_numpy()[:, None]
            - CANDIDATE_FACTORS[None, :]
        )
        / CANDIDATE_FACTORS[None, :]
    )

    nearest_factor_index = (
        factor_matrix.argmin(
            axis=1
        )
    )

    audit[
        "Suggested_Correction_Factor"
    ] = CANDIDATE_FACTORS[
        nearest_factor_index
    ]

    audit[
        "Factor_Relative_Deviation"
    ] = factor_matrix[
        np.arange(
            len(audit)
        ),
        nearest_factor_index,
    ]

    audit = audit.loc[
        audit[
            "Factor_Relative_Deviation"
        ]
        <= FACTOR_RELATIVE_TOLERANCE
    ].copy()

    audit[
        "Corrected_Adj_Close"
    ] = (
        audit["Adj Close"]
        * audit[
            "Suggested_Correction_Factor"
        ]
    )

    audit[
        "Corrected_to_Previous_Ratio"
    ] = (
        audit["Corrected_Adj_Close"]
        / audit["Previous_Adj_Close"]
    )

    audit[
        "Corrected_to_Next_Ratio"
    ] = (
        audit["Corrected_Adj_Close"]
        / audit["Next_Adj_Close"]
    )

    audit = audit.loc[
        audit[
            "Corrected_to_Previous_Ratio"
        ].between(
            0.5,
            2.0,
        )
        & audit[
            "Corrected_to_Next_Ratio"
        ].between(
            0.5,
            2.0,
        )
    ].copy()

    for column in PRICE_COLUMNS:
        safe_name = column.replace(
            " ",
            "_",
        )

        audit[
            f"Original_{safe_name}"
        ] = audit[column]

        audit[
            f"Suggested_{safe_name}"
        ] = (
            audit[column]
            * audit[
                "Suggested_Correction_Factor"
            ]
        )

    output_columns = [
        "Ticker",
        "Date",
        "Previous_Date",
        "Next_Date",
        "Previous_Adj_Close",
        "Adj Close",
        "Next_Adj_Close",
        "Implied_Factor_From_Previous",
        "Implied_Factor_From_Next",
        "Suggested_Correction_Factor",
        "Factor_Relative_Deviation",
        "Corrected_Adj_Close",
        "Corrected_to_Previous_Ratio",
        "Corrected_to_Next_Ratio",
        "Previous_Volume",
        "Volume",
        "Next_Volume",
    ]

    for column in PRICE_COLUMNS:
        safe_name = column.replace(
            " ",
            "_",
        )

        output_columns.extend(
            [
                f"Original_{safe_name}",
                f"Suggested_{safe_name}",
            ]
        )

    candidates = (
        audit[
            output_columns
        ]
        .sort_values(
            ["Date", "Ticker"]
        )
        .reset_index(
            drop=True
        )
    )

    candidates.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Input file: "
        f"{INPUT_FILE.relative_to(PROJECT_ROOT)}"
    )

    print(
        f"Broad-panel rows audited: "
        f"{len(panel):,}"
    )

    print(
        "Candidate scale-error rows: "
        f"{len(candidates):,}"
    )

    print(
        f"Audit output: "
        f"{OUTPUT_FILE.relative_to(PROJECT_ROOT)}"
    )

    if candidates.empty:
        print(
            "\nIntegrity check passed: "
            "no isolated mechanical price-scale errors were detected."
        )
    else:
        print(
            "\nCandidate rows:"
        )

        print(
            candidates[
                [
                    "Ticker",
                    "Date",
                    "Previous_Adj_Close",
                    "Adj Close",
                    "Next_Adj_Close",
                    "Suggested_Correction_Factor",
                ]
            ].to_string(
                index=False
            )
        )

    print("=" * 78)


if __name__ == "__main__":
    main()