"""
Script 01B: Isolated Price-Scale Error Audit
--------------------------------------------

Audits the frozen broad JSE panel for isolated mechanical price-scale errors.
It does not modify the data.

Input
-----
JSE_Broad_Master_Panel.csv

Output
------
outputs/audits/JSE_Price_Scale_Candidates.csv

Run
---
python 01B_Price_Scale_Audit.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path("JSE_Broad_Master_Panel.csv")
OUTPUT_FILE = Path("outputs") / "audits" / "JSE_Price_Scale_Candidates.csv"

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close"]
REQUIRED_COLUMNS = {"Date", "Ticker", "Volume", *PRICE_COLUMNS}

CANDIDATE_FACTORS = np.array([0.01, 0.1, 10.0, 100.0])
FACTOR_RELATIVE_TOLERANCE = 0.15
MAX_NEIGHBOUR_RATIO = 2.0


def main() -> None:
    print("=" * 78)
    print("SCRIPT 01B: ISOLATED PRICE-SCALE ERROR AUDIT")
    print("=" * 78)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Required input file not found: {INPUT_FILE.resolve()}"
        )

    panel = pd.read_csv(INPUT_FILE)
    missing = REQUIRED_COLUMNS.difference(panel.columns)

    if missing:
        raise ValueError(
            f"Broad panel is missing required columns: {sorted(missing)}"
        )

    panel["Date"] = pd.to_datetime(panel["Date"], errors="raise")

    for column in PRICE_COLUMNS + ["Volume"]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")

    if panel.duplicated(["Date", "Ticker"]).any():
        raise ValueError("Duplicate Date-Ticker observations were found.")

    if panel["Adj Close"].isna().any() or panel["Adj Close"].le(0).any():
        raise ValueError(
            "Adjusted Close contains missing or non-positive values."
        )

    panel = (
        panel.sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    grouped = panel.groupby("Ticker", sort=False)

    panel["Previous_Date"] = grouped["Date"].shift(1)
    panel["Next_Date"] = grouped["Date"].shift(-1)
    panel["Previous_Adj_Close"] = grouped["Adj Close"].shift(1)
    panel["Next_Adj_Close"] = grouped["Adj Close"].shift(-1)
    panel["Previous_Volume"] = grouped["Volume"].shift(1)
    panel["Next_Volume"] = grouped["Volume"].shift(-1)

    valid = (
        panel["Previous_Adj_Close"].notna()
        & panel["Next_Adj_Close"].notna()
        & panel["Previous_Adj_Close"].gt(0)
        & panel["Next_Adj_Close"].gt(0)
    )

    audit = panel.loc[valid].copy()

    audit["Neighbour_Ratio"] = np.maximum(
        audit["Previous_Adj_Close"] / audit["Next_Adj_Close"],
        audit["Next_Adj_Close"] / audit["Previous_Adj_Close"],
    )

    audit = audit.loc[
        audit["Neighbour_Ratio"] <= MAX_NEIGHBOUR_RATIO
    ].copy()

    audit["Implied_Factor_From_Previous"] = (
        audit["Previous_Adj_Close"] / audit["Adj Close"]
    )
    audit["Implied_Factor_From_Next"] = (
        audit["Next_Adj_Close"] / audit["Adj Close"]
    )
    audit["Implied_Factor"] = audit[
        ["Implied_Factor_From_Previous", "Implied_Factor_From_Next"]
    ].median(axis=1)

    factor_matrix = np.abs(
        audit["Implied_Factor"].to_numpy()[:, None]
        - CANDIDATE_FACTORS[None, :]
    ) / CANDIDATE_FACTORS[None, :]

    nearest_index = factor_matrix.argmin(axis=1)
    audit["Suggested_Correction_Factor"] = (
        CANDIDATE_FACTORS[nearest_index]
    )
    audit["Factor_Relative_Deviation"] = (
        factor_matrix[np.arange(len(audit)), nearest_index]
    )

    audit = audit.loc[
        audit["Factor_Relative_Deviation"]
        <= FACTOR_RELATIVE_TOLERANCE
    ].copy()

    audit["Corrected_Adj_Close"] = (
        audit["Adj Close"]
        * audit["Suggested_Correction_Factor"]
    )
    audit["Corrected_to_Previous_Ratio"] = (
        audit["Corrected_Adj_Close"]
        / audit["Previous_Adj_Close"]
    )
    audit["Corrected_to_Next_Ratio"] = (
        audit["Corrected_Adj_Close"]
        / audit["Next_Adj_Close"]
    )

    audit = audit.loc[
        audit["Corrected_to_Previous_Ratio"].between(0.5, 2.0)
        & audit["Corrected_to_Next_Ratio"].between(0.5, 2.0)
    ].copy()

    for column in PRICE_COLUMNS:
        safe_name = column.replace(" ", "_")
        audit[f"Original_{safe_name}"] = audit[column]
        audit[f"Suggested_{safe_name}"] = (
            audit[column]
            * audit["Suggested_Correction_Factor"]
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
        safe_name = column.replace(" ", "_")
        output_columns.extend(
            [f"Original_{safe_name}", f"Suggested_{safe_name}"]
        )

    candidates = (
        audit[output_columns]
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(OUTPUT_FILE, index=False)

    print(f"Broad-panel rows audited: {len(panel):,}")
    print(f"Candidate scale-error rows: {len(candidates):,}")
    print(f"Audit output: {OUTPUT_FILE}")

    if not candidates.empty:
        print("\nCandidate rows:")
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
            ].to_string(index=False)
        )

    print("=" * 78)


if __name__ == "__main__":
    main()
