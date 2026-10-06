"""
Script 01C: Apply Confirmed Price-Scale Corrections
--------------------------------------------------

Purpose
-------
Applies the confirmed factor-of-100 corrections listed in
outputs/audits/JSE_Price_Scale_Candidates.csv to the frozen broad panel.

The script:
1. validates every candidate against the current broad panel;
2. creates a one-time backup of the uncorrected panel;
3. multiplies Open, High, Low, Close and Adjusted Close by the confirmed factor;
4. leaves Volume unchanged;
5. writes a detailed correction audit;
6. replaces JSE_Broad_Master_Panel.csv with the corrected panel; and
7. verifies that no confirmed candidate remains uncorrected.

Inputs
------
1. JSE_Broad_Master_Panel.csv
2. outputs/audits/JSE_Price_Scale_Candidates.csv

Outputs
-------
1. archive/pre_scale_correction/JSE_Broad_Master_Panel_PreScaleCorrection.csv
2. outputs/audits/JSE_Price_Scale_Corrections_Applied.csv
3. corrected JSE_Broad_Master_Panel.csv

Run
---
python 01C_Apply_Price_Scale_Corrections.py
"""

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_PANEL = Path("JSE_Broad_Master_Panel.csv")
INPUT_CANDIDATES = (
    Path("outputs") / "audits" / "JSE_Price_Scale_Candidates.csv"
)

BACKUP_DIRECTORY = Path("archive") / "pre_scale_correction"
BACKUP_PANEL = (
    BACKUP_DIRECTORY / "JSE_Broad_Master_Panel_PreScaleCorrection.csv"
)

OUTPUT_AUDIT = (
    Path("outputs") / "audits"
    / "JSE_Price_Scale_Corrections_Applied.csv"
)

TEMP_CORRECTED_PANEL = Path(
    "JSE_Broad_Master_Panel_corrected_temp.csv"
)

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close"]

REQUIRED_PANEL_COLUMNS = {
    "Date",
    "Ticker",
    "Volume",
    *PRICE_COLUMNS,
}

REQUIRED_CANDIDATE_COLUMNS = {
    "Date",
    "Ticker",
    "Suggested_Correction_Factor",
    *{
        f"Original_{column.replace(' ', '_')}"
        for column in PRICE_COLUMNS
    },
}

EXPECTED_CONFIRMED_ROWS = {
    ("DTC.JO", "2025-01-10"),
    ("SLM.JO", "2025-01-10"),
    ("VOD.JO", "2025-01-10"),
    ("SBK.JO", "2025-04-25"),
}

EXPECTED_CORRECTION_FACTOR = 100.0
NUMERIC_TOLERANCE = 1e-8


# ============================================================
# 2. VALIDATION HELPERS
# ============================================================

def validate_files() -> None:
    missing = [
        str(path)
        for path in [INPUT_PANEL, INPUT_CANDIDATES]
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required files are missing:\n- "
            + "\n- ".join(missing)
        )


def validate_inputs(
    panel: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    missing_panel_columns = REQUIRED_PANEL_COLUMNS.difference(
        panel.columns
    )

    if missing_panel_columns:
        raise ValueError(
            "Broad panel is missing required columns: "
            f"{sorted(missing_panel_columns)}"
        )

    missing_candidate_columns = REQUIRED_CANDIDATE_COLUMNS.difference(
        candidates.columns
    )

    if missing_candidate_columns:
        raise ValueError(
            "Candidate audit is missing required columns: "
            f"{sorted(missing_candidate_columns)}"
        )

    if panel.duplicated(["Date", "Ticker"]).any():
        raise ValueError(
            "Broad panel contains duplicate Date-Ticker observations."
        )

    if candidates.duplicated(["Date", "Ticker"]).any():
        raise ValueError(
            "Candidate audit contains duplicate Date-Ticker observations."
        )

    actual_candidate_rows = {
        (
            str(row.Ticker),
            pd.Timestamp(row.Date).strftime("%Y-%m-%d"),
        )
        for row in candidates.itertuples(index=False)
    }

    if actual_candidate_rows != EXPECTED_CONFIRMED_ROWS:
        raise ValueError(
            "Candidate rows do not match the four confirmed corrections.\n"
            f"Expected: {sorted(EXPECTED_CONFIRMED_ROWS)}\n"
            f"Observed: {sorted(actual_candidate_rows)}"
        )

    factors = candidates["Suggested_Correction_Factor"].to_numpy(
        dtype=float
    )

    if not np.allclose(
        factors,
        EXPECTED_CORRECTION_FACTOR,
        rtol=0.0,
        atol=NUMERIC_TOLERANCE,
    ):
        raise ValueError(
            "At least one candidate does not use the confirmed "
            "factor-of-100 correction."
        )


# ============================================================
# 3. MAIN
# ============================================================

def main() -> None:
    print("=" * 78)
    print("SCRIPT 01C: APPLY CONFIRMED PRICE-SCALE CORRECTIONS")
    print("=" * 78)

    validate_files()

    panel = pd.read_csv(INPUT_PANEL)
    candidates = pd.read_csv(INPUT_CANDIDATES)

    panel["Date"] = pd.to_datetime(panel["Date"], errors="raise")
    candidates["Date"] = pd.to_datetime(
        candidates["Date"],
        errors="raise",
    )

    for column in PRICE_COLUMNS + ["Volume"]:
        panel[column] = pd.to_numeric(
            panel[column],
            errors="coerce",
        )

    candidates["Suggested_Correction_Factor"] = pd.to_numeric(
        candidates["Suggested_Correction_Factor"],
        errors="raise",
    )

    validate_inputs(panel, candidates)

    # --------------------------------------------------------
    # 3.1 Validate every current value against the audit
    # --------------------------------------------------------

    panel_index = panel.set_index(["Ticker", "Date"], drop=False)

    audit_records: list[dict[str, object]] = []

    for candidate in candidates.itertuples(index=False):
        key = (candidate.Ticker, candidate.Date)

        if key not in panel_index.index:
            raise KeyError(
                f"Candidate row not found in broad panel: {key}"
            )

        current_row = panel_index.loc[key]

        if isinstance(current_row, pd.DataFrame):
            raise ValueError(
                f"Multiple broad-panel rows found for candidate: {key}"
            )

        factor = float(candidate.Suggested_Correction_Factor)

        record: dict[str, object] = {
            "Ticker": candidate.Ticker,
            "Date": candidate.Date,
            "Correction_Factor": factor,
            "Original_Volume": float(current_row["Volume"]),
            "Corrected_Volume": float(current_row["Volume"]),
            "Volume_Changed": False,
        }

        for column in PRICE_COLUMNS:
            safe_name = column.replace(" ", "_")
            audited_original = float(
                getattr(candidate, f"Original_{safe_name}")
            )
            current_value = float(current_row[column])

            if not np.isclose(
                current_value,
                audited_original,
                rtol=NUMERIC_TOLERANCE,
                atol=NUMERIC_TOLERANCE,
            ):
                raise ValueError(
                    f"{candidate.Ticker} on {candidate.Date.date()}: "
                    f"current {column} ({current_value}) does not match "
                    f"audited original ({audited_original}). "
                    "The panel may already have been modified."
                )

            corrected_value = current_value * factor

            record[f"Original_{safe_name}"] = current_value
            record[f"Corrected_{safe_name}"] = corrected_value

        audit_records.append(record)

    # --------------------------------------------------------
    # 3.2 Create one-time backup
    # --------------------------------------------------------

    BACKUP_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if BACKUP_PANEL.exists():
        raise FileExistsError(
            "The pre-correction backup already exists:\n"
            f"{BACKUP_PANEL.resolve()}\n"
            "No changes were applied. Confirm whether the correction "
            "has already been run before proceeding."
        )

    shutil.copy2(INPUT_PANEL, BACKUP_PANEL)

    # --------------------------------------------------------
    # 3.3 Apply corrections
    # --------------------------------------------------------

    corrected_panel = panel.copy()

    for candidate in candidates.itertuples(index=False):
        mask = (
            corrected_panel["Ticker"].eq(candidate.Ticker)
            & corrected_panel["Date"].eq(candidate.Date)
        )

        if int(mask.sum()) != 1:
            raise AssertionError(
                f"Expected one row for {candidate.Ticker} "
                f"on {candidate.Date.date()}, found {int(mask.sum())}."
            )

        factor = float(candidate.Suggested_Correction_Factor)

        corrected_panel.loc[mask, PRICE_COLUMNS] = (
            corrected_panel.loc[mask, PRICE_COLUMNS] * factor
        )

    # --------------------------------------------------------
    # 3.4 Post-correction validation
    # --------------------------------------------------------

    if corrected_panel.duplicated(["Date", "Ticker"]).any():
        raise AssertionError(
            "Duplicate Date-Ticker rows appeared after correction."
        )

    if corrected_panel[PRICE_COLUMNS].isna().any().any():
        raise AssertionError(
            "Missing price values appeared after correction."
        )

    if corrected_panel[PRICE_COLUMNS].le(0).any().any():
        raise AssertionError(
            "Non-positive price values appeared after correction."
        )

    if not corrected_panel["Volume"].equals(panel["Volume"]):
        raise AssertionError(
            "Volume changed during price-scale correction."
        )

    # Confirm exact corrected values against the audit records.
    corrected_index = corrected_panel.set_index(
        ["Ticker", "Date"],
        drop=False,
    )

    for record in audit_records:
        key = (record["Ticker"], pd.Timestamp(record["Date"]))
        corrected_row = corrected_index.loc[key]

        for column in PRICE_COLUMNS:
            safe_name = column.replace(" ", "_")
            expected_value = float(
                record[f"Corrected_{safe_name}"]
            )
            observed_value = float(corrected_row[column])

            if not np.isclose(
                observed_value,
                expected_value,
                rtol=NUMERIC_TOLERANCE,
                atol=NUMERIC_TOLERANCE,
            ):
                raise AssertionError(
                    f"Post-correction validation failed for "
                    f"{record['Ticker']} {record['Date']} {column}."
                )

    # --------------------------------------------------------
    # 3.5 Save atomically and write audit
    # --------------------------------------------------------

    corrected_panel = (
        corrected_panel
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    corrected_panel.to_csv(
        TEMP_CORRECTED_PANEL,
        index=False,
    )

    # Replace the original only after all validations and temporary
    # writing have succeeded.
    TEMP_CORRECTED_PANEL.replace(INPUT_PANEL)

    correction_audit = pd.DataFrame(audit_records)
    correction_audit.to_csv(OUTPUT_AUDIT, index=False)

    print(f"Confirmed rows corrected: {len(correction_audit):,}")
    print(f"Correction factor applied: {EXPECTED_CORRECTION_FACTOR:.0f}")
    print(f"Price fields corrected: {', '.join(PRICE_COLUMNS)}")
    print("Volume fields changed: 0")
    print(f"Backup created: {BACKUP_PANEL}")
    print(f"Correction audit: {OUTPUT_AUDIT}")
    print(f"Corrected panel: {INPUT_PANEL}")

    print("\nCorrected rows:")
    print(
        correction_audit[
            [
                "Ticker",
                "Date",
                "Original_Adj_Close",
                "Corrected_Adj_Close",
            ]
        ].to_string(index=False)
    )

    print("\nIntegrity checks passed:")
    print("- Every current value matched its audited original value.")
    print("- Exactly four confirmed rows were corrected.")
    print("- Open, High, Low, Close and Adjusted Close used the same factor.")
    print("- Volume remained unchanged.")
    print("- The uncorrected panel was preserved in a one-time backup.")
    print("=" * 78)


if __name__ == "__main__":
    main()
