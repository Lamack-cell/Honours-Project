"""
Script 06B: Alternative Walk-Forward Validation Setup (21-Day Purge)
--------------------------------------------------------------------

Purpose
-------
Creates the expanding-window walk-forward schedule for the alternative
experimental setup.  Differs from Script 06 in two ways only:

    1. TARGET_HORIZON_PURGE_DATES = 21   (was 1 in Script 06)
    2. Reads the 21-day Target_Y from Script 03B and splices it into
       the existing feature panel, replacing the 1-day Target_Y.

All window logic, inner-fold construction, and leakage checks are
identical to Script 06.

Inputs
------
data/processed/05_features/JSE_Panel_Features_Engineered.csv   (features)
data/interim/03B_alt_target/JSE_Alt_Panel_Target_21D.csv        (alt target)

Core outputs
------------
data/processed/06B_alt_validation/JSE_Alt_Master_Model_Panel.csv
data/processed/06B_alt_validation/JSE_Alt_Walk_Forward_Windows.csv
data/processed/06B_alt_validation/JSE_Alt_Inner_Validation_Windows.csv

Audit outputs
-------------
outputs/audits/06B_alt_validation/JSE_Alt_Walk_Forward_Summary.csv

Run from the project root
-------------------------
python .\\scripts\\06B_Alt_Walk_Forward_Setup.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FEATURE_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "05_features"
    / "JSE_Panel_Features_Engineered.csv"
)

INPUT_ALT_TARGET = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "03B_alt_target"
    / "JSE_Alt_Panel_Target_21D.csv"
)

DATA_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "06B_alt_validation"
)

AUDIT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "audits"
    / "06B_alt_validation"
)

OUTPUT_MASTER_PANEL = (
    DATA_OUTPUT_DIRECTORY
    / "JSE_Alt_Master_Model_Panel.csv"
)

OUTPUT_OUTER_WINDOWS = (
    DATA_OUTPUT_DIRECTORY
    / "JSE_Alt_Walk_Forward_Windows.csv"
)

OUTPUT_INNER_WINDOWS = (
    DATA_OUTPUT_DIRECTORY
    / "JSE_Alt_Inner_Validation_Windows.csv"
)

OUTPUT_SUMMARY = (
    AUDIT_DIRECTORY
    / "JSE_Alt_Walk_Forward_Summary.csv"
)


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

PRIMARY_FEATURES = [
    "Mom_1M",
    "Mom_3M",
    "Mom_6M",
    "Vol_21D",
    "RSI_14",
    "SMA_50_Ratio",
    "SMA_200_Ratio",
    "Volume_Shock_21D",
    "Value_Traded_Shock_63D",
]

TARGET_COLUMN = "Target_Y"

INITIAL_TEST_QUARTER = pd.Period("2024Q1", freq="Q")

# *** KEY CHANGE vs Script 06 ***
TARGET_HORIZON_PURGE_DATES = 21

INNER_VALIDATION_FOLDS = 3
INNER_VALIDATION_DATES_PER_FOLD = 126
MINIMUM_INNER_TRAINING_DATES = 504


# ============================================================
# 3. HELPERS
# ============================================================

def validate_required_files() -> None:
    required = [INPUT_FEATURE_PANEL, INPUT_ALT_TARGET]
    missing = [str(p.resolve()) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required input files are missing:\n- " + "\n- ".join(missing)
        )


def count_rows_between(
    panel: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> int:
    mask = panel["Date"].between(start, end, inclusive="both")
    return int(mask.sum())


def count_dates_between(
    dates: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> int:
    return int(((dates >= start) & (dates <= end)).sum())


# ============================================================
# 4. OUTER WALK-FORWARD WINDOWS
# ============================================================

def construct_outer_windows(
    master_panel: pd.DataFrame,
    model_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Quarterly expanding outer windows — identical logic to Script 06."""

    observed_periods = pd.PeriodIndex(
        model_dates.to_period("Q").unique()
    ).sort_values()

    test_periods = [
        p for p in observed_periods if p >= INITIAL_TEST_QUARTER
    ]

    if not test_periods:
        raise ValueError(
            f"No model dates exist in or after {INITIAL_TEST_QUARTER}."
        )

    max_period = observed_periods.max()
    records: list[dict] = []

    for window_number, test_period in enumerate(test_periods, start=1):
        mask = model_dates.to_period("Q") == test_period
        test_dates = model_dates[mask]

        if len(test_dates) == 0:
            continue

        test_start = pd.Timestamp(test_dates.min())
        test_end = pd.Timestamp(test_dates.max())

        test_start_position = int(model_dates.get_loc(test_start))

        minimum_required = TARGET_HORIZON_PURGE_DATES + 1
        if test_start_position < minimum_required:
            raise ValueError(
                f"Insufficient pre-test dates for {test_period}."
            )

        purge_position = test_start_position - 1
        train_end_position = purge_position - TARGET_HORIZON_PURGE_DATES

        train_start = pd.Timestamp(model_dates.min())
        train_end = pd.Timestamp(model_dates[train_end_position])
        purge_date = pd.Timestamp(model_dates[purge_position])

        if not (train_end < purge_date < test_start):
            raise AssertionError(
                f"Date ordering invalid for {test_period}."
            )

        records.append(
            {
                "Outer_Window_ID": f"OW_{window_number:02d}",
                "Test_Quarter": str(test_period),
                "Train_Start": train_start,
                "Train_End": train_end,
                "Purged_Boundary_Date": purge_date,
                "Test_Start": test_start,
                "Test_End": test_end,
                "Purged_Dates": TARGET_HORIZON_PURGE_DATES,
                "Train_Dates": count_dates_between(
                    model_dates, train_start, train_end
                ),
                "Train_Observations": count_rows_between(
                    master_panel, train_start, train_end
                ),
                "Test_Dates": len(test_dates),
                "Test_Observations": count_rows_between(
                    master_panel, test_start, test_end
                ),
                "Is_Complete_Calendar_Quarter": (
                    test_period < max_period
                ),
            }
        )

    outer_windows = (
        pd.DataFrame(records)
        .sort_values("Test_Start")
        .reset_index(drop=True)
    )

    if outer_windows.empty:
        raise ValueError("No outer walk-forward windows were constructed.")

    assert (
        outer_windows["Train_End"] < outer_windows["Purged_Boundary_Date"]
    ).all(), "A training window reaches its purge date."

    assert (
        outer_windows["Purged_Boundary_Date"] < outer_windows["Test_Start"]
    ).all(), "A purge date is not before its test start."

    # No overlapping test windows
    for i in range(1, len(outer_windows)):
        prev_end = pd.Timestamp(outer_windows.loc[i - 1, "Test_End"])
        curr_start = pd.Timestamp(outer_windows.loc[i, "Test_Start"])
        if curr_start <= prev_end:
            raise AssertionError("Outer test windows overlap.")

    return outer_windows


# ============================================================
# 5. INNER VALIDATION WINDOWS
# ============================================================

def construct_inner_windows(
    master_panel: pd.DataFrame,
    model_dates: pd.DatetimeIndex,
    outer_windows: pd.DataFrame,
) -> pd.DataFrame:
    """Nested chronological validation folds — identical logic to Script 06."""

    total_val_dates = (
        INNER_VALIDATION_FOLDS * INNER_VALIDATION_DATES_PER_FOLD
    )
    minimum_required = (
        MINIMUM_INNER_TRAINING_DATES
        + total_val_dates
        + TARGET_HORIZON_PURGE_DATES
    )

    records: list[dict] = []

    for outer_row in outer_windows.itertuples(index=False):
        outer_train_start = pd.Timestamp(outer_row.Train_Start)
        outer_train_end = pd.Timestamp(outer_row.Train_End)

        outer_train_dates = model_dates[
            (model_dates >= outer_train_start)
            & (model_dates <= outer_train_end)
        ]

        if len(outer_train_dates) < minimum_required:
            raise ValueError(
                f"{outer_row.Outer_Window_ID} has only "
                f"{len(outer_train_dates):,} training dates; "
                f"{minimum_required:,} required."
            )

        first_val_pos = len(outer_train_dates) - total_val_dates

        for fold_number in range(1, INNER_VALIDATION_FOLDS + 1):
            val_start_pos = (
                first_val_pos
                + (fold_number - 1) * INNER_VALIDATION_DATES_PER_FOLD
            )
            val_end_pos = val_start_pos + INNER_VALIDATION_DATES_PER_FOLD - 1

            purge_pos = val_start_pos - 1
            inner_train_end_pos = purge_pos - TARGET_HORIZON_PURGE_DATES

            if inner_train_end_pos < 0:
                raise ValueError(
                    f"Inner training window has no dates for "
                    f"{outer_row.Outer_Window_ID} fold {fold_number}."
                )

            inner_train_start = pd.Timestamp(outer_train_dates.min())
            inner_train_end = pd.Timestamp(
                outer_train_dates[inner_train_end_pos]
            )
            inner_purge_date = pd.Timestamp(
                outer_train_dates[purge_pos]
            )
            val_start = pd.Timestamp(outer_train_dates[val_start_pos])
            val_end = pd.Timestamp(outer_train_dates[val_end_pos])

            assert (
                inner_train_end < inner_purge_date < val_start <= val_end
            ), f"Inner date ordering invalid for fold {fold_number}."

            records.append(
                {
                    "Outer_Window_ID": outer_row.Outer_Window_ID,
                    "Test_Quarter": outer_row.Test_Quarter,
                    "Inner_Fold_ID": (
                        f"{outer_row.Outer_Window_ID}_IF_{fold_number}"
                    ),
                    "Inner_Fold_Number": fold_number,
                    "Inner_Train_Start": inner_train_start,
                    "Inner_Train_End": inner_train_end,
                    "Inner_Purged_Boundary_Date": inner_purge_date,
                    "Validation_Start": val_start,
                    "Validation_End": val_end,
                    "Inner_Purged_Dates": TARGET_HORIZON_PURGE_DATES,
                    "Inner_Train_Dates": inner_train_end_pos + 1,
                    "Inner_Train_Observations": count_rows_between(
                        master_panel, inner_train_start, inner_train_end
                    ),
                    "Validation_Dates": (
                        val_end_pos - val_start_pos + 1
                    ),
                    "Validation_Observations": count_rows_between(
                        master_panel, val_start, val_end
                    ),
                }
            )

    inner_windows = pd.DataFrame(records)

    expected = len(outer_windows) * INNER_VALIDATION_FOLDS
    if len(inner_windows) != expected:
        raise AssertionError(
            f"Expected {expected} inner folds, got {len(inner_windows)}."
        )

    assert (
        inner_windows["Inner_Train_End"]
        < inner_windows["Inner_Purged_Boundary_Date"]
    ).all()

    assert (
        inner_windows["Inner_Purged_Boundary_Date"]
        < inner_windows["Validation_Start"]
    ).all()

    return inner_windows


# ============================================================
# 6. MAIN PIPELINE
# ============================================================

def main() -> None:
    print("=" * 78)
    print("SCRIPT 06B: ALTERNATIVE WALK-FORWARD SETUP  (21-DATE PURGE)")
    print("=" * 78)

    validate_required_files()

    DATA_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    AUDIT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 6.1 Load feature panel (9 features + 1-day Target_Y)
    # --------------------------------------------------------

    feature_panel = pd.read_csv(INPUT_FEATURE_PANEL)
    feature_panel["Date"] = pd.to_datetime(
        feature_panel["Date"], errors="raise"
    )
    feature_panel = (
        feature_panel
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )
    print(f"  Feature panel loaded: {len(feature_panel):>10,} rows")

    # --------------------------------------------------------
    # 6.2 Load alternative 21-day target
    # --------------------------------------------------------

    alt_target = pd.read_csv(INPUT_ALT_TARGET)
    alt_target["Date"] = pd.to_datetime(alt_target["Date"], errors="raise")
    alt_target["Next_Date"] = pd.to_datetime(
        alt_target["Next_Date"], errors="coerce"
    )
    print(f"  Alt target loaded:    {len(alt_target):>10,} rows")

    # --------------------------------------------------------
    # 6.3 Splice alt target into feature panel
    #     Drop the 1-day Target_Y, Forward_Return, Next_Date
    #     and replace with 21-day versions from 03B.
    # --------------------------------------------------------

    drop_cols = [
        c for c in ["Target_Y", "Forward_Return", "Next_Date", "N_t", "Model_N_t"]
        if c in feature_panel.columns
    ]
    feature_stripped = feature_panel.drop(columns=drop_cols)

    # Merge on Date + Ticker (inner join: keeps only rows where both
    # the feature and the 21-day target exist)
    master_panel = feature_stripped.merge(
        alt_target[["Date", "Ticker", "Forward_Return", "Next_Date", "N_t", "Target_Y"]],
        on=["Date", "Ticker"],
        how="inner",
        validate="one_to_one",
    )

    print(f"  Master panel (joined): {len(master_panel):>9,} rows")

    # Recompute Model_N_t to match the merged cross-section.
    master_panel["Model_N_t"] = (
        master_panel
        .groupby("Date")["Ticker"]
        .transform("nunique")
        .astype(int)
    )

    # Basic validation
    required_cols = {
        "Date",
        "Ticker",
        "Target_Y",
        "Forward_Return",
        "N_t",
        "Model_N_t",
        *PRIMARY_FEATURES,
    }
    missing_cols = required_cols - set(master_panel.columns)
    if missing_cols:
        raise ValueError(
            f"Master panel missing columns: {sorted(missing_cols)}"
        )

    if master_panel.duplicated(["Date", "Ticker"]).any():
        raise AssertionError(
            "Duplicate Date-Ticker rows in master panel."
        )

    if not master_panel["Target_Y"].between(0, 1, inclusive="both").all():
        raise ValueError("Target_Y outside [0, 1].")

    master_panel = (
        master_panel
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 6.4 Build model-date index
    # --------------------------------------------------------

    model_dates = pd.DatetimeIndex(
        master_panel["Date"].drop_duplicates().sort_values()
    )

    print(
        f"  Model dates:  {len(model_dates):>6,}  "
        f"({model_dates.min().date()} to {model_dates.max().date()})"
    )

    # --------------------------------------------------------
    # 6.5 Construct outer and inner schedules
    # --------------------------------------------------------

    outer_windows = construct_outer_windows(
        master_panel=master_panel,
        model_dates=model_dates,
    )

    inner_windows = construct_inner_windows(
        master_panel=master_panel,
        model_dates=model_dates,
        outer_windows=outer_windows,
    )

    print(f"  Outer windows: {len(outer_windows)}")
    print(f"  Inner folds:   {len(inner_windows)}")

    # --------------------------------------------------------
    # 6.6 Target-boundary leakage check using Next_Date
    # --------------------------------------------------------

    if "Next_Date" in master_panel.columns:
        for outer_row in outer_windows.itertuples(index=False):
            train_rows = master_panel.loc[
                master_panel["Date"].between(
                    pd.Timestamp(outer_row.Train_Start),
                    pd.Timestamp(outer_row.Train_End),
                    inclusive="both",
                )
            ]
            leakage = int(
                (
                    train_rows["Next_Date"]
                    >= pd.Timestamp(outer_row.Test_Start)
                ).sum()
            )
            if leakage > 0:
                raise AssertionError(
                    f"{leakage:,} leakage rows in {outer_row.Outer_Window_ID}."
                )

        for inner_row in inner_windows.itertuples(index=False):
            inner_train_rows = master_panel.loc[
                master_panel["Date"].between(
                    pd.Timestamp(inner_row.Inner_Train_Start),
                    pd.Timestamp(inner_row.Inner_Train_End),
                    inclusive="both",
                )
            ]
            leakage = int(
                (
                    inner_train_rows["Next_Date"]
                    >= pd.Timestamp(inner_row.Validation_Start)
                ).sum()
            )
            if leakage > 0:
                raise AssertionError(
                    f"{leakage:,} inner leakage rows in "
                    f"{inner_row.Inner_Fold_ID}."
                )

        print("  Leakage audit: PASSED (all outer and inner windows clean)")

    # --------------------------------------------------------
    # 6.7 Coverage check
    # --------------------------------------------------------

    first_test_start = pd.Timestamp(outer_windows.iloc[0]["Test_Start"])
    total_test_obs = int(outer_windows["Test_Observations"].sum())
    total_test_dates = int(outer_windows["Test_Dates"].sum())

    expected_obs = int(
        (master_panel["Date"] >= first_test_start).sum()
    )
    expected_dates = int(
        master_panel.loc[
            master_panel["Date"] >= first_test_start, "Date"
        ].nunique()
    )

    if total_test_obs != expected_obs:
        raise AssertionError(
            f"Coverage mismatch: windows cover {total_test_obs:,} obs, "
            f"panel has {expected_obs:,} post-test-start obs."
        )
    if total_test_dates != expected_dates:
        raise AssertionError(
            f"Coverage mismatch: windows cover {total_test_dates:,} dates, "
            f"panel has {expected_dates:,} post-test-start dates."
        )

    # --------------------------------------------------------
    # 6.8 Write outputs
    # --------------------------------------------------------

    master_panel.to_csv(OUTPUT_MASTER_PANEL, index=False)
    outer_windows.to_csv(OUTPUT_OUTER_WINDOWS, index=False)
    inner_windows.to_csv(OUTPUT_INNER_WINDOWS, index=False)

    summary = pd.DataFrame(
        [
            {
                "Parameter": "Forward_Return_Horizon_Days",
                "Value": 21,
            },
            {
                "Parameter": "TARGET_HORIZON_PURGE_DATES",
                "Value": TARGET_HORIZON_PURGE_DATES,
            },
            {
                "Parameter": "INNER_VALIDATION_FOLDS",
                "Value": INNER_VALIDATION_FOLDS,
            },
            {
                "Parameter": "INNER_VALIDATION_DATES_PER_FOLD",
                "Value": INNER_VALIDATION_DATES_PER_FOLD,
            },
            {
                "Parameter": "MINIMUM_INNER_TRAINING_DATES",
                "Value": MINIMUM_INNER_TRAINING_DATES,
            },
            {
                "Parameter": "Master_Panel_Rows",
                "Value": len(master_panel),
            },
            {
                "Parameter": "Master_Panel_Dates",
                "Value": len(model_dates),
            },
            {
                "Parameter": "Outer_Windows",
                "Value": len(outer_windows),
            },
            {
                "Parameter": "Inner_Folds",
                "Value": len(inner_windows),
            },
            {
                "Parameter": "Total_Test_Observations",
                "Value": total_test_obs,
            },
            {
                "Parameter": "Total_Test_Dates",
                "Value": total_test_dates,
            },
        ]
    )
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    print(f"\n  Master panel saved:     {OUTPUT_MASTER_PANEL.relative_to(PROJECT_ROOT)}")
    print(f"  Outer windows saved:    {OUTPUT_OUTER_WINDOWS.relative_to(PROJECT_ROOT)}")
    print(f"  Inner windows saved:    {OUTPUT_INNER_WINDOWS.relative_to(PROJECT_ROOT)}")
    print("\n  Script 06B complete.")

    # Print outer window table
    print("\n  Outer Window Schedule:")
    print(
        outer_windows[[
            "Outer_Window_ID",
            "Test_Quarter",
            "Train_End",
            "Purged_Boundary_Date",
            "Test_Start",
            "Test_End",
            "Train_Observations",
            "Test_Dates",
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
