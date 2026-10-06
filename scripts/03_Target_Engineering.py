"""
Script 03: Target Variable Engineering
--------------------------------------

Purpose
-------
Calculates one-day forward returns using Adjusted Close on each security's
complete broad trading history, then applies the point-in-time active-universe
membership and constructs the daily cross-sectional percentile target.

The order is critical:

1. Calculate forward returns on complete ticker histories.
2. Verify that each return spans exactly one retained market trading date.
3. Apply the point-in-time active-universe membership.
4. Remove observations without an observable next-day return.
5. Rank realised forward returns within each active daily cross-section.

Inputs
------
data/interim/02_universe/JSE_Broad_Trading_Panel.csv
data/interim/02_universe/JSE_Active_Universe_Membership.csv
outputs/audits/02_universe/JSE_Market_Date_Audit.csv

Data outputs
------------
data/interim/03_target/JSE_Broad_Trading_Panel_With_Returns.csv
data/interim/03_target/JSE_Active_Universe_Panel_With_Returns.csv
data/interim/03_target/JSE_Panel_Target_Engineered.csv

Audit outputs
-------------
outputs/audits/03_target/JSE_Target_Daily_Diagnostics.csv
outputs/audits/03_target/JSE_Target_Tie_Diagnostics.csv

Run from the project root
-------------------------
python .\\scripts\\03_Target_Engineering.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCRIPT_02_DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "02_universe"
)

SCRIPT_02_AUDIT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "audits"
    / "02_universe"
)

DATA_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "03_target"
)

AUDIT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "audits"
    / "03_target"
)


INPUT_BROAD_TRADING_PANEL = (
    SCRIPT_02_DATA_DIRECTORY
    / "JSE_Broad_Trading_Panel.csv"
)

INPUT_MEMBERSHIP = (
    SCRIPT_02_DATA_DIRECTORY
    / "JSE_Active_Universe_Membership.csv"
)

INPUT_MARKET_DATE_AUDIT = (
    SCRIPT_02_AUDIT_DIRECTORY
    / "JSE_Market_Date_Audit.csv"
)


OUTPUT_BROAD_WITH_RETURNS = (
    DATA_OUTPUT_DIRECTORY
    / "JSE_Broad_Trading_Panel_With_Returns.csv"
)

OUTPUT_ACTIVE_WITH_RETURNS = (
    DATA_OUTPUT_DIRECTORY
    / "JSE_Active_Universe_Panel_With_Returns.csv"
)

OUTPUT_TARGET_PANEL = (
    DATA_OUTPUT_DIRECTORY
    / "JSE_Panel_Target_Engineered.csv"
)

OUTPUT_DAILY_DIAGNOSTICS = (
    AUDIT_OUTPUT_DIRECTORY
    / "JSE_Target_Daily_Diagnostics.csv"
)

OUTPUT_TIE_DIAGNOSTICS = (
    AUDIT_OUTPUT_DIRECTORY
    / "JSE_Target_Tie_Diagnostics.csv"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

# A return larger than 500% in absolute magnitude is treated as a likely
# unresolved mechanical price-scale problem. The current corrected panel
# should remain well inside this threshold.
MAX_ABSOLUTE_FORWARD_RETURN = 5.0

BROAD_REQUIRED_COLUMNS = {
    "Date",
    "Ticker",
    "Adj Close",
    "Close",
    "Volume",
}

MEMBERSHIP_REQUIRED_COLUMNS = {
    "Active_Quarter",
    "Ticker",
    "Selection_Quarter",
    "Market_Quarter_End",
    "Selection_Rolling_VT_63D",
    "Liquidity_Rank",
    "Membership_Status",
    "In_Active_Universe",
}

MARKET_AUDIT_REQUIRED_COLUMNS = {
    "Date",
    "Is_Market_Trading_Date",
}


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def relative_path(path: Path) -> str:
    """Return a readable path relative to the project root."""

    return str(
        path.relative_to(
            PROJECT_ROOT
        )
    )


def validate_required_files() -> None:
    """Confirm that all Script 02 inputs exist."""

    required_files = [
        INPUT_BROAD_TRADING_PANEL,
        INPUT_MEMBERSHIP,
        INPUT_MARKET_DATE_AUDIT,
    ]

    missing_files = [
        str(path.resolve())
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "The following required input files are missing:\n- "
            + "\n- ".join(
                missing_files
            )
        )


def validate_broad_panel(
    panel: pd.DataFrame,
) -> None:
    """Validate the complete broad trading panel."""

    missing_columns = BROAD_REQUIRED_COLUMNS.difference(
        panel.columns
    )

    if missing_columns:
        raise ValueError(
            "The broad trading panel is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if panel.empty:
        raise ValueError(
            "The broad trading panel contains no observations."
        )

    duplicate_count = int(
        panel.duplicated(
            ["Date", "Ticker"]
        ).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"The broad trading panel contains {duplicate_count:,} "
            "duplicate Date-Ticker observations."
        )

    required_nonmissing = [
        "Date",
        "Ticker",
        "Adj Close",
    ]

    missing_values = int(
        panel[
            required_nonmissing
        ].isna().sum().sum()
    )

    if missing_values > 0:
        raise ValueError(
            f"The broad trading panel contains {missing_values:,} "
            "missing required values."
        )

    if panel["Adj Close"].le(0).any():
        raise ValueError(
            "Adjusted Close contains non-positive values."
        )


def validate_membership(
    membership: pd.DataFrame,
) -> None:
    """Validate the point-in-time active-universe membership table."""

    missing_columns = MEMBERSHIP_REQUIRED_COLUMNS.difference(
        membership.columns
    )

    if missing_columns:
        raise ValueError(
            "The membership file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if membership.empty:
        raise ValueError(
            "The active-universe membership file contains no records."
        )

    duplicate_count = int(
        membership.duplicated(
            [
                "Active_Quarter",
                "Ticker",
            ]
        ).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"The membership file contains {duplicate_count:,} duplicate "
            "Active_Quarter-Ticker records."
        )

    if not membership[
        "In_Active_Universe"
    ].eq(
        True
    ).all():
        raise ValueError(
            "At least one membership record is not marked as active."
        )


def validate_market_date_audit(
    audit: pd.DataFrame,
) -> None:
    """Validate the retained market-date audit."""

    missing_columns = MARKET_AUDIT_REQUIRED_COLUMNS.difference(
        audit.columns
    )

    if missing_columns:
        raise ValueError(
            "The market-date audit is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if audit["Date"].duplicated().any():
        raise ValueError(
            "The market-date audit contains duplicate dates."
        )


# ============================================================
# 4. MAIN PIPELINE
# ============================================================

def main() -> None:
    print("=" * 74)
    print("SCRIPT 03: TARGET VARIABLE ENGINEERING")
    print("=" * 74)

    validate_required_files()

    DATA_OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 4.1 Load inputs
    # --------------------------------------------------------

    broad_panel = pd.read_csv(
        INPUT_BROAD_TRADING_PANEL
    )

    membership = pd.read_csv(
        INPUT_MEMBERSHIP
    )

    market_date_audit = pd.read_csv(
        INPUT_MARKET_DATE_AUDIT
    )

    broad_panel["Date"] = pd.to_datetime(
        broad_panel["Date"],
        errors="raise",
    )

    market_date_audit["Date"] = pd.to_datetime(
        market_date_audit["Date"],
        errors="raise",
    )

    numeric_columns = [
        "Adj Close",
        "Close",
        "Volume",
    ]

    for column in numeric_columns:
        broad_panel[column] = pd.to_numeric(
            broad_panel[column],
            errors="raise",
        )

    validate_broad_panel(
        broad_panel
    )

    validate_membership(
        membership
    )

    validate_market_date_audit(
        market_date_audit
    )

    broad_row_count = len(
        broad_panel
    )

    # --------------------------------------------------------
    # 4.2 Prepare retained market-date index
    # --------------------------------------------------------

    retained_market_dates = (
        market_date_audit.loc[
            market_date_audit[
                "Is_Market_Trading_Date"
            ].eq(
                True
            ),
            "Date",
        ]
        .sort_values()
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    if retained_market_dates.empty:
        raise ValueError(
            "The market-date audit contains no retained trading dates."
        )

    market_date_positions = pd.Series(
        np.arange(
            len(retained_market_dates),
            dtype=int,
        ),
        index=retained_market_dates,
    )

    # --------------------------------------------------------
    # 4.3 Prepare broad panel and membership keys
    # --------------------------------------------------------

    broad_panel = (
        broad_panel.sort_values(
            ["Ticker", "Date"]
        )
        .reset_index(
            drop=True
        )
    )

    broad_panel["Active_Quarter"] = (
        broad_panel[
            "Date"
        ]
        .dt.to_period(
            "Q"
        )
        .astype(
            str
        )
    )

    membership["Active_Quarter"] = (
        membership[
            "Active_Quarter"
        ].astype(
            str
        )
    )

    if "Selection_Quarter" in membership.columns:
        membership["Selection_Quarter"] = (
            membership[
                "Selection_Quarter"
            ].astype(
                str
            )
        )

    # --------------------------------------------------------
    # 4.4 Calculate forward returns on complete histories
    # --------------------------------------------------------

    grouped = broad_panel.groupby(
        "Ticker",
        sort=False,
    )

    broad_panel["Next_Date"] = grouped[
        "Date"
    ].shift(
        -1
    )

    broad_panel["Next_Adj_Close"] = grouped[
        "Adj Close"
    ].shift(
        -1
    )

    broad_panel[
        "Current_Market_Date_Position"
    ] = broad_panel[
        "Date"
    ].map(
        market_date_positions
    )

    broad_panel[
        "Next_Market_Date_Position"
    ] = broad_panel[
        "Next_Date"
    ].map(
        market_date_positions
    )

    if broad_panel[
        "Current_Market_Date_Position"
    ].isna().any():
        raise ValueError(
            "At least one broad-panel date is not present among the "
            "retained market trading dates."
        )

    broad_panel[
        "Trading_Day_Gap_To_Next"
    ] = (
        broad_panel[
            "Next_Market_Date_Position"
        ]
        - broad_panel[
            "Current_Market_Date_Position"
        ]
    )

    non_final_observations = broad_panel[
        "Next_Date"
    ].notna()

    invalid_gap_mask = (
        non_final_observations
        & broad_panel[
            "Trading_Day_Gap_To_Next"
        ].ne(
            1
        )
    )

    if invalid_gap_mask.any():
        invalid_count = int(
            invalid_gap_mask.sum()
        )

        invalid_example = broad_panel.loc[
            invalid_gap_mask,
            [
                "Ticker",
                "Date",
                "Next_Date",
                "Trading_Day_Gap_To_Next",
            ],
        ].head(
            10
        )

        raise ValueError(
            f"{invalid_count:,} observations have non-consecutive "
            "forward-return links.\n\nExamples:\n"
            f"{invalid_example.to_string(index=False)}"
        )

    broad_panel["Forward_Return"] = (
        broad_panel[
            "Next_Adj_Close"
        ]
        / broad_panel[
            "Adj Close"
        ]
        - 1.0
    )

    finite_forward_returns = broad_panel[
        "Forward_Return"
    ].dropna()

    if not np.isfinite(
        finite_forward_returns.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "The calculated forward returns contain non-finite values."
        )

    extreme_return_mask = (
        broad_panel[
            "Forward_Return"
        ].abs()
        > MAX_ABSOLUTE_FORWARD_RETURN
    )

    if extreme_return_mask.any():
        extreme_count = int(
            extreme_return_mask.sum()
        )

        extreme_examples = broad_panel.loc[
            extreme_return_mask,
            [
                "Ticker",
                "Date",
                "Adj Close",
                "Next_Date",
                "Next_Adj_Close",
                "Forward_Return",
            ],
        ].sort_values(
            "Forward_Return"
        )

        raise ValueError(
            f"{extreme_count:,} forward returns exceed the "
            f"{MAX_ABSOLUTE_FORWARD_RETURN:.0%} absolute audit threshold. "
            "This may indicate an unresolved price-scale error.\n\n"
            f"{extreme_examples.head(20).to_string(index=False)}"
        )

    # Remove temporary mapping columns before saving.
    broad_panel = broad_panel.drop(
        columns=[
            "Current_Market_Date_Position",
            "Next_Market_Date_Position",
        ]
    )

    broad_panel.to_csv(
        OUTPUT_BROAD_WITH_RETURNS,
        index=False,
    )

    # --------------------------------------------------------
    # 4.5 Apply point-in-time active-universe membership
    # --------------------------------------------------------

    membership_columns = [
        "Active_Quarter",
        "Ticker",
        "Selection_Quarter",
        "Market_Quarter_End",
        "Selection_Rolling_VT_63D",
        "Liquidity_Rank",
        "Membership_Status",
        "In_Active_Universe",
    ]

    active_panel = broad_panel.merge(
        membership[
            membership_columns
        ],
        on=[
            "Active_Quarter",
            "Ticker",
        ],
        how="inner",
        validate="many_to_one",
    )

    active_panel = (
        active_panel.sort_values(
            ["Date", "Ticker"]
        )
        .reset_index(
            drop=True
        )
    )

    active_row_count_before_target_filter = len(
        active_panel
    )

    if active_panel.duplicated(
        ["Date", "Ticker"]
    ).any():
        raise AssertionError(
            "Duplicate Date-Ticker observations were created when "
            "active-universe membership was applied."
        )

    active_panel.to_csv(
        OUTPUT_ACTIVE_WITH_RETURNS,
        index=False,
    )

    # --------------------------------------------------------
    # 4.6 Remove rows without an observable next-day return
    # --------------------------------------------------------

    target_panel = active_panel.dropna(
        subset=[
            "Forward_Return"
        ]
    ).copy()

    rows_without_forward_return = (
        active_row_count_before_target_filter
        - len(
            target_panel
        )
    )

    if target_panel.empty:
        raise ValueError(
            "No observations remain after removing rows without "
            "observable forward returns."
        )

    # --------------------------------------------------------
    # 4.7 Construct the daily percentile target
    # --------------------------------------------------------

    target_panel["N_t"] = (
        target_panel
        .groupby(
            "Date"
        )[
            "Ticker"
        ]
        .transform(
            "count"
        )
    )

    target_panel[
        "Daily_Return_Rank"
    ] = (
        target_panel
        .groupby(
            "Date"
        )[
            "Forward_Return"
        ]
        .rank(
            method="average",
            ascending=True,
        )
    )

    target_panel[
        "Target_Y"
    ] = np.where(
        target_panel[
            "N_t"
        ].gt(
            1
        ),
        (
            target_panel[
                "Daily_Return_Rank"
            ]
            - 1
        )
        / (
            target_panel[
                "N_t"
            ]
            - 1
        ),
        0.5,
    )

    target_panel = (
        target_panel.sort_values(
            ["Date", "Ticker"]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # 4.8 Construct daily diagnostics
    # --------------------------------------------------------

    daily_diagnostics = (
        target_panel
        .groupby(
            "Date",
            as_index=False,
        )
        .agg(
            Active_Securities=(
                "Ticker",
                "count",
            ),
            Unique_Forward_Returns=(
                "Forward_Return",
                "nunique",
            ),
            Min_Forward_Return=(
                "Forward_Return",
                "min",
            ),
            Median_Forward_Return=(
                "Forward_Return",
                "median",
            ),
            Max_Forward_Return=(
                "Forward_Return",
                "max",
            ),
            Mean_Target_Y=(
                "Target_Y",
                "mean",
            ),
            Std_Target_Y=(
                "Target_Y",
                "std",
            ),
            Min_Target_Y=(
                "Target_Y",
                "min",
            ),
            Max_Target_Y=(
                "Target_Y",
                "max",
            ),
        )
        .sort_values(
            "Date"
        )
        .reset_index(
            drop=True
        )
    )

    daily_diagnostics[
        "Tied_Returns_Present"
    ] = (
        daily_diagnostics[
            "Unique_Forward_Returns"
        ]
        < daily_diagnostics[
            "Active_Securities"
        ]
    )

    # --------------------------------------------------------
    # 4.9 Construct tie diagnostics
    # --------------------------------------------------------

    tie_source = target_panel.assign(
        Is_Zero_Return=np.isclose(
            target_panel[
                "Forward_Return"
            ].to_numpy(
                dtype=float
            ),
            0.0,
            rtol=0.0,
            atol=1e-15,
        )
    )

    tie_diagnostics = (
        tie_source
        .groupby(
            "Date",
            as_index=False,
        )
        .agg(
            Active_Securities=(
                "Ticker",
                "count",
            ),
            Unique_Forward_Returns=(
                "Forward_Return",
                "nunique",
            ),
            Zero_Return_Observations=(
                "Is_Zero_Return",
                "sum",
            ),
        )
        .sort_values(
            "Date"
        )
        .reset_index(
            drop=True
        )
    )

    tie_diagnostics[
        "Tied_Observations"
    ] = (
        tie_diagnostics[
            "Active_Securities"
        ]
        - tie_diagnostics[
            "Unique_Forward_Returns"
        ]
    )

    tie_diagnostics[
        "Tied_Zero_Return_Observations"
    ] = np.where(
        tie_diagnostics[
            "Zero_Return_Observations"
        ].gt(
            1
        ),
        tie_diagnostics[
            "Zero_Return_Observations"
        ],
        0,
    )

    tie_diagnostics[
        "Any_Ties"
    ] = tie_diagnostics[
        "Tied_Observations"
    ].gt(
        0
    )

    tie_diagnostics[
        "Any_Zero_Return_Ties"
    ] = tie_diagnostics[
        "Tied_Zero_Return_Observations"
    ].gt(
        1
    )

    # --------------------------------------------------------
    # 4.10 Final integrity checks
    # --------------------------------------------------------

    if target_panel.duplicated(
        ["Date", "Ticker"]
    ).any():
        raise AssertionError(
            "Duplicate Date-Ticker observations exist in the target panel."
        )

    required_target_columns = [
        "Date",
        "Ticker",
        "Forward_Return",
        "N_t",
        "Daily_Return_Rank",
        "Target_Y",
    ]

    if target_panel[
        required_target_columns
    ].isna().any().any():
        raise AssertionError(
            "The final target panel contains missing required values."
        )

    if not target_panel[
        "Target_Y"
    ].between(
        0.0,
        1.0,
        inclusive="both",
    ).all():
        raise AssertionError(
            "Target values outside [0, 1] were detected."
        )

    invalid_daily_target_means = daily_diagnostics.loc[
        ~np.isclose(
            daily_diagnostics[
                "Mean_Target_Y"
            ].to_numpy(
                dtype=float
            ),
            0.5,
            rtol=0.0,
            atol=1e-12,
        )
    ]

    if not invalid_daily_target_means.empty:
        raise AssertionError(
            "At least one daily target mean differs from 0.5."
        )

    if target_panel[
        "N_t"
    ].le(
        1
    ).any():
        raise AssertionError(
            "A daily target cross-section contains fewer than two securities."
        )

    # --------------------------------------------------------
    # 4.11 Save target and audit outputs
    # --------------------------------------------------------

    target_panel.to_csv(
        OUTPUT_TARGET_PANEL,
        index=False,
    )

    daily_diagnostics.to_csv(
        OUTPUT_DAILY_DIAGNOSTICS,
        index=False,
    )

    tie_diagnostics.to_csv(
        OUTPUT_TIE_DIAGNOSTICS,
        index=False,
    )

    # --------------------------------------------------------
    # 4.12 Terminal summary
    # --------------------------------------------------------

    target_date_count = target_panel[
        "Date"
    ].nunique()

    target_daily_sizes = (
        target_panel
        .groupby(
            "Date"
        )[
            "Ticker"
        ]
        .count()
    )

    days_with_ties = int(
        tie_diagnostics[
            "Any_Ties"
        ].sum()
    )

    days_with_zero_return_ties = int(
        tie_diagnostics[
            "Any_Zero_Return_Ties"
        ].sum()
    )

    print(
        "\n"
        + "=" * 74
    )

    print(
        "SCRIPT 03 EMPIRICAL OUTPUTS"
    )

    print(
        "=" * 74
    )

    print(
        f"Broad trading-panel rows loaded: "
        f"{broad_row_count:,}"
    )

    print(
        "Active-universe rows before dropping "
        "non-observable forward returns: "
        f"{active_row_count_before_target_filter:,}"
    )

    print(
        "Rows dropped due to unobservable next-day return: "
        f"{rows_without_forward_return:,}"
    )

    print(
        "Final target-engineered observations: "
        f"{len(target_panel):,}"
    )

    print(
        "Target-engineered trading dates: "
        f"{target_date_count:,}"
    )

    print(
        "Target-panel start date: "
        f"{target_panel['Date'].min().date()}"
    )

    print(
        "Target-panel end date: "
        f"{target_panel['Date'].max().date()}"
    )

    print(
        "Average cross-sectional size (N_t): "
        f"{target_daily_sizes.mean():.2f}"
    )

    print(
        "Minimum/maximum cross-sectional size (N_t): "
        f"{target_daily_sizes.min():,}/"
        f"{target_daily_sizes.max():,}"
    )

    print(
        "Days with any tied forward returns: "
        f"{days_with_ties:,}"
    )

    print(
        "Days with tied zero-return observations: "
        f"{days_with_zero_return_ties:,}"
    )

    print(
        "Minimum observed forward return: "
        f"{target_panel['Forward_Return'].min():.6f}"
    )

    print(
        "Maximum observed forward return: "
        f"{target_panel['Forward_Return'].max():.6f}"
    )

    print(
        "Target minimum value: "
        f"{target_panel['Target_Y'].min():.4f}"
    )

    print(
        "Target maximum value: "
        f"{target_panel['Target_Y'].max():.4f}"
    )

    print(
        "Target mean: "
        f"{target_panel['Target_Y'].mean():.4f}"
    )

    print(
        "Target standard deviation: "
        f"{target_panel['Target_Y'].std():.4f}"
    )

    print(
        "\nIntegrity checks passed:"
    )

    print(
        "- Forward returns were calculated on complete ticker histories "
        "before active-universe filtering."
    )

    print(
        "- Forward-return links were verified to span exactly one "
        "retained market trading date."
    )

    print(
        "- No unresolved extreme mechanical price-scale returns "
        "were detected."
    )

    print(
        "- Cross-sectional ranking was calculated only within the "
        "point-in-time active universe."
    )

    print(
        "- Every daily target mean equals 0.5."
    )

    print(
        "- Target values are bounded between 0 and 1."
    )

    print(
        "\nOutput files:"
    )

    output_files = [
        OUTPUT_BROAD_WITH_RETURNS,
        OUTPUT_ACTIVE_WITH_RETURNS,
        OUTPUT_TARGET_PANEL,
        OUTPUT_DAILY_DIAGNOSTICS,
        OUTPUT_TIE_DIAGNOSTICS,
    ]

    for output_file in output_files:
        print(
            f"- {relative_path(output_file)}"
        )

    print(
        "=" * 74
    )


if __name__ == "__main__":
    main()