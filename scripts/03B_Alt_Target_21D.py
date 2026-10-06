"""
Script 03B: Alternative 21-Day Forward Return Target Engineering
----------------------------------------------------------------

Purpose
-------
Replicates the target-engineering logic of Script 03 using a 21-trading-day
forward-return horizon instead of 1-day.  All other pipeline choices (universe
membership, cross-sectional percentile ranking, mid-rank tie-breaking) are
identical to Script 03.

This script writes to a SEPARATE output directory so that Script 03's outputs
are never overwritten.

Inputs  (same as Script 03)
------
data/interim/02_universe/JSE_Broad_Trading_Panel.csv
data/interim/02_universe/JSE_Active_Universe_Membership.csv
outputs/audits/02_universe/JSE_Market_Date_Audit.csv

Output
------
data/interim/03B_alt_target/JSE_Alt_Panel_Target_21D.csv

Run from the project root
-------------------------
python .\\scripts\\03B_Alt_Target_21D.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_BROAD_TRADING_PANEL = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "02_universe"
    / "JSE_Broad_Trading_Panel.csv"
)

INPUT_MEMBERSHIP = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "02_universe"
    / "JSE_Active_Universe_Membership.csv"
)

INPUT_MARKET_DATE_AUDIT = (
    PROJECT_ROOT
    / "outputs"
    / "audits"
    / "02_universe"
    / "JSE_Market_Date_Audit.csv"
)

DATA_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "03B_alt_target"
)

OUTPUT_TARGET_PANEL = (
    DATA_OUTPUT_DIRECTORY
    / "JSE_Alt_Panel_Target_21D.csv"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

# The only parameter that differs from Script 03.
FORWARD_RETURN_HORIZON = 21

# A 21-day return spanning more than 1000 % is a likely data error.
MAX_ABSOLUTE_FORWARD_RETURN = 10.0


# ============================================================
# 3. HELPERS
# ============================================================

def validate_required_files() -> None:
    """Confirm that all required inputs exist."""

    required = [
        INPUT_BROAD_TRADING_PANEL,
        INPUT_MEMBERSHIP,
        INPUT_MARKET_DATE_AUDIT,
    ]

    missing = [
        str(p.resolve())
        for p in required
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required input files are missing:\n- "
            + "\n- ".join(missing)
        )


# ============================================================
# 4. MAIN PIPELINE
# ============================================================

def main() -> None:
    print("=" * 74)
    print("SCRIPT 03B: ALTERNATIVE 21-DAY TARGET ENGINEERING")
    print("=" * 74)

    validate_required_files()

    DATA_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 4.1 Load inputs
    # --------------------------------------------------------

    broad = pd.read_csv(INPUT_BROAD_TRADING_PANEL)
    membership = pd.read_csv(INPUT_MEMBERSHIP)
    audit = pd.read_csv(INPUT_MARKET_DATE_AUDIT)

    broad["Date"] = pd.to_datetime(broad["Date"], errors="raise")
    audit["Date"] = pd.to_datetime(audit["Date"], errors="raise")

    for col in ["Adj Close", "Close", "Volume"]:
        broad[col] = pd.to_numeric(broad[col], errors="raise")

    broad = (
        broad
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    print(f"  Broad panel loaded:  {len(broad):>10,} rows")
    print(f"  Membership loaded:   {len(membership):>10,} rows")

    # --------------------------------------------------------
    # 4.2 Build retained market-date index
    # --------------------------------------------------------

    retained_dates = (
        audit.loc[
            audit["Is_Market_Trading_Date"].eq(True),
            "Date",
        ]
        .sort_values()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if retained_dates.empty:
        raise ValueError("No retained market trading dates found.")

    date_positions = pd.Series(
        np.arange(len(retained_dates), dtype=int),
        index=retained_dates,
    )

    # --------------------------------------------------------
    # 4.3 Compute 21-day forward prices and dates
    # --------------------------------------------------------

    grouped = broad.groupby("Ticker", sort=False)

    # shift(-21): price and date 21 trading positions later
    broad["Next_Date_21D"] = grouped["Date"].shift(-FORWARD_RETURN_HORIZON)
    broad["Next_Adj_Close_21D"] = grouped["Adj Close"].shift(-FORWARD_RETURN_HORIZON)

    broad["Current_Pos"] = broad["Date"].map(date_positions)
    broad["Next_Pos_21D"] = broad["Next_Date_21D"].map(date_positions)

    if broad["Current_Pos"].isna().any():
        raise ValueError(
            "At least one broad-panel date is absent from the "
            "retained market-date index."
        )

    broad["Gap_21D"] = broad["Next_Pos_21D"] - broad["Current_Pos"]

    # Validate: for rows that have a 21D next date, the gap must be 21.
    has_next = broad["Next_Date_21D"].notna()
    invalid_gap = has_next & broad["Gap_21D"].ne(FORWARD_RETURN_HORIZON)

    if invalid_gap.any():
        n_bad = int(invalid_gap.sum())
        raise ValueError(
            f"{n_bad:,} observations have a non-{FORWARD_RETURN_HORIZON} "
            "trading-day gap to their 21-day forward date. "
            "This indicates missing dates in the broad panel."
        )

    # --------------------------------------------------------
    # 4.4 Compute 21-day forward return
    # --------------------------------------------------------

    broad["Forward_Return"] = (
        broad["Next_Adj_Close_21D"] / broad["Adj Close"] - 1.0
    )

    extreme = broad["Forward_Return"].abs().gt(MAX_ABSOLUTE_FORWARD_RETURN)
    if extreme.any():
        n_ext = int(extreme.sum())
        print(
            f"  WARNING: {n_ext:,} observations exceed the "
            f"{MAX_ABSOLUTE_FORWARD_RETURN:.0%} absolute return cap and "
            "will be dropped."
        )
        broad = broad.loc[~extreme].copy()

    # --------------------------------------------------------
    # 4.5 Apply point-in-time active-universe membership
    # --------------------------------------------------------

    membership["Active_Quarter"] = membership["Active_Quarter"].astype(str)
    broad["Active_Quarter"] = (
        broad["Date"].dt.to_period("Q").astype(str)
    )

    active_keys = membership.loc[
        membership["In_Active_Universe"].eq(True),
        ["Active_Quarter", "Ticker"],
    ].drop_duplicates()

    active_panel = broad.merge(
        active_keys,
        on=["Active_Quarter", "Ticker"],
        how="inner",
    )

    print(f"  Rows after universe filter: {len(active_panel):>10,}")

    # --------------------------------------------------------
    # 4.6 Drop observations with no 21-day-ahead return
    # --------------------------------------------------------

    active_panel = active_panel.loc[
        active_panel["Forward_Return"].notna()
    ].copy()

    print(f"  Rows with valid 21D return: {len(active_panel):>10,}")

    # --------------------------------------------------------
    # 4.7 Cross-sectional daily percentile rank (same as Script 03)
    # --------------------------------------------------------

    # N_t: number of active securities on each date
    active_panel["N_t"] = (
        active_panel
        .groupby("Date")["Ticker"]
        .transform("nunique")
        .astype(int)
    )

    def rank_group(group: pd.DataFrame) -> pd.Series:
        n = len(group)
        if n == 1:
            return pd.Series([0.5], index=group.index)
        # mid-rank: average rank of ties, then rescale to [0, 1]
        ranks = group["Forward_Return"].rank(
            method="average",
            ascending=True,
        )
        return (ranks - 1.0) / (n - 1.0)

    active_panel["Target_Y"] = (
        active_panel
        .groupby("Date", group_keys=False)
        .apply(rank_group)
    )

    # Verify Target_Y is in [0, 1]
    if not active_panel["Target_Y"].between(0, 1, inclusive="both").all():
        raise ValueError("Target_Y contains values outside [0, 1].")

    # --------------------------------------------------------
    # 4.8 Rename forward-return date column to Next_Date
    #     (used by Script 06B for the leakage audit)
    # --------------------------------------------------------

    active_panel = active_panel.rename(
        columns={"Next_Date_21D": "Next_Date"}
    )

    # --------------------------------------------------------
    # 4.9 Select and write output columns
    # --------------------------------------------------------

    output_cols = [
        "Date",
        "Ticker",
        "Forward_Return",
        "Next_Date",
        "N_t",
        "Target_Y",
    ]

    output = (
        active_panel[output_cols]
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )

    output.to_csv(OUTPUT_TARGET_PANEL, index=False)

    print(f"\n  Output rows:  {len(output):>10,}")
    print(f"  Unique dates: {output['Date'].nunique():>10,}")
    print(
        f"  Date range:   "
        f"{output['Date'].min().date()} to {output['Date'].max().date()}"
    )
    print(
        f"  Target_Y mean:  {output['Target_Y'].mean():.4f}  "
        f"(expected ≈ 0.5)"
    )
    print(f"\n  Saved: {OUTPUT_TARGET_PANEL.relative_to(PROJECT_ROOT)}")
    print("\n  Script 03B complete.")


if __name__ == "__main__":
    main()
