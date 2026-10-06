"""
Script 03: Target Variable Engineering
--------------------------------------

Purpose
-------
Calculates one-day forward returns using Adjusted Close on the complete
broad trading history, then applies the point-in-time active-universe
membership and constructs the cross-sectional percentile target.

This order is critical:
- forward returns are calculated BEFORE filtering to the active universe;
- active-universe filtering is then applied;
- and the target is ranked only within the active cross-section on each date.

Inputs
------
1. JSE_Broad_Trading_Panel.csv
2. JSE_Active_Universe_Membership.csv
3. JSE_Market_Date_Audit.csv

Outputs
-------
1. JSE_Broad_Trading_Panel_With_Returns.csv
2. JSE_Active_Universe_Panel_With_Returns.csv
3. JSE_Panel_Target_Engineered.csv
4. JSE_Target_Daily_Diagnostics.csv
5. JSE_Target_Tie_Diagnostics.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_BROAD_TRADING_PANEL = Path("JSE_Broad_Trading_Panel.csv")
INPUT_MEMBERSHIP = Path("JSE_Active_Universe_Membership.csv")
INPUT_MARKET_DATE_AUDIT = Path("JSE_Market_Date_Audit.csv")

OUTPUT_BROAD_WITH_RETURNS = Path("JSE_Broad_Trading_Panel_With_Returns.csv")
OUTPUT_ACTIVE_WITH_RETURNS = Path("JSE_Active_Universe_Panel_With_Returns.csv")
OUTPUT_TARGET_PANEL = Path("JSE_Panel_Target_Engineered.csv")
OUTPUT_DAILY_DIAGNOSTICS = Path("JSE_Target_Daily_Diagnostics.csv")
OUTPUT_TIE_DIAGNOSTICS = Path("JSE_Target_Tie_Diagnostics.csv")


# ============================================================
# 2. INPUT VALIDATION
# ============================================================

def validate_required_files() -> None:
    required_files = [
        INPUT_BROAD_TRADING_PANEL,
        INPUT_MEMBERSHIP,
        INPUT_MARKET_DATE_AUDIT,
    ]

    missing_files = [str(file) for file in required_files if not file.exists()]

    if missing_files:
        raise FileNotFoundError(
            "The following required input files are missing:\n- "
            + "\n- ".join(missing_files)
        )


def validate_broad_panel(df: pd.DataFrame) -> None:
    required_columns = {
        "Date",
        "Ticker",
        "Adj Close",
        "Close",
        "Volume",
        "Selection_Quarter",
    }

    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "Broad trading panel is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df.duplicated(["Date", "Ticker"]).any():
        raise ValueError(
            "Duplicate Date-Ticker observations detected in "
            "JSE_Broad_Trading_Panel.csv."
        )

    if df["Adj Close"].isna().any():
        raise ValueError(
            "Adjusted close contains missing values, which is not allowed "
            "for target construction."
        )

    if (df["Adj Close"] <= 0).any():
        raise ValueError(
            "Adjusted close contains non-positive values."
        )


def validate_membership(df: pd.DataFrame) -> None:
    required_columns = {
        "Active_Quarter",
        "Ticker",
        "In_Active_Universe",
    }

    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "Membership file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df.duplicated(["Active_Quarter", "Ticker"]).any():
        raise ValueError(
            "Duplicate Active_Quarter-Ticker pairs detected in "
            "JSE_Active_Universe_Membership.csv."
        )


# ============================================================
# 3. LOAD INPUTS
# ============================================================

validate_required_files()

df_broad = pd.read_csv(INPUT_BROAD_TRADING_PANEL)
df_membership = pd.read_csv(INPUT_MEMBERSHIP)
df_market_dates = pd.read_csv(INPUT_MARKET_DATE_AUDIT)

df_broad["Date"] = pd.to_datetime(df_broad["Date"])
df_market_dates["Date"] = pd.to_datetime(df_market_dates["Date"])

validate_broad_panel(df_broad)
validate_membership(df_membership)

# Keep only actual market trading dates from the audit file
market_dates = (
    df_market_dates.loc[
        df_market_dates["Is_Market_Trading_Date"] == True,
        "Date"
    ]
    .sort_values()
    .reset_index(drop=True)
)

market_date_positions = {date: idx for idx, date in enumerate(market_dates)}

# Prepare broad panel
df_broad = df_broad.sort_values(["Ticker", "Date"]).reset_index(drop=True)

# Prepare active-quarter key for merging
df_broad["Active_Quarter"] = df_broad["Date"].dt.to_period("Q").astype(str)
df_membership["Active_Quarter"] = df_membership["Active_Quarter"].astype(str)


# ============================================================
# 4. CALCULATE FORWARD RETURNS ON COMPLETE TRADING HISTORIES
# ============================================================

# Forward adjusted close and forward date per ticker
df_broad["Next_Date"] = df_broad.groupby("Ticker")["Date"].shift(-1)
df_broad["Next_Adj_Close"] = df_broad.groupby("Ticker")["Adj Close"].shift(-1)

# Verify that returns are being calculated only across consecutive market dates
def compute_gap_to_next(row):
    if pd.isna(row["Next_Date"]):
        return np.nan
    return market_date_positions[row["Next_Date"]] - market_date_positions[row["Date"]]

df_broad["Trading_Day_Gap_To_Next"] = df_broad.apply(compute_gap_to_next, axis=1)

# Any non-final observation should have gap exactly 1
invalid_gap_mask = df_broad["Next_Date"].notna() & (df_broad["Trading_Day_Gap_To_Next"] != 1)

if invalid_gap_mask.any():
    invalid_count = int(invalid_gap_mask.sum())
    raise ValueError(
        f"{invalid_count:,} observations have non-consecutive forward-return "
        "links. Resolve this before continuing."
    )

# One-day forward return
df_broad["Forward_Return"] = (
    df_broad["Next_Adj_Close"] / df_broad["Adj Close"] - 1
)

# Save full broad panel with returns for audit purposes
df_broad.to_csv(OUTPUT_BROAD_WITH_RETURNS, index=False)


# ============================================================
# 5. APPLY POINT-IN-TIME ACTIVE-UNIVERSE MEMBERSHIP
# ============================================================

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

df_active = df_broad.merge(
    df_membership[membership_columns],
    on=["Active_Quarter", "Ticker"],
    how="inner",
    validate="many_to_one"
)

df_active = df_active.sort_values(["Date", "Ticker"]).reset_index(drop=True)

# Save active panel before target engineering
df_active.to_csv(OUTPUT_ACTIVE_WITH_RETURNS, index=False)


# ============================================================
# 6. REMOVE OBSERVATIONS WITHOUT OBSERVABLE NEXT-DAY RETURNS
# ============================================================

# Final date for each ticker has no observable next-day return
df_target = df_active.dropna(subset=["Forward_Return"]).copy()

if df_target.empty:
    raise ValueError(
        "No observations remain after dropping rows without forward returns."
    )


# ============================================================
# 7. CROSS-SECTIONAL RANKING WITHIN THE ACTIVE UNIVERSE
# ============================================================

# Daily cross-sectional count N_t
df_target["N_t"] = df_target.groupby("Date")["Ticker"].transform("count")

# Cross-sectional rank of realised forward returns within each day
# Lowest realised return gets lowest rank; highest realised return gets highest rank
df_target["Daily_Return_Rank"] = df_target.groupby("Date")["Forward_Return"].rank(
    method="average",
    ascending=True
)

# Percentile normalisation
df_target["Target_Y"] = np.where(
    df_target["N_t"] > 1,
    (df_target["Daily_Return_Rank"] - 1) / (df_target["N_t"] - 1),
    0.5
)

# Final sort
df_target = df_target.sort_values(["Date", "Ticker"]).reset_index(drop=True)


# ============================================================
# 8. DIAGNOSTIC TABLES
# ============================================================

# Daily diagnostics
daily_diagnostics = (
    df_target.groupby("Date", as_index=False)
    .agg(
        Active_Securities=("Ticker", "count"),
        Unique_Forward_Returns=("Forward_Return", "nunique"),
        Min_Forward_Return=("Forward_Return", "min"),
        Median_Forward_Return=("Forward_Return", "median"),
        Max_Forward_Return=("Forward_Return", "max"),
        Mean_Target_Y=("Target_Y", "mean"),
        Std_Target_Y=("Target_Y", "std"),
        Min_Target_Y=("Target_Y", "min"),
        Max_Target_Y=("Target_Y", "max"),
    )
)

daily_diagnostics["Tied_Returns_Present"] = (
    daily_diagnostics["Unique_Forward_Returns"]
    < daily_diagnostics["Active_Securities"]
)

# Tie diagnostics
tie_diagnostics = (
    df_target.groupby("Date")
    .apply(
        lambda group: pd.Series({
            "Active_Securities": len(group),
            "Unique_Forward_Returns": group["Forward_Return"].nunique(),
            "Tied_Observations": len(group) - group["Forward_Return"].nunique(),
            "Zero_Return_Observations": int((group["Forward_Return"] == 0).sum()),
            "Tied_Zero_Return_Observations": int((group["Forward_Return"] == 0).sum())
            if int((group["Forward_Return"] == 0).sum()) > 1 else 0,
        })
    )
    .reset_index()
)

tie_diagnostics["Any_Ties"] = tie_diagnostics["Tied_Observations"] > 0
tie_diagnostics["Any_Zero_Return_Ties"] = tie_diagnostics["Tied_Zero_Return_Observations"] > 1

# Save diagnostics
daily_diagnostics.to_csv(OUTPUT_DAILY_DIAGNOSTICS, index=False)
tie_diagnostics.to_csv(OUTPUT_TIE_DIAGNOSTICS, index=False)

# Save final target-engineered panel
df_target.to_csv(OUTPUT_TARGET_PANEL, index=False)


# ============================================================
# 9. EMPIRICAL OUTPUTS
# ============================================================

total_broad_rows = len(df_broad)
total_active_rows_before_drop = len(df_active)
total_target_rows = len(df_target)

dropped_no_forward_return = total_active_rows_before_drop - total_target_rows

total_target_dates = df_target["Date"].nunique()
mean_nt = df_target["N_t"].mean()
min_nt = int(df_target["N_t"].min())
max_nt = int(df_target["N_t"].max())

days_with_any_ties = int(tie_diagnostics["Any_Ties"].sum())
days_with_zero_return_ties = int(tie_diagnostics["Any_Zero_Return_Ties"].sum())

target_min = df_target["Target_Y"].min()
target_max = df_target["Target_Y"].max()
target_mean = df_target["Target_Y"].mean()
target_std = df_target["Target_Y"].std()

print("=" * 74)
print("SCRIPT 03: TARGET VARIABLE ENGINEERING")
print("=" * 74)

print("\n" + "=" * 74)
print("SCRIPT 03 EMPIRICAL OUTPUTS")
print("=" * 74)

print(f"Broad trading-panel rows loaded: {total_broad_rows:,}")
print(f"Active-universe rows before dropping non-observable forward returns: {total_active_rows_before_drop:,}")
print(f"Rows dropped due to unobservable next-day return: {dropped_no_forward_return:,}")
print(f"Final target-engineered observations: {total_target_rows:,}")
print(f"Target-engineered trading dates: {total_target_dates:,}")

if total_target_dates > 0:
    print(f"Target-panel start date: {df_target['Date'].min().date()}")
    print(f"Target-panel end date: {df_target['Date'].max().date()}")

print(f"Average cross-sectional size (N_t): {mean_nt:.2f}")
print(f"Minimum/maximum cross-sectional size (N_t): {min_nt}/{max_nt}")

print(f"Days with any tied forward returns: {days_with_any_ties:,}")
print(f"Days with tied zero-return observations: {days_with_zero_return_ties:,}")

print(f"Target minimum value: {target_min:.4f}")
print(f"Target maximum value: {target_max:.4f}")
print(f"Target mean: {target_mean:.4f}")
print(f"Target standard deviation: {target_std:.4f}")

print("\nIntegrity checks passed:")
print("- Forward returns were calculated on complete ticker histories before active-universe filtering.")
print("- Forward-return links were verified to span exactly one market trading date.")
print("- Cross-sectional target ranking was calculated only within the point-in-time active universe.")
print("- Target values are bounded between 0 and 1 by construction.")

print("\nOutput files:")
print(f"- {OUTPUT_BROAD_WITH_RETURNS}")
print(f"- {OUTPUT_ACTIVE_WITH_RETURNS}")
print(f"- {OUTPUT_TARGET_PANEL}")
print(f"- {OUTPUT_DAILY_DIAGNOSTICS}")
print(f"- {OUTPUT_TIE_DIAGNOSTICS}")
print("=" * 74)