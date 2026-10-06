"""
Script 09B: Alternative OLS-vs-XGBoost Comparison (21-Day Target)
-----------------------------------------------------------------

Purpose
-------
Replicates the full paired comparison of Script 09 using the alternative
21-day-target prediction files from Scripts 07B and 08B.

The only methodological difference from Script 09 is:

    NW lag = max( floor[4(T/100)^(2/9)],  21 )

With overlapping 21-day forward return labels the IC difference series
inherits ~20-lag overlap-induced autocorrelation, so the minimum lag
floor is raised from 1 to 21 to ensure valid HAC inference.

Inputs
------
results/07_ols_alt/JSE_Alt_OLS_Walk_Forward_Predictions.csv
results/07_ols_alt/JSE_Alt_OLS_Daily_Performance.csv
results/08_xgboost_alt/JSE_Alt_XGB_Walk_Forward_Predictions.csv
results/08_xgboost_alt/JSE_Alt_XGB_Daily_Performance.csv

Outputs
-------
results/09B_comparison_alt/JSE_Alt_Model_Comparison_Paired_Predictions.csv
results/09B_comparison_alt/JSE_Alt_Model_Comparison_Daily_Paired.csv
results/09B_comparison_alt/JSE_Alt_Model_Comparison_Quarterly.csv
results/09B_comparison_alt/JSE_Alt_Model_Comparison_Overall.csv
results/09B_comparison_alt/JSE_Alt_Model_Comparison_Newey_West_IC_Test.csv

Run from the project root
-------------------------
python .\\scripts\\09B_Alt_Comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, spearmanr


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OLS_DIRECTORY = PROJECT_ROOT / "results" / "07_ols_alt"
XGB_DIRECTORY = PROJECT_ROOT / "results" / "08_xgboost_alt"
RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "09B_comparison_alt"
AUDIT_DIRECTORY = PROJECT_ROOT / "outputs" / "audits" / "09B_comparison_alt"

INPUT_OLS_PREDICTIONS = (
    OLS_DIRECTORY / "JSE_Alt_OLS_Walk_Forward_Predictions.csv"
)
INPUT_OLS_DAILY = (
    OLS_DIRECTORY / "JSE_Alt_OLS_Daily_Performance.csv"
)
INPUT_XGB_PREDICTIONS = (
    XGB_DIRECTORY / "JSE_Alt_XGB_Walk_Forward_Predictions.csv"
)
INPUT_XGB_DAILY = (
    XGB_DIRECTORY / "JSE_Alt_XGB_Daily_Performance.csv"
)

OUTPUT_PAIRED_PREDICTIONS = (
    RESULTS_DIRECTORY / "JSE_Alt_Model_Comparison_Paired_Predictions.csv"
)
OUTPUT_DAILY_COMPARISON = (
    RESULTS_DIRECTORY / "JSE_Alt_Model_Comparison_Daily_Paired.csv"
)
OUTPUT_QUARTERLY_COMPARISON = (
    RESULTS_DIRECTORY / "JSE_Alt_Model_Comparison_Quarterly.csv"
)
OUTPUT_OVERALL_COMPARISON = (
    RESULTS_DIRECTORY / "JSE_Alt_Model_Comparison_Overall.csv"
)
OUTPUT_NW_TEST = (
    RESULTS_DIRECTORY / "JSE_Alt_Model_Comparison_Newey_West_IC_Test.csv"
)
OUTPUT_SUMMARY = (
    AUDIT_DIRECTORY / "JSE_Alt_Model_Comparison_Summary.csv"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

OLS_SCORE_COLUMN = "OLS_Raw_Score"
XGB_SCORE_COLUMN = "XGB_Raw_Score"
OLS_PERCENTILE_COLUMN = "OLS_Predicted_Percentile"
XGB_PERCENTILE_COLUMN = "XGB_Predicted_Percentile"

MINIMUM_DAILY_CROSS_SECTION = 5
TOP_BOTTOM_FRACTION = 0.25
SIGNIFICANCE_LEVEL = 0.05

# *** KEY CHANGE: floor raised from 1 to 21 to handle overlapping labels ***
MINIMUM_NW_LAG = 21


# ============================================================
# 3. HELPERS
# ============================================================

def safe_spearman_ic(
    predicted: pd.Series,
    target: pd.Series,
) -> float:
    if len(predicted) < MINIMUM_DAILY_CROSS_SECTION:
        return np.nan
    if predicted.nunique(dropna=True) <= 1:
        return 0.0
    if target.nunique(dropna=True) <= 1:
        return 0.0
    result = spearmanr(predicted, target)
    stat = float(result.statistic)
    return stat if np.isfinite(stat) else 0.0


def quartile_hit_rate(
    group: pd.DataFrame,
    score_col: str,
) -> tuple[float, float, int]:
    n = len(group)
    q = max(1, int(np.ceil(n * TOP_BOTTOM_FRACTION)))
    pred_sorted = group.sort_values(
        [score_col, "Ticker"], ascending=[False, True], kind="mergesort"
    )
    real_sorted = group.sort_values(
        ["Target_Y", "Ticker"], ascending=[False, True], kind="mergesort"
    )
    top_hr = len(
        set(pred_sorted.head(q)["Ticker"]) & set(real_sorted.head(q)["Ticker"])
    ) / q
    bot_hr = len(
        set(pred_sorted.tail(q)["Ticker"]) & set(real_sorted.tail(q)["Ticker"])
    ) / q
    return float(top_hr), float(bot_hr), q


# ============================================================
# 4. NEWEY-WEST INFERENCE
# ============================================================

def automatic_nw_lag(n: int) -> int:
    """
    NW lag = max( floor[4(T/100)^(2/9)],  MINIMUM_NW_LAG )

    The minimum floor is raised to 21 for 21-day overlapping returns.
    """
    raw_lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    return max(MINIMUM_NW_LAG, min(raw_lag, n - 1))


def newey_west_ic_test(ic_differences: pd.Series) -> dict:
    """
    H0: E[d_t] <= 0   H1: E[d_t] > 0  (one-sided, XGB > OLS)
    """
    values = ic_differences.dropna().to_numpy(dtype=float)
    n = len(values)
    if n < 2:
        raise ValueError("Too few observations for Newey-West test.")

    lag = automatic_nw_lag(n)
    intercept = np.ones((n, 1))
    model = sm.OLS(values, intercept).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": lag, "use_correction": True},
    )

    mean_diff = float(model.params[0])
    nw_se = float(model.bse[0])

    if not np.isfinite(nw_se) or nw_se <= 0:
        raise ValueError("Invalid Newey-West standard error.")

    t_stat = mean_diff / nw_se
    p_one = float(norm.sf(t_stat))
    p_two = float(2.0 * norm.sf(abs(t_stat)))
    cv = float(norm.ppf(1.0 - SIGNIFICANCE_LEVEL / 2.0))

    return {
        "Sample_Size_T": n,
        "NW_Lag_q": lag,
        "Minimum_NW_Lag_Floor": MINIMUM_NW_LAG,
        "Mean_IC_Difference": mean_diff,
        "Newey_West_SE": nw_se,
        "Test_Statistic": t_stat,
        "One_Sided_P_Value": p_one,
        "Two_Sided_P_Value": p_two,
        "Critical_Value_95pct": cv,
        "Reject_H0_One_Sided_5pct": (
            t_stat > cv
        ),
        "Interpretation": (
            "XGBoost significantly outperforms OLS (alpha=5%, one-sided)"
            if t_stat > cv
            else "Cannot reject H0: no significant XGB advantage"
        ),
    }


# ============================================================
# 5. PAIRED DAILY PERFORMANCE
# ============================================================

def build_paired_daily(paired: pd.DataFrame) -> pd.DataFrame:
    records = []
    for date, group in paired.groupby("Date", sort=True):
        ols_ic = safe_spearman_ic(group[OLS_SCORE_COLUMN], group["Target_Y"])
        xgb_ic = safe_spearman_ic(group[XGB_SCORE_COLUMN], group["Target_Y"])

        ols_errors = group[OLS_SCORE_COLUMN] - group["Target_Y"]
        xgb_errors = group[XGB_SCORE_COLUMN] - group["Target_Y"]

        ols_top, ols_bot, _ = quartile_hit_rate(group, OLS_SCORE_COLUMN)
        xgb_top, xgb_bot, _ = quartile_hit_rate(group, XGB_SCORE_COLUMN)

        records.append(
            {
                "Date": date,
                "Outer_Window_ID": group["Outer_Window_ID"].iloc[0],
                "Test_Quarter": group["Test_Quarter"].iloc[0],
                "Model_Eligible_Securities": len(group),
                "OLS_Daily_IC": ols_ic,
                "XGB_Daily_IC": xgb_ic,
                "IC_Difference_XGB_Minus_OLS": xgb_ic - ols_ic,
                "OLS_Daily_MAE": float(np.mean(np.abs(ols_errors))),
                "XGB_Daily_MAE": float(np.mean(np.abs(xgb_errors))),
                "OLS_Daily_RMSE": float(np.sqrt(np.mean(ols_errors**2))),
                "XGB_Daily_RMSE": float(np.sqrt(np.mean(xgb_errors**2))),
                "OLS_Top_Quartile_Hit_Rate": ols_top,
                "XGB_Top_Quartile_Hit_Rate": xgb_top,
                "OLS_Bottom_Quartile_Hit_Rate": ols_bot,
                "XGB_Bottom_Quartile_Hit_Rate": xgb_bot,
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values("Date")
        .reset_index(drop=True)
    )


def quarterly_summary(daily: pd.DataFrame) -> pd.DataFrame:
    records = []
    for tq, group in daily.groupby("Test_Quarter", sort=True):
        ic_diff = group["IC_Difference_XGB_Minus_OLS"].dropna()
        records.append(
            {
                "Test_Quarter": tq,
                "Outer_Window_ID": group["Outer_Window_ID"].iloc[0],
                "Test_Dates": len(group),
                "OLS_Mean_Daily_IC": float(group["OLS_Daily_IC"].mean()),
                "XGB_Mean_Daily_IC": float(group["XGB_Daily_IC"].mean()),
                "Mean_IC_Difference": float(ic_diff.mean()),
                "OLS_Mean_Top_Hit": float(group["OLS_Top_Quartile_Hit_Rate"].mean()),
                "XGB_Mean_Top_Hit": float(group["XGB_Top_Quartile_Hit_Rate"].mean()),
                "OLS_Mean_Bot_Hit": float(group["OLS_Bottom_Quartile_Hit_Rate"].mean()),
                "XGB_Mean_Bot_Hit": float(group["XGB_Bottom_Quartile_Hit_Rate"].mean()),
                "XGB_Wins_Days": int(
                    group["IC_Difference_XGB_Minus_OLS"].gt(0).sum()
                ),
            }
        )
    return pd.DataFrame(records)


def overall_summary(
    daily: pd.DataFrame,
    paired: pd.DataFrame,
    nw_result: dict,
) -> dict:
    ic_diff = daily["IC_Difference_XGB_Minus_OLS"].dropna()
    ols_ic = daily["OLS_Daily_IC"].dropna()
    xgb_ic = daily["XGB_Daily_IC"].dropna()

    ols_errors = paired[OLS_SCORE_COLUMN] - paired["Target_Y"]
    xgb_errors = paired[XGB_SCORE_COLUMN] - paired["Target_Y"]

    return {
        "Setup": "Alternative_21D",
        "Total_Paired_Observations": len(paired),
        "Total_Paired_Dates": paired["Date"].nunique(),
        "OLS_Mean_Daily_IC": float(ols_ic.mean()),
        "OLS_Median_Daily_IC": float(ols_ic.median()),
        "OLS_IC_Std": float(ols_ic.std(ddof=1)),
        "OLS_ICIR": float(ols_ic.mean() / ols_ic.std(ddof=1)),
        "OLS_Positive_IC_Pct": float(ols_ic.gt(0).mean() * 100),
        "OLS_Observation_RMSE": float(np.sqrt(np.mean(ols_errors**2))),
        "OLS_Observation_MAE": float(np.mean(np.abs(ols_errors))),
        "XGB_Mean_Daily_IC": float(xgb_ic.mean()),
        "XGB_Median_Daily_IC": float(xgb_ic.median()),
        "XGB_IC_Std": float(xgb_ic.std(ddof=1)),
        "XGB_ICIR": float(xgb_ic.mean() / xgb_ic.std(ddof=1)),
        "XGB_Positive_IC_Pct": float(xgb_ic.gt(0).mean() * 100),
        "XGB_Observation_RMSE": float(np.sqrt(np.mean(xgb_errors**2))),
        "XGB_Observation_MAE": float(np.mean(np.abs(xgb_errors))),
        "Mean_IC_Difference_XGB_Minus_OLS": float(ic_diff.mean()),
        "NW_Lag_q": nw_result["NW_Lag_q"],
        "NW_T_Stat": nw_result["Test_Statistic"],
        "NW_One_Sided_P_Value": nw_result["One_Sided_P_Value"],
        "Reject_H0": nw_result["Reject_H0_One_Sided_5pct"],
        "Interpretation": nw_result["Interpretation"],
    }


# ============================================================
# 6. MAIN PIPELINE
# ============================================================

def main() -> None:
    print("=" * 78)
    print("SCRIPT 09B: ALT OLS VS XGB COMPARISON  (21-DAY TARGET)")
    print("=" * 78)

    for path in [
        INPUT_OLS_PREDICTIONS, INPUT_OLS_DAILY,
        INPUT_XGB_PREDICTIONS, INPUT_XGB_DAILY,
    ]:
        if not path.exists():
            sys.exit(f"ERROR: Required file missing: {path}")

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    AUDIT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # Load
    # -------------------------------------------------------
    ols_pred = pd.read_csv(INPUT_OLS_PREDICTIONS)
    ols_daily = pd.read_csv(INPUT_OLS_DAILY)
    xgb_pred = pd.read_csv(INPUT_XGB_PREDICTIONS)
    xgb_daily = pd.read_csv(INPUT_XGB_DAILY)

    for df in [ols_pred, xgb_pred]:
        df["Date"] = pd.to_datetime(df["Date"])

    for df in [ols_daily, xgb_daily]:
        df["Date"] = pd.to_datetime(df["Date"])

    print(f"  OLS predictions: {len(ols_pred):,} rows")
    print(f"  XGB predictions: {len(xgb_pred):,} rows")

    # -------------------------------------------------------
    # Pair on identical Date-Ticker sample
    # -------------------------------------------------------
    merge_keys = ["Date", "Ticker", "Outer_Window_ID", "Test_Quarter"]
    shared_cols = ["Target_Y", "Forward_Return", "N_t", "Model_N_t"]

    paired = ols_pred[merge_keys + shared_cols + [OLS_SCORE_COLUMN, OLS_PERCENTILE_COLUMN]].merge(
        xgb_pred[merge_keys + [XGB_SCORE_COLUMN, XGB_PERCENTILE_COLUMN]],
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )

    n_ols = len(ols_pred)
    n_xgb = len(xgb_pred)
    n_paired = len(paired)

    print(f"  Paired observations: {n_paired:,}  "
          f"(dropped: {n_ols - n_paired} OLS, {n_xgb - n_paired} XGB)")

    if n_paired == 0:
        sys.exit("ERROR: No paired observations. Cannot proceed.")

    paired = paired.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # -------------------------------------------------------
    # Build paired daily performance
    # -------------------------------------------------------
    paired_daily = build_paired_daily(paired)

    T = int(paired_daily["IC_Difference_XGB_Minus_OLS"].dropna().count())
    nw_lag = automatic_nw_lag(T)
    print(f"  Newey-West: T={T}, q={nw_lag}  (minimum floor = {MINIMUM_NW_LAG})")

    # -------------------------------------------------------
    # Newey-West test
    # -------------------------------------------------------
    nw_result = newey_west_ic_test(
        paired_daily["IC_Difference_XGB_Minus_OLS"]
    )

    print(f"\n  Mean IC difference (XGB - OLS): {nw_result['Mean_IC_Difference']:+.6f}")
    print(f"  NW t-statistic:                 {nw_result['Test_Statistic']:+.4f}")
    print(f"  One-sided p-value:              {nw_result['One_Sided_P_Value']:.4f}")
    print(f"  {nw_result['Interpretation']}")

    # -------------------------------------------------------
    # Quarterly and overall summaries
    # -------------------------------------------------------
    quarterly = quarterly_summary(paired_daily)
    overall = overall_summary(paired_daily, paired, nw_result)

    # -------------------------------------------------------
    # Save
    # -------------------------------------------------------
    paired.to_csv(OUTPUT_PAIRED_PREDICTIONS, index=False)
    paired_daily.to_csv(OUTPUT_DAILY_COMPARISON, index=False)
    quarterly.to_csv(OUTPUT_QUARTERLY_COMPARISON, index=False)
    pd.DataFrame([overall]).to_csv(OUTPUT_OVERALL_COMPARISON, index=False)
    pd.DataFrame([nw_result]).to_csv(OUTPUT_NW_TEST, index=False)

    summary_records = [
        {"Parameter": k, "Value": str(v)} for k, v in overall.items()
    ]
    pd.DataFrame(summary_records).to_csv(OUTPUT_SUMMARY, index=False)

    print(f"\n  Outputs saved to: {RESULTS_DIRECTORY.relative_to(PROJECT_ROOT)}")

    print("\n  === Quarterly IC Summary ===")
    for _, row in quarterly.iterrows():
        print(
            f"    {row['Test_Quarter']}:  "
            f"OLS={row['OLS_Mean_Daily_IC']:+.5f}  "
            f"XGB={row['XGB_Mean_Daily_IC']:+.5f}  "
            f"Diff={row['Mean_IC_Difference']:+.5f}"
        )

    print("\n  Script 09B complete.")


if __name__ == "__main__":
    main()
