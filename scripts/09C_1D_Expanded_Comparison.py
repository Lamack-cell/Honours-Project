"""
Script 09C: 1-Day Target Model Comparison (OLS Baseline vs 96-Candidate XGBoost)
--------------------------------------------------------------------------------

Purpose
-------
Performs a rigorous head-to-head statistical comparison between the primary Pooled OLS
baseline and the 1-day XGBoost model fitted with the expanded 96-candidate regularisation
grid (searching over reg_lambda in {1.0, 5.0, 10.0}).

Hypothesis Testing:
    Paired Newey-West HAC test on daily IC differences (Delta_t = IC_XGB - IC_OLS)
    using Bartlett kernel with automatic lag selection:
        q = floor( 4 * (T / 100)^(2/9) ) = 5 lags

Inputs
------
results/07_ols/JSE_OLS_Walk_Forward_Predictions.csv
results/07_ols/JSE_OLS_Daily_Performance.csv
results/08C_xgboost_1d_expanded/JSE_1D_Expanded_XGB_Walk_Forward_Predictions.csv
results/08C_xgboost_1d_expanded/JSE_1D_Expanded_XGB_Daily_Performance.csv

Outputs
-------
results/09C_comparison_1d_expanded/JSE_1D_Expanded_Model_Comparison_Paired_Predictions.csv
results/09C_comparison_1d_expanded/JSE_1D_Expanded_Model_Comparison_Daily_Paired.csv
results/09C_comparison_1d_expanded/JSE_1D_Expanded_Model_Comparison_Quarterly.csv
results/09C_comparison_1d_expanded/JSE_1D_Expanded_Model_Comparison_Overall.csv
results/09C_comparison_1d_expanded/JSE_1D_Expanded_Model_Comparison_Newey_West_IC_Test.csv

Run from project root
---------------------
python .\\scripts\\09C_1D_Expanded_Comparison.py
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

OLS_DIRECTORY = PROJECT_ROOT / "results" / "07_ols"
XGB_DIRECTORY = PROJECT_ROOT / "results" / "08C_xgboost_1d_expanded"
RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "09C_comparison_1d_expanded"
AUDIT_DIRECTORY = PROJECT_ROOT / "outputs" / "audits" / "09C_comparison_1d_expanded"

INPUT_OLS_PREDICTIONS = OLS_DIRECTORY / "JSE_OLS_Walk_Forward_Predictions.csv"
INPUT_OLS_DAILY = OLS_DIRECTORY / "JSE_OLS_Daily_Performance.csv"
INPUT_XGB_PREDICTIONS = XGB_DIRECTORY / "JSE_1D_Expanded_XGB_Walk_Forward_Predictions.csv"
INPUT_XGB_DAILY = XGB_DIRECTORY / "JSE_1D_Expanded_XGB_Daily_Performance.csv"

OUTPUT_PAIRED_PREDICTIONS = (
    RESULTS_DIRECTORY / "JSE_1D_Expanded_Model_Comparison_Paired_Predictions.csv"
)
OUTPUT_DAILY_COMPARISON = (
    RESULTS_DIRECTORY / "JSE_1D_Expanded_Model_Comparison_Daily_Paired.csv"
)
OUTPUT_QUARTERLY_COMPARISON = (
    RESULTS_DIRECTORY / "JSE_1D_Expanded_Model_Comparison_Quarterly.csv"
)
OUTPUT_OVERALL_COMPARISON = (
    RESULTS_DIRECTORY / "JSE_1D_Expanded_Model_Comparison_Overall.csv"
)
OUTPUT_NW_TEST = (
    RESULTS_DIRECTORY / "JSE_1D_Expanded_Model_Comparison_Newey_West_IC_Test.csv"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

OLS_SCORE_COLUMN = "OLS_Raw_Score"
XGB_SCORE_COLUMN = "XGB_Raw_Score"
TARGET_COLUMN = "Target_Y"
TOP_BOTTOM_FRACTION = 0.25
ALPHA_LEVEL = 0.05


# ============================================================
# 3. STATISTICAL HELPERS
# ============================================================

def safe_spearman_ic(pred: pd.Series, target: pd.Series) -> float:
    if len(pred) < 5 or pred.nunique(dropna=True) <= 1 or target.nunique(dropna=True) <= 1:
        return 0.0
    r = spearmanr(pred, target)
    stat = float(r.statistic)
    return stat if np.isfinite(stat) else 0.0


def quartile_hit_rate(group: pd.DataFrame, score_col: str) -> tuple[float, float]:
    n = len(group)
    q = max(1, int(np.ceil(n * TOP_BOTTOM_FRACTION)))
    pred_sorted = group.sort_values([score_col, "Ticker"], ascending=[False, True])
    real_sorted = group.sort_values([TARGET_COLUMN, "Ticker"], ascending=[False, True])
    top_hr = len(set(pred_sorted.head(q)["Ticker"]) & set(real_sorted.head(q)["Ticker"])) / q
    bot_hr = len(set(pred_sorted.tail(q)["Ticker"]) & set(real_sorted.tail(q)["Ticker"])) / q
    return float(top_hr), float(bot_hr)


def newey_west_test(diff_series: pd.Series) -> dict:
    clean = diff_series.dropna().to_numpy(dtype=float)
    t_len = len(clean)
    mean_diff = float(np.mean(clean))
    
    # Standard 1-day Newey-West rule
    q = max(1, int(np.floor(4.0 * ((t_len / 100.0) ** (2.0 / 9.0)))))

    X = np.ones((t_len, 1), dtype=float)
    res = sm.OLS(clean, X).fit(cov_type="HAC", cov_kwds={"maxlags": q})
    se = float(res.bse[0])
    stat = mean_diff / se if se > 0 else 0.0
    p_one_sided = float(1.0 - norm.cdf(stat))
    p_two_sided = float(2.0 * (1.0 - norm.cdf(abs(stat))))
    crit = float(norm.ppf(1.0 - ALPHA_LEVEL / 2.0))
    ci_lower = mean_diff - crit * se
    ci_upper = mean_diff + crit * se

    return {
        "Daily_Observations": t_len,
        "Newey_West_Max_Lag": q,
        "Mean_IC_Difference_XGB_Minus_OLS": mean_diff,
        "Newey_West_Standard_Error": se,
        "Normal_Test_Statistic": stat,
        "One_Sided_P_Value_XGB_Superior": p_one_sided,
        "Two_Sided_P_Value": p_two_sided,
        "Confidence_Interval_Lower": ci_lower,
        "Confidence_Interval_Upper": ci_upper,
        "Directional_Decision": "Reject H0 (XGB superior)" if p_one_sided < ALPHA_LEVEL else "Fail to reject H0",
        "Two_Sided_Decision": "Significant difference" if p_two_sided < ALPHA_LEVEL else "No statistically significant difference",
    }


# ============================================================
# 4. MAIN PIPELINE
# ============================================================

def main() -> None:
    print("=" * 78)
    print("SCRIPT 09C: 1-DAY MODEL COMPARISON (POOLED OLS vs 96-CANDIDATE XGBOOST)")
    print("=" * 78)

    for p in [INPUT_OLS_PREDICTIONS, INPUT_XGB_PREDICTIONS]:
        if not p.exists():
            sys.exit(f"ERROR: Required file missing: {p}")

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    AUDIT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    ols_preds = pd.read_csv(INPUT_OLS_PREDICTIONS)
    xgb_preds = pd.read_csv(INPUT_XGB_PREDICTIONS)

    ols_preds["Date"] = pd.to_datetime(ols_preds["Date"])
    xgb_preds["Date"] = pd.to_datetime(xgb_preds["Date"])

    # Merge paired predictions
    paired = pd.merge(
        ols_preds[["Date", "Ticker", "Outer_Window_ID", "Test_Quarter", TARGET_COLUMN, OLS_SCORE_COLUMN, "OLS_Predicted_Percentile"]],
        xgb_preds[["Date", "Ticker", XGB_SCORE_COLUMN, "XGB_Predicted_Percentile"]],
        on=["Date", "Ticker"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    print(f"  Paired observations: {len(paired):,} across {paired['Date'].nunique()} dates")

    # Daily performance
    daily_records = []
    for date, group in paired.groupby("Date", sort=True):
        wid = group["Outer_Window_ID"].iloc[0]
        qtr = group["Test_Quarter"].iloc[0]
        ols_ic = safe_spearman_ic(group[OLS_SCORE_COLUMN], group[TARGET_COLUMN])
        xgb_ic = safe_spearman_ic(group[XGB_SCORE_COLUMN], group[TARGET_COLUMN])
        ols_top, ols_bot = quartile_hit_rate(group, OLS_SCORE_COLUMN)
        xgb_top, xgb_bot = quartile_hit_rate(group, XGB_SCORE_COLUMN)

        daily_records.append(
            {
                "Date": date,
                "Outer_Window_ID": wid,
                "Test_Quarter": qtr,
                "Securities": len(group),
                "OLS_Daily_IC": ols_ic,
                "XGB_Daily_IC": xgb_ic,
                "Daily_IC_Difference_XGB_Minus_OLS": xgb_ic - ols_ic,
                "OLS_Top_Hit_Rate": ols_top,
                "XGB_Top_Hit_Rate": xgb_top,
                "OLS_Bottom_Hit_Rate": ols_bot,
                "XGB_Bottom_Hit_Rate": xgb_bot,
            }
        )
    daily_df = pd.DataFrame(daily_records)

    # Quarterly aggregation
    quarterly_records = []
    for qtr, group in daily_df.groupby("Test_Quarter", sort=False):
        wid = group["Outer_Window_ID"].iloc[0]
        ols_m = float(group["OLS_Daily_IC"].mean())
        xgb_m = float(group["XGB_Daily_IC"].mean())
        ols_s = float(group["OLS_Daily_IC"].std(ddof=1))
        xgb_s = float(group["XGB_Daily_IC"].std(ddof=1))
        quarterly_records.append(
            {
                "Outer_Window_ID": wid,
                "Test_Quarter": qtr,
                "Trading_Dates": len(group),
                "OLS_Mean_Daily_IC": ols_m,
                "XGB_Mean_Daily_IC": xgb_m,
                "IC_Difference_XGB_Minus_OLS": xgb_m - ols_m,
                "OLS_IC_StdDev": ols_s,
                "XGB_IC_StdDev": xgb_s,
                "OLS_ICIR": ols_m / ols_s if ols_s > 0 else np.nan,
                "XGB_ICIR": xgb_m / xgb_s if xgb_s > 0 else np.nan,
                "OLS_Positive_Days_Pct": float(group["OLS_Daily_IC"].gt(0).mean() * 100),
                "XGB_Positive_Days_Pct": float(group["XGB_Daily_IC"].gt(0).mean() * 100),
                "Quarterly_Winner": "XGBoost" if xgb_m > ols_m else "Pooled OLS",
            }
        )
    quarterly_df = pd.DataFrame(quarterly_records)

    # Overall metrics
    ols_mean_ic = float(daily_df["OLS_Daily_IC"].mean())
    xgb_mean_ic = float(daily_df["XGB_Daily_IC"].mean())
    ols_std_ic = float(daily_df["OLS_Daily_IC"].std(ddof=1))
    xgb_std_ic = float(daily_df["XGB_Daily_IC"].std(ddof=1))

    # Observation-level prediction errors for MAE / RMSE
    ols_errors = paired[OLS_SCORE_COLUMN] - paired[TARGET_COLUMN]
    xgb_errors = paired[XGB_SCORE_COLUMN] - paired[TARGET_COLUMN]

    ols_mae = float(np.mean(np.abs(ols_errors)))
    xgb_mae = float(np.mean(np.abs(xgb_errors)))
    ols_rmse = float(np.sqrt(np.mean(ols_errors ** 2)))
    xgb_rmse = float(np.sqrt(np.mean(xgb_errors ** 2)))

    nw_results = newey_west_test(daily_df["Daily_IC_Difference_XGB_Minus_OLS"])

    def _winner_higher(ols_val, xgb_val):
        return "XGBoost" if xgb_val > ols_val else "Pooled OLS"

    def _winner_lower(ols_val, xgb_val):
        return "XGBoost" if xgb_val < ols_val else "Pooled OLS"

    overall_metrics = [
        {"Metric": "Out-of-sample stock-day predictions", "Pooled_OLS": float(len(paired)), "XGBoost": float(len(paired)), "Difference_XGB_Minus_OLS": 0.0, "Preferred_Direction": "Count", "Descriptive_Winner": "Tie"},
        {"Metric": "Out-of-sample trading dates", "Pooled_OLS": float(daily_df["Date"].nunique()), "XGBoost": float(daily_df["Date"].nunique()), "Difference_XGB_Minus_OLS": 0.0, "Preferred_Direction": "Count", "Descriptive_Winner": "Tie"},
        {"Metric": "Mean daily IC", "Pooled_OLS": ols_mean_ic, "XGBoost": xgb_mean_ic, "Difference_XGB_Minus_OLS": xgb_mean_ic - ols_mean_ic, "Preferred_Direction": "Higher is better", "Descriptive_Winner": _winner_higher(ols_mean_ic, xgb_mean_ic)},
        {"Metric": "Median daily IC", "Pooled_OLS": float(daily_df["OLS_Daily_IC"].median()), "XGBoost": float(daily_df["XGB_Daily_IC"].median()), "Difference_XGB_Minus_OLS": float(daily_df["XGB_Daily_IC"].median()) - float(daily_df["OLS_Daily_IC"].median()), "Preferred_Direction": "Higher is better", "Descriptive_Winner": _winner_higher(float(daily_df["OLS_Daily_IC"].median()), float(daily_df["XGB_Daily_IC"].median()))},
        {"Metric": "Daily IC standard deviation", "Pooled_OLS": ols_std_ic, "XGBoost": xgb_std_ic, "Difference_XGB_Minus_OLS": xgb_std_ic - ols_std_ic, "Preferred_Direction": "Lower indicates greater stability", "Descriptive_Winner": _winner_lower(ols_std_ic, xgb_std_ic)},
        {"Metric": "Unannualised IC information ratio", "Pooled_OLS": ols_mean_ic / ols_std_ic, "XGBoost": xgb_mean_ic / xgb_std_ic, "Difference_XGB_Minus_OLS": (xgb_mean_ic / xgb_std_ic) - (ols_mean_ic / ols_std_ic), "Preferred_Direction": "Higher is better", "Descriptive_Winner": _winner_higher(ols_mean_ic / ols_std_ic, xgb_mean_ic / xgb_std_ic)},
        {"Metric": "Positive-IC trading days (%)", "Pooled_OLS": float(daily_df["OLS_Daily_IC"].gt(0).mean() * 100), "XGBoost": float(daily_df["XGB_Daily_IC"].gt(0).mean() * 100), "Difference_XGB_Minus_OLS": float(daily_df["XGB_Daily_IC"].gt(0).mean() * 100) - float(daily_df["OLS_Daily_IC"].gt(0).mean() * 100), "Preferred_Direction": "Higher is better", "Descriptive_Winner": _winner_higher(float(daily_df["OLS_Daily_IC"].gt(0).mean()), float(daily_df["XGB_Daily_IC"].gt(0).mean()))},
        {"Metric": "Observation-level MAE", "Pooled_OLS": ols_mae, "XGBoost": xgb_mae, "Difference_XGB_Minus_OLS": xgb_mae - ols_mae, "Preferred_Direction": "Lower is better", "Descriptive_Winner": _winner_lower(ols_mae, xgb_mae)},
        {"Metric": "Observation-level RMSE", "Pooled_OLS": ols_rmse, "XGBoost": xgb_rmse, "Difference_XGB_Minus_OLS": xgb_rmse - ols_rmse, "Preferred_Direction": "Lower is better", "Descriptive_Winner": _winner_lower(ols_rmse, xgb_rmse)},
        {"Metric": "Mean daily top-quartile hit rate", "Pooled_OLS": float(daily_df["OLS_Top_Hit_Rate"].mean()), "XGBoost": float(daily_df["XGB_Top_Hit_Rate"].mean()), "Difference_XGB_Minus_OLS": float(daily_df["XGB_Top_Hit_Rate"].mean()) - float(daily_df["OLS_Top_Hit_Rate"].mean()), "Preferred_Direction": "Higher is better", "Descriptive_Winner": _winner_higher(float(daily_df["OLS_Top_Hit_Rate"].mean()), float(daily_df["XGB_Top_Hit_Rate"].mean()))},
        {"Metric": "Mean daily bottom-quartile hit rate", "Pooled_OLS": float(daily_df["OLS_Bottom_Hit_Rate"].mean()), "XGBoost": float(daily_df["XGB_Bottom_Hit_Rate"].mean()), "Difference_XGB_Minus_OLS": float(daily_df["XGB_Bottom_Hit_Rate"].mean()) - float(daily_df["OLS_Bottom_Hit_Rate"].mean()), "Preferred_Direction": "Higher is better", "Descriptive_Winner": _winner_higher(float(daily_df["OLS_Bottom_Hit_Rate"].mean()), float(daily_df["XGB_Bottom_Hit_Rate"].mean()))},
    ]
    overall_df = pd.DataFrame(overall_metrics)

    # Save
    paired.to_csv(OUTPUT_PAIRED_PREDICTIONS, index=False)
    daily_df.to_csv(OUTPUT_DAILY_COMPARISON, index=False)
    quarterly_df.to_csv(OUTPUT_QUARTERLY_COMPARISON, index=False)
    overall_df.to_csv(OUTPUT_OVERALL_COMPARISON, index=False)
    pd.DataFrame([nw_results]).to_csv(OUTPUT_NW_TEST, index=False)

    print("\n" + "=" * 78)
    print("SCRIPT 09C COMPLETE")
    print(f"  OLS Mean Daily IC:      {ols_mean_ic:.6f}")
    print(f"  XGB (96-grid) Mean IC:  {xgb_mean_ic:.6f}")
    print(f"  Difference (XGB - OLS): {xgb_mean_ic - ols_mean_ic:.6f}")
    print(f"  Newey-West t-statistic: {nw_results['Normal_Test_Statistic']:.4f}")
    print(f"  One-sided p-value:      {nw_results['One_Sided_P_Value_XGB_Superior']:.4f}")
    print(f"  Directional Decision:   {nw_results['Directional_Decision']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
