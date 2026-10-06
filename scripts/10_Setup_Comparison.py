"""
Script 10: Original-vs-Alternative Setup Comparison
----------------------------------------------------

Purpose
-------
Produces a single side-by-side summary table comparing the original
1-day-horizon experimental setup (Scripts 07-09) against the alternative
21-day-horizon setup with expanded XGBoost grid (Scripts 07B-09B).

Inputs
------
results/09_comparison/JSE_Model_Comparison_Newey_West_IC_Test.csv
results/09_comparison/JSE_Model_Comparison_Overall.csv
results/09_comparison/JSE_Model_Comparison_Quarterly.csv
results/09B_comparison_alt/JSE_Alt_Model_Comparison_Newey_West_IC_Test.csv
results/09B_comparison_alt/JSE_Alt_Model_Comparison_Overall.csv
results/09B_comparison_alt/JSE_Alt_Model_Comparison_Quarterly.csv

Outputs
-------
results/10_setup_comparison/JSE_Setup_Comparison_Summary.csv
results/10_setup_comparison/JSE_Setup_Comparison_Quarterly.csv

Run from the project root
-------------------------
python .\\scripts\\10_Setup_Comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Original setup (Scripts 07-09)
ORIG_OVERALL = (
    PROJECT_ROOT / "results" / "09_comparison"
    / "JSE_Model_Comparison_Overall.csv"
)
ORIG_NW = (
    PROJECT_ROOT / "results" / "09_comparison"
    / "JSE_Model_Comparison_Newey_West_IC_Test.csv"
)
ORIG_QUARTERLY = (
    PROJECT_ROOT / "results" / "09_comparison"
    / "JSE_Model_Comparison_Quarterly.csv"
)

# Alternative setup (Scripts 07B-09B)
ALT_OVERALL = (
    PROJECT_ROOT / "results" / "09B_comparison_alt"
    / "JSE_Alt_Model_Comparison_Overall.csv"
)
ALT_NW = (
    PROJECT_ROOT / "results" / "09B_comparison_alt"
    / "JSE_Alt_Model_Comparison_Newey_West_IC_Test.csv"
)
ALT_QUARTERLY = (
    PROJECT_ROOT / "results" / "09B_comparison_alt"
    / "JSE_Alt_Model_Comparison_Quarterly.csv"
)

OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "10_setup_comparison"
OUTPUT_SUMMARY = OUTPUT_DIRECTORY / "JSE_Setup_Comparison_Summary.csv"
OUTPUT_QUARTERLY = OUTPUT_DIRECTORY / "JSE_Setup_Comparison_Quarterly.csv"


# ============================================================
# 2. MAIN
# ============================================================

def main() -> None:
    print("=" * 78)
    print("SCRIPT 10: ORIGINAL vs ALTERNATIVE SETUP COMPARISON")
    print("=" * 78)

    for path in [ORIG_OVERALL, ORIG_NW, ALT_OVERALL, ALT_NW]:
        if not path.exists():
            sys.exit(f"ERROR: Required file missing: {path}")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load and index original results
    # --------------------------------------------------------
    orig_overall_df = pd.read_csv(ORIG_OVERALL).set_index("Metric")
    orig_nw_df = pd.read_csv(ORIG_NW).iloc[0]

    def get_orig_stat(metric_name: str, col: str = "Pooled_OLS") -> float:
        try:
            return float(orig_overall_df.loc[metric_name, col])
        except (KeyError, ValueError):
            return float("nan")

    # --------------------------------------------------------
    # Load alternative results
    # --------------------------------------------------------
    alt_overall = pd.read_csv(ALT_OVERALL).iloc[0]
    alt_nw = pd.read_csv(ALT_NW).iloc[0]

    # --------------------------------------------------------
    # Build comparison records
    # --------------------------------------------------------
    records = [
        # Setup metadata
        {"Metric": "Target Return Horizon",
         "Original_Setup": "1 trading day",
         "Alternative_Setup": "21 trading days (~1 month)"},
        {"Metric": "Purge Buffer Between Windows",
         "Original_Setup": "1 trading date",
         "Alternative_Setup": "21 trading dates"},
        {"Metric": "XGBoost Regularisation Lambda",
         "Original_Setup": "Fixed (10.0)",
         "Alternative_Setup": "Grid searched: {1.0, 5.0, 10.0}"},
        {"Metric": "XGBoost Candidate Grid Size",
         "Original_Setup": "32 candidates",
         "Alternative_Setup": "96 candidates"},
        {"Metric": "Newey-West HAC Lag q",
         "Original_Setup": str(int(orig_nw_df["Newey_West_Max_Lag"])),
         "Alternative_Setup": str(int(alt_nw["NW_Lag_q"]))},

        # Sample sizes
        {"Metric": "Out-of-sample Predictions",
         "Original_Setup": f"{int(get_orig_stat('Out-of-sample stock-day predictions')):,}",
         "Alternative_Setup": f"{int(alt_overall['Total_Paired_Observations']):,}"},
        {"Metric": "Out-of-sample Trading Dates (T)",
         "Original_Setup": f"{int(get_orig_stat('Out-of-sample trading dates')):,}",
         "Alternative_Setup": f"{int(alt_overall['Total_Paired_Dates']):,}"},

        # OLS Metrics
        {"Metric": "OLS Mean Daily IC",
         "Original_Setup": f"{get_orig_stat('Mean daily IC', 'Pooled_OLS'):.6f}",
         "Alternative_Setup": f"{float(alt_overall['OLS_Mean_Daily_IC']):.6f}"},
        {"Metric": "OLS Median Daily IC",
         "Original_Setup": f"{get_orig_stat('Median daily IC', 'Pooled_OLS'):.6f}",
         "Alternative_Setup": f"{float(alt_overall['OLS_Median_Daily_IC']):.6f}"},
        {"Metric": "OLS Daily IC StdDev",
         "Original_Setup": f"{get_orig_stat('Daily IC standard deviation', 'Pooled_OLS'):.6f}",
         "Alternative_Setup": f"{float(alt_overall['OLS_IC_Std']):.6f}"},
        {"Metric": "OLS IC Information Ratio",
         "Original_Setup": f"{get_orig_stat('Unannualised IC information ratio', 'Pooled_OLS'):.4f}",
         "Alternative_Setup": f"{float(alt_overall['OLS_ICIR']):.4f}"},
        {"Metric": "OLS Positive IC Days (%)",
         "Original_Setup": f"{get_orig_stat('Positive-IC trading days (%)', 'Pooled_OLS'):.2f}%",
         "Alternative_Setup": f"{float(alt_overall['OLS_Positive_IC_Pct']):.2f}%"},
        {"Metric": "OLS Observation-level RMSE",
         "Original_Setup": f"{get_orig_stat('Observation-level RMSE', 'Pooled_OLS'):.6f}",
         "Alternative_Setup": f"{float(alt_overall['OLS_Observation_RMSE']):.6f}"},
        {"Metric": "OLS Observation-level MAE",
         "Original_Setup": f"{get_orig_stat('Observation-level MAE', 'Pooled_OLS'):.6f}",
         "Alternative_Setup": f"{float(alt_overall['OLS_Observation_MAE']):.6f}"},

        # XGBoost Metrics
        {"Metric": "XGBoost Mean Daily IC",
         "Original_Setup": f"{get_orig_stat('Mean daily IC', 'XGBoost'):.6f}",
         "Alternative_Setup": f"{float(alt_overall['XGB_Mean_Daily_IC']):.6f}"},
        {"Metric": "XGBoost Median Daily IC",
         "Original_Setup": f"{get_orig_stat('Median daily IC', 'XGBoost'):.6f}",
         "Alternative_Setup": f"{float(alt_overall['XGB_Median_Daily_IC']):.6f}"},
        {"Metric": "XGBoost Daily IC StdDev",
         "Original_Setup": f"{get_orig_stat('Daily IC standard deviation', 'XGBoost'):.6f}",
         "Alternative_Setup": f"{float(alt_overall['XGB_IC_Std']):.6f}"},
        {"Metric": "XGBoost IC Information Ratio",
         "Original_Setup": f"{get_orig_stat('Unannualised IC information ratio', 'XGBoost'):.4f}",
         "Alternative_Setup": f"{float(alt_overall['XGB_ICIR']):.4f}"},
        {"Metric": "XGBoost Positive IC Days (%)",
         "Original_Setup": f"{get_orig_stat('Positive-IC trading days (%)', 'XGBoost'):.2f}%",
         "Alternative_Setup": f"{float(alt_overall['XGB_Positive_IC_Pct']):.2f}%"},
        {"Metric": "XGBoost Observation-level RMSE",
         "Original_Setup": f"{get_orig_stat('Observation-level RMSE', 'XGBoost'):.6f}",
         "Alternative_Setup": f"{float(alt_overall['XGB_Observation_RMSE']):.6f}"},
        {"Metric": "XGBoost Observation-level MAE",
         "Original_Setup": f"{get_orig_stat('Observation-level MAE', 'XGBoost'):.6f}",
         "Alternative_Setup": f"{float(alt_overall['XGB_Observation_MAE']):.6f}"},

        # Comparative Statistical Inference
        {"Metric": "Mean IC Difference (XGB - OLS)",
         "Original_Setup": f"{float(orig_nw_df['Mean_IC_Difference_XGB_Minus_OLS']):+.6f}",
         "Alternative_Setup": f"{float(alt_nw['Mean_IC_Difference']):+.6f}"},
        {"Metric": "Newey-West HAC Standard Error",
         "Original_Setup": f"{float(orig_nw_df['Newey_West_Standard_Error']):.6f}",
         "Alternative_Setup": f"{float(alt_nw['Newey_West_SE']):.6f}"},
        {"Metric": "Newey-West t-statistic",
         "Original_Setup": f"{float(orig_nw_df['Normal_Test_Statistic']):+.4f}",
         "Alternative_Setup": f"{float(alt_nw['Test_Statistic']):+.4f}"},
        {"Metric": "One-sided p-value (H1: XGB > OLS)",
         "Original_Setup": f"{float(orig_nw_df['One_Sided_P_Value_XGB_Superior']):.4f}",
         "Alternative_Setup": f"{float(alt_nw['One_Sided_P_Value']):.4f}"},
        {"Metric": "Statistical Significance at 5%",
         "Original_Setup": "No (Fail to reject H0)",
         "Alternative_Setup": "No (Fail to reject H0)"},
    ]

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)

    # --------------------------------------------------------
    # Quarterly side-by-side
    # --------------------------------------------------------
    orig_q = pd.read_csv(ORIG_QUARTERLY)
    alt_q = pd.read_csv(ALT_QUARTERLY)

    orig_q_sub = orig_q[["Test_Quarter", "OLS_Mean_Daily_IC", "XGB_Mean_Daily_IC"]].rename(
        columns={
            "OLS_Mean_Daily_IC": "Orig_OLS_Mean_IC",
            "XGB_Mean_Daily_IC": "Orig_XGB_Mean_IC",
        }
    )
    alt_q_sub = alt_q[["Test_Quarter", "OLS_Mean_Daily_IC", "XGB_Mean_Daily_IC"]].rename(
        columns={
            "OLS_Mean_Daily_IC": "Alt_OLS_Mean_IC",
            "XGB_Mean_Daily_IC": "Alt_XGB_Mean_IC",
        }
    )

    quarterly_merged = orig_q_sub.merge(alt_q_sub, on="Test_Quarter", how="outer").sort_values("Test_Quarter")
    quarterly_merged["Orig_Diff_XGB_minus_OLS"] = quarterly_merged["Orig_XGB_Mean_IC"] - quarterly_merged["Orig_OLS_Mean_IC"]
    quarterly_merged["Alt_Diff_XGB_minus_OLS"] = quarterly_merged["Alt_XGB_Mean_IC"] - quarterly_merged["Alt_OLS_Mean_IC"]

    quarterly_merged.to_csv(OUTPUT_QUARTERLY, index=False)

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------
    print("\n  =========================================================================")
    print("         ORIGINAL (1-DAY)  vs  ALTERNATIVE (21-DAY) SETUP COMPARISON")
    print("  =========================================================================")
    print(f"  {'Metric':<36} {'Original (1-Day)':>18} {'Alternative (21-Day)':>22}")
    print("  " + "-" * 78)
    for _, row in summary_df.iterrows():
        print(f"  {row['Metric']:<36} {row['Original_Setup']:>18} {row['Alternative_Setup']:>22}")
    print("  " + "-" * 78)

    print("\n  Quarterly Mean Information Coefficient (IC) Comparison:")
    print("  " + "-" * 78)
    print(f"  {'Quarter':<10} {'Orig OLS':>11} {'Orig XGB':>11} {'Orig Diff':>11} | {'Alt OLS':>11} {'Alt XGB':>11} {'Alt Diff':>11}")
    print("  " + "-" * 78)
    for _, r in quarterly_merged.iterrows():
        print(
            f"  {r['Test_Quarter']:<10} "
            f"{r['Orig_OLS_Mean_IC']:>+11.5f} {r['Orig_XGB_Mean_IC']:>+11.5f} {r['Orig_Diff_XGB_minus_OLS']:>+11.5f} | "
            f"{r['Alt_OLS_Mean_IC']:>+11.5f} {r['Alt_XGB_Mean_IC']:>+11.5f} {r['Alt_Diff_XGB_minus_OLS']:>+11.5f}"
        )
    print("  " + "-" * 78)

    print(f"\n  Saved summary table to:    {OUTPUT_SUMMARY.relative_to(PROJECT_ROOT)}")
    print(f"  Saved quarterly table to:  {OUTPUT_QUARTERLY.relative_to(PROJECT_ROOT)}")
    print("\n  Script 10 complete.")


if __name__ == "__main__":
    main()
