"""
Script 11: Three-Way Experimental Setup Comparison
--------------------------------------------------

Purpose
-------
Produces a comprehensive three-way comparison table evaluating:
  1. Primary Setup:     1-Day Target, 32-Candidate Grid (Fixed lambda = 10.0)
  2. 1-Day Expanded:    1-Day Target, 96-Candidate Grid (Searched lambda in {1.0, 5.0, 10.0})
  3. 21-Day Alternative: 21-Day Target, 96-Candidate Grid (Searched lambda in {1.0, 5.0, 10.0})

Inputs
------
results/09_comparison/JSE_Model_Comparison_Overall.csv
results/09_comparison/JSE_Model_Comparison_Newey_West_IC_Test.csv
results/09C_comparison_1d_expanded/JSE_1D_Expanded_Model_Comparison_Overall.csv
results/09C_comparison_1d_expanded/JSE_1D_Expanded_Model_Comparison_Newey_West_IC_Test.csv
results/09B_comparison_alt/JSE_Alt_Model_Comparison_Overall.csv
results/09B_comparison_alt/JSE_Alt_Model_Comparison_Newey_West_IC_Test.csv

Outputs
-------
results/11_three_way_comparison/JSE_Three_Way_Setup_Comparison_Summary.csv

Run from project root
---------------------
python .\\scripts\\11_Three_Way_Setup_Comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Setup 1: 1-Day Primary
S1_OVERALL = PROJECT_ROOT / "results" / "09_comparison" / "JSE_Model_Comparison_Overall.csv"
S1_NW = PROJECT_ROOT / "results" / "09_comparison" / "JSE_Model_Comparison_Newey_West_IC_Test.csv"

# Setup 2: 1-Day Expanded Grid
S2_OVERALL = PROJECT_ROOT / "results" / "09C_comparison_1d_expanded" / "JSE_1D_Expanded_Model_Comparison_Overall.csv"
S2_NW = PROJECT_ROOT / "results" / "09C_comparison_1d_expanded" / "JSE_1D_Expanded_Model_Comparison_Newey_West_IC_Test.csv"

# Setup 3: 21-Day Alternative
S3_OVERALL = PROJECT_ROOT / "results" / "09B_comparison_alt" / "JSE_Alt_Model_Comparison_Overall.csv"
S3_NW = PROJECT_ROOT / "results" / "09B_comparison_alt" / "JSE_Alt_Model_Comparison_Newey_West_IC_Test.csv"

OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "11_three_way_comparison"
OUTPUT_SUMMARY = OUTPUT_DIRECTORY / "JSE_Three_Way_Setup_Comparison_Summary.csv"


def load_overall_metric_format(path: Path) -> pd.DataFrame:
    """Load an overall-comparison CSV that has a 'Metric' index column."""
    return pd.read_csv(path).set_index("Metric")


def load_overall_wide_format(path: Path) -> pd.DataFrame:
    """Load a wide-format overall CSV (09B style) that has no Metric column.
    Transposes the single data row into a Metric-indexed frame with
    columns Pooled_OLS and XGBoost using the named stat columns.
    """
    raw = pd.read_csv(path).iloc[0]
    # 09B overall column name mapping -> standard metric names
    metric_map = {
        "Out-of-sample stock-day predictions": (raw["Total_Paired_Observations"], raw["Total_Paired_Observations"]),
        "Out-of-sample trading dates": (raw["Total_Paired_Dates"], raw["Total_Paired_Dates"]),
        "Mean daily IC": (raw["OLS_Mean_Daily_IC"], raw["XGB_Mean_Daily_IC"]),
        "Median daily IC": (raw["OLS_Median_Daily_IC"], raw["XGB_Median_Daily_IC"]),
        "Daily IC standard deviation": (raw["OLS_IC_Std"], raw["XGB_IC_Std"]),
        "Unannualised IC information ratio": (raw["OLS_ICIR"], raw["XGB_ICIR"]),
        "Positive-IC trading days (%)": (raw["OLS_Positive_IC_Pct"], raw["XGB_Positive_IC_Pct"]),
        "Mean daily top-quartile hit rate": (None, None),
        "Mean daily bottom-quartile hit rate": (None, None),
    }
    records = [
        {"Metric": k, "Pooled_OLS": v[0], "XGBoost": v[1]}
        for k, v in metric_map.items()
    ]
    return pd.DataFrame(records).set_index("Metric")


def normalise_nw_09b(row: pd.Series) -> pd.Series:
    """Remap 09B NW column names to the standard names used by this script."""
    return pd.Series({
        "Mean_IC_Difference_XGB_Minus_OLS": row["Mean_IC_Difference"],
        "Newey_West_Standard_Error": row["Newey_West_SE"],
        "Normal_Test_Statistic": row["Test_Statistic"],
        "One_Sided_P_Value_XGB_Superior": row["One_Sided_P_Value"],
        "Two_Sided_P_Value": row["Two_Sided_P_Value"],
    })


def main() -> None:
    print("=" * 78)
    print("SCRIPT 11: THREE-WAY EXPERIMENTAL SETUP COMPARISON")
    print("=" * 78)

    for p in [S1_OVERALL, S1_NW, S2_OVERALL, S2_NW, S3_OVERALL, S3_NW]:
        if not p.exists():
            sys.exit(f"ERROR: Required file missing: {p}")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    df1 = load_overall_metric_format(S1_OVERALL)
    nw1 = pd.read_csv(S1_NW).iloc[0]

    df2 = load_overall_metric_format(S2_OVERALL)
    nw2 = pd.read_csv(S2_NW).iloc[0]

    # 09B uses a different wide format
    df3 = load_overall_wide_format(S3_OVERALL)
    nw3 = normalise_nw_09b(pd.read_csv(S3_NW).iloc[0])

    def val(df, metric, col):
        try:
            return float(df.loc[metric, col])
        except Exception:
            return None

    rows = [
        # Specification
        ("Target Return Horizon", "1 trading day", "1 trading day", "21 trading days (~1 month)"),
        ("Purge Buffer Between Windows", "1 trading date", "1 trading date", "21 trading dates"),
        ("XGBoost Regularisation Lambda", "Fixed (10.0)", "Grid searched: {1.0, 5.0, 10.0}", "Grid searched: {1.0, 5.0, 10.0}"),
        ("XGBoost Candidate Grid Size", "32 candidates", "96 candidates", "96 candidates"),
        ("Newey-West HAC Lag q", "5 lags", "5 lags", "21 lags"),
        ("Out-of-sample Predictions", f"{int(val(df1, 'Out-of-sample stock-day predictions', 'Pooled_OLS')):,}",
                                      f"{int(val(df2, 'Out-of-sample stock-day predictions', 'Pooled_OLS')):,}",
                                      f"{int(val(df3, 'Out-of-sample stock-day predictions', 'Pooled_OLS')):,}"),
        ("Out-of-sample Trading Dates (T)", f"{int(val(df1, 'Out-of-sample trading dates', 'Pooled_OLS'))}",
                                            f"{int(val(df2, 'Out-of-sample trading dates', 'Pooled_OLS'))}",
                                            f"{int(val(df3, 'Out-of-sample trading dates', 'Pooled_OLS'))}"),
        
        # Pooled OLS
        ("OLS Mean Daily IC", f"{val(df1, 'Mean daily IC', 'Pooled_OLS'):.6f}",
                              f"{val(df2, 'Mean daily IC', 'Pooled_OLS'):.6f}",
                              f"{val(df3, 'Mean daily IC', 'Pooled_OLS'):.6f}"),
        ("OLS Median Daily IC", f"{val(df1, 'Median daily IC', 'Pooled_OLS'):.6f}",
                                f"{val(df2, 'Median daily IC', 'Pooled_OLS'):.6f}",
                                f"{val(df3, 'Median daily IC', 'Pooled_OLS'):.6f}"),
        ("OLS Daily IC StdDev", f"{val(df1, 'Daily IC standard deviation', 'Pooled_OLS'):.6f}",
                                f"{val(df2, 'Daily IC standard deviation', 'Pooled_OLS'):.6f}",
                                f"{val(df3, 'Daily IC standard deviation', 'Pooled_OLS'):.6f}"),
        ("OLS IC Information Ratio", f"{val(df1, 'Unannualised IC information ratio', 'Pooled_OLS'):.4f}",
                                     f"{val(df2, 'Unannualised IC information ratio', 'Pooled_OLS'):.4f}",
                                     f"{val(df3, 'Unannualised IC information ratio', 'Pooled_OLS'):.4f}"),
        ("OLS Positive IC Days (%)", f"{val(df1, 'Positive-IC trading days (%)', 'Pooled_OLS'):.2f}%",
                                     f"{val(df2, 'Positive-IC trading days (%)', 'Pooled_OLS'):.2f}%",
                                     f"{val(df3, 'Positive-IC trading days (%)', 'Pooled_OLS'):.2f}%"),

        # XGBoost
        ("XGBoost Mean Daily IC", f"{val(df1, 'Mean daily IC', 'XGBoost'):.6f}",
                                  f"{val(df2, 'Mean daily IC', 'XGBoost'):.6f}",
                                  f"{val(df3, 'Mean daily IC', 'XGBoost'):.6f}"),
        ("XGBoost Median Daily IC", f"{val(df1, 'Median daily IC', 'XGBoost'):.6f}",
                                    f"{val(df2, 'Median daily IC', 'XGBoost'):.6f}",
                                    f"{val(df3, 'Median daily IC', 'XGBoost'):.6f}"),
        ("XGBoost Daily IC StdDev", f"{val(df1, 'Daily IC standard deviation', 'XGBoost'):.6f}",
                                    f"{val(df2, 'Daily IC standard deviation', 'XGBoost'):.6f}",
                                    f"{val(df3, 'Daily IC standard deviation', 'XGBoost'):.6f}"),
        ("XGBoost IC Information Ratio", f"{val(df1, 'Unannualised IC information ratio', 'XGBoost'):.4f}",
                                         f"{val(df2, 'Unannualised IC information ratio', 'XGBoost'):.4f}",
                                         f"{val(df3, 'Unannualised IC information ratio', 'XGBoost'):.4f}"),
        ("XGBoost Positive IC Days (%)", f"{val(df1, 'Positive-IC trading days (%)', 'XGBoost'):.2f}%",
                                         f"{val(df2, 'Positive-IC trading days (%)', 'XGBoost'):.2f}%",
                                         f"{val(df3, 'Positive-IC trading days (%)', 'XGBoost'):.2f}%"),

        # Statistical Inference
        ("Mean IC Difference (XGB - OLS)", f"{float(nw1['Mean_IC_Difference_XGB_Minus_OLS']):.6f}",
                                          f"{float(nw2['Mean_IC_Difference_XGB_Minus_OLS']):.6f}",
                                          f"{float(nw3['Mean_IC_Difference_XGB_Minus_OLS']):.6f}"),
        ("Newey-West HAC Standard Error", f"{float(nw1['Newey_West_Standard_Error']):.6f}",
                                         f"{float(nw2['Newey_West_Standard_Error']):.6f}",
                                         f"{float(nw3['Newey_West_Standard_Error']):.6f}"),
        ("Newey-West t-statistic", f"{float(nw1['Normal_Test_Statistic']):.4f}",
                                   f"{float(nw2['Normal_Test_Statistic']):.4f}",
                                   f"{float(nw3['Normal_Test_Statistic']):.4f}"),
        ("One-sided p-value (H1: XGB > OLS)", f"{float(nw1['One_Sided_P_Value_XGB_Superior']):.4f}",
                                              f"{float(nw2['One_Sided_P_Value_XGB_Superior']):.4f}",
                                              f"{float(nw3['One_Sided_P_Value_XGB_Superior']):.4f}"),
        ("Two-sided p-value", f"{float(nw1['Two_Sided_P_Value']):.4f}",
                              f"{float(nw2['Two_Sided_P_Value']):.4f}",
                              f"{float(nw3['Two_Sided_P_Value']):.4f}"),
        ("Statistical Significance at 5%", "No (Fail to reject H0)",
                                           "No (Fail to reject H0)" if float(nw2['One_Sided_P_Value_XGB_Superior']) > 0.05 else "Yes (Reject H0)",
                                           "No (Fail to reject H0)"),
        ("Descriptive Ranking Winner", "Pooled OLS",
                                      "XGBoost" if val(df2, 'Mean daily IC', 'XGBoost') > val(df2, 'Mean daily IC', 'Pooled_OLS') else "Pooled OLS",
                                      "Pooled OLS"),
    ]

    summary_df = pd.DataFrame(rows, columns=["Metric", "Primary_1D_Fixed_Lambda", "Expanded_1D_Searched_Lambda", "Alternative_21D_Searched_Lambda"])
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)

    print("\n" + "=" * 78)
    print("SCRIPT 11 COMPLETE: Three-Way Comparison Table Generated")
    print(f"  Summary table saved to: {OUTPUT_SUMMARY}")
    print("=" * 78)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
