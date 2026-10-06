"""
Script 07B: Alternative Pooled OLS Baseline (21-Day Target)
-----------------------------------------------------------

Purpose
-------
Runs the exact same Pooled Cross-Sectional OLS estimation as Script 07,
but uses the alternative 21-day-target master panel and walk-forward
windows produced by Scripts 03B and 06B.

No algorithmic changes to OLS itself.  Only the input paths and output
paths differ.

Inputs
------
data/processed/06B_alt_validation/JSE_Alt_Master_Model_Panel.csv
data/processed/06B_alt_validation/JSE_Alt_Walk_Forward_Windows.csv

Core outputs
------------
results/07_ols_alt/JSE_Alt_OLS_Walk_Forward_Predictions.csv
results/07_ols_alt/JSE_Alt_OLS_Daily_Performance.csv
results/07_ols_alt/JSE_Alt_OLS_Window_Performance.csv
results/07_ols_alt/JSE_Alt_OLS_Overall_Performance.csv

Run from the project root
-------------------------
python .\\scripts\\07B_Alt_OLS_Baseline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, spearmanr
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALIDATION_DIRECTORY = (
    PROJECT_ROOT / "data" / "processed" / "06B_alt_validation"
)

RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "07_ols_alt"
AUDIT_DIRECTORY = PROJECT_ROOT / "outputs" / "audits" / "07_ols_alt"

INPUT_MASTER_PANEL = VALIDATION_DIRECTORY / "JSE_Alt_Master_Model_Panel.csv"
INPUT_OUTER_WINDOWS = VALIDATION_DIRECTORY / "JSE_Alt_Walk_Forward_Windows.csv"

OUTPUT_PREDICTIONS = RESULTS_DIRECTORY / "JSE_Alt_OLS_Walk_Forward_Predictions.csv"
OUTPUT_DAILY_PERFORMANCE = RESULTS_DIRECTORY / "JSE_Alt_OLS_Daily_Performance.csv"
OUTPUT_WINDOW_PERFORMANCE = RESULTS_DIRECTORY / "JSE_Alt_OLS_Window_Performance.csv"
OUTPUT_OVERALL_PERFORMANCE = RESULTS_DIRECTORY / "JSE_Alt_OLS_Overall_Performance.csv"
OUTPUT_COEFFICIENTS = AUDIT_DIRECTORY / "JSE_Alt_OLS_Coefficients.csv"
OUTPUT_WINSORISATION = AUDIT_DIRECTORY / "JSE_Alt_OLS_Winsorisation_Thresholds.csv"
OUTPUT_ESTIMATION_SUMMARY = AUDIT_DIRECTORY / "JSE_Alt_OLS_Estimation_Summary.csv"


# ============================================================
# 2. CONFIGURATION
# ============================================================

FEATURES = [
    "Mom_1M", "Mom_3M", "Mom_6M",
    "Vol_21D", "RSI_14",
    "SMA_50_Ratio", "SMA_200_Ratio",
    "Volume_Shock_21D", "Value_Traded_Shock_63D",
]

TARGET_COLUMN = "Target_Y"
MINIMUM_DAILY_CROSS_SECTION = 5
TOP_BOTTOM_FRACTION = 0.25
NORMAL_CRITICAL_VALUE = float(norm.ppf(0.975))
WINSOR_LOWER_QUANTILE = 0.01
WINSOR_UPPER_QUANTILE = 0.99


# ============================================================
# 3. WINSORISATION
# ============================================================

def estimate_winsorisation_thresholds(
    training_data: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    for feature in FEATURES:
        lower = float(training_data[feature].quantile(WINSOR_LOWER_QUANTILE))
        upper = float(training_data[feature].quantile(WINSOR_UPPER_QUANTILE))
        if not (np.isfinite(lower) and np.isfinite(upper)):
            raise ValueError(f"Non-finite winsorisation threshold for {feature}.")
        if lower > upper:
            raise ValueError(f"Reversed winsorisation bounds for {feature}.")
        records.append(
            {
                "Feature": feature,
                "Lower_Quantile": WINSOR_LOWER_QUANTILE,
                "Upper_Quantile": WINSOR_UPPER_QUANTILE,
                "Lower_Bound": lower,
                "Upper_Bound": upper,
            }
        )
    return pd.DataFrame(records)


def apply_winsorisation(
    data: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    transformed = data.copy()
    tmap = thresholds.set_index("Feature")
    for feature in FEATURES:
        transformed[feature] = transformed[feature].clip(
            lower=float(tmap.loc[feature, "Lower_Bound"]),
            upper=float(tmap.loc[feature, "Upper_Bound"]),
        )
    return transformed


# ============================================================
# 4. OLS ESTIMATION
# ============================================================

def fit_ols(
    training_data: pd.DataFrame,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    X = sm.add_constant(training_data[FEATURES].to_numpy(dtype=float))
    y = training_data[TARGET_COLUMN].to_numpy(dtype=float)
    return sm.OLS(y, X).fit()


def two_way_clustered_inference(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    training_data: pd.DataFrame,
) -> pd.DataFrame:
    date_groups = pd.factorize(training_data["Date"], sort=True)[0]
    ticker_groups = pd.factorize(training_data["Ticker"], sort=True)[0]
    cov_both, _, _ = cov_cluster_2groups(
        model, date_groups, ticker_groups, use_correction=True
    )
    diag = np.diag(cov_both)
    se = np.sqrt(np.clip(diag, 0.0, None))
    params = model.params
    z = np.divide(
        params, se, out=np.full_like(params, np.nan), where=se > 0
    )
    p = 2.0 * norm.sf(np.abs(z))
    terms = ["const"] + FEATURES
    return pd.DataFrame(
        {
            "Term": terms,
            "Coefficient": params,
            "Two_Way_Cluster_SE": se,
            "Z_Statistic": z,
            "P_Value": p,
            "CI_95_Lower": params - NORMAL_CRITICAL_VALUE * se,
            "CI_95_Upper": params + NORMAL_CRITICAL_VALUE * se,
        }
    )


# ============================================================
# 5. DAILY PERFORMANCE
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
        [TARGET_COLUMN, "Ticker"], ascending=[False, True], kind="mergesort"
    )
    pred_top = set(pred_sorted.head(q)["Ticker"])
    pred_bot = set(pred_sorted.tail(q)["Ticker"])
    real_top = set(real_sorted.head(q)["Ticker"])
    real_bot = set(real_sorted.tail(q)["Ticker"])
    top_hr = len(pred_top & real_top) / q
    bot_hr = len(pred_bot & real_bot) / q
    return float(top_hr), float(bot_hr), q


def compute_daily_performance(
    predictions: pd.DataFrame,
    score_col: str = "OLS_Raw_Score",
) -> pd.DataFrame:
    records = []
    for date, group in predictions.groupby("Date", sort=True):
        ic = safe_spearman_ic(group[score_col], group[TARGET_COLUMN])
        errors = group[score_col] - group[TARGET_COLUMN]
        top_hr, bot_hr, _ = quartile_hit_rate(group, score_col)
        records.append(
            {
                "Date": date,
                "Outer_Window_ID": group["Outer_Window_ID"].iloc[0],
                "Test_Quarter": group["Test_Quarter"].iloc[0],
                "Model_Eligible_Securities": len(group),
                "Daily_IC": ic,
                "Daily_MAE": float(np.mean(np.abs(errors))),
                "Daily_RMSE": float(np.sqrt(np.mean(errors**2))),
                "Top_Quartile_Hit_Rate": top_hr,
                "Bottom_Quartile_Hit_Rate": bot_hr,
            }
        )
    return (
        pd.DataFrame(records)
        .sort_values("Date")
        .reset_index(drop=True)
    )


def summarise_performance(
    predictions: pd.DataFrame,
    daily: pd.DataFrame,
    score_col: str = "OLS_Raw_Score",
) -> dict:
    ic = daily["Daily_IC"].dropna()
    mean_ic = float(ic.mean()) if len(ic) > 0 else np.nan
    std_ic = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
    icir = mean_ic / std_ic if std_ic and std_ic > 0 else np.nan
    errors = predictions[score_col] - predictions[TARGET_COLUMN]
    return {
        "Out_Of_Sample_Observations": len(predictions),
        "Out_Of_Sample_Dates": predictions["Date"].nunique(),
        "Mean_Daily_IC": mean_ic,
        "Median_Daily_IC": float(ic.median()) if len(ic) > 0 else np.nan,
        "Daily_IC_Standard_Deviation": std_ic,
        "Daily_IC_Information_Ratio": icir,
        "Positive_IC_Days": int(ic.gt(0).sum()),
        "Positive_IC_Percentage": float(ic.gt(0).mean() * 100.0),
        "Observation_MAE": float(np.mean(np.abs(errors))),
        "Observation_RMSE": float(np.sqrt(np.mean(errors**2))),
        "Mean_Top_Quartile_Hit_Rate": float(daily["Top_Quartile_Hit_Rate"].mean()),
        "Mean_Bottom_Quartile_Hit_Rate": float(daily["Bottom_Quartile_Hit_Rate"].mean()),
    }


# ============================================================
# 6. MAIN PIPELINE
# ============================================================

def main() -> None:
    print("=" * 78)
    print("SCRIPT 07B: ALTERNATIVE POOLED OLS BASELINE  (21-DAY TARGET)")
    print("=" * 78)

    for path in [INPUT_MASTER_PANEL, INPUT_OUTER_WINDOWS]:
        if not path.exists():
            sys.exit(f"ERROR: Required file missing: {path}")

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    AUDIT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    overall_start = time.perf_counter()

    # Load data
    master_panel = pd.read_csv(INPUT_MASTER_PANEL)
    master_panel["Date"] = pd.to_datetime(master_panel["Date"])
    if "Next_Date" in master_panel.columns:
        master_panel["Next_Date"] = pd.to_datetime(
            master_panel["Next_Date"], errors="coerce"
        )
    master_panel = master_panel.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    outer_windows = pd.read_csv(INPUT_OUTER_WINDOWS)
    for col in ["Train_Start", "Train_End", "Purged_Boundary_Date",
                "Test_Start", "Test_End"]:
        outer_windows[col] = pd.to_datetime(outer_windows[col])
    outer_windows = outer_windows.sort_values("Test_Start").reset_index(drop=True)

    print(f"  Master panel: {len(master_panel):,} rows, "
          f"{master_panel['Date'].nunique()} dates")
    print(f"  Outer windows: {len(outer_windows)}")

    all_predictions: list[pd.DataFrame] = []
    all_coefficients: list[pd.DataFrame] = []
    all_thresholds: list[pd.DataFrame] = []
    estimation_records: list[dict] = []

    # -------------------------------------------------------
    # Expand loop: one OLS model per outer window
    # -------------------------------------------------------
    for outer_row in outer_windows.itertuples(index=False):
        window_start = time.perf_counter()
        wid = outer_row.Outer_Window_ID
        print(f"\n  [{wid}]  Test quarter: {outer_row.Test_Quarter}")

        train_mask = master_panel["Date"].between(
            pd.Timestamp(outer_row.Train_Start),
            pd.Timestamp(outer_row.Train_End),
            inclusive="both",
        )
        test_mask = master_panel["Date"].between(
            pd.Timestamp(outer_row.Test_Start),
            pd.Timestamp(outer_row.Test_End),
            inclusive="both",
        )

        train = master_panel.loc[train_mask].copy()
        test = master_panel.loc[test_mask].copy()

        print(f"    Train obs: {len(train):>8,}   Test obs: {len(test):>7,}")

        # Winsorise (fit on train, apply to both)
        thresholds = estimate_winsorisation_thresholds(train)
        train_w = apply_winsorisation(train, thresholds)
        test_w = apply_winsorisation(test, thresholds)

        thresholds.insert(0, "Outer_Window_ID", wid)
        thresholds.insert(1, "Test_Quarter", outer_row.Test_Quarter)
        all_thresholds.append(thresholds)

        # Fit OLS
        model = fit_ols(train_w)
        print(
            f"    OLS in-sample R^2: {model.rsquared:.6f}  "
            f"  Intercept: {model.params[0]:.4f}"
        )

        # Two-way clustered inference on training data
        try:
            coef_df = two_way_clustered_inference(model, train_w)
        except Exception as exc:
            print(f"    WARNING: Two-way cluster SE failed ({exc}). "
                  "Using OLS SE.")
            coef_df = pd.DataFrame(
                {
                    "Term": ["const"] + FEATURES,
                    "Coefficient": model.params,
                    "Two_Way_Cluster_SE": model.bse,
                    "Z_Statistic": model.tvalues,
                    "P_Value": model.pvalues,
                    "CI_95_Lower": model.conf_int()[0],
                    "CI_95_Upper": model.conf_int()[1],
                }
            )

        coef_df.insert(0, "Outer_Window_ID", wid)
        coef_df.insert(1, "Test_Quarter", outer_row.Test_Quarter)
        all_coefficients.append(coef_df)

        # Out-of-sample predictions
        X_test = sm.add_constant(
            test_w[FEATURES].to_numpy(dtype=float),
            has_constant="add",
        )
        raw_scores = model.predict(X_test)

        test_out = test_w[
            ["Date", "Ticker", TARGET_COLUMN, "Forward_Return", "N_t", "Model_N_t"]
        ].copy()
        test_out["Outer_Window_ID"] = wid
        test_out["Test_Quarter"] = outer_row.Test_Quarter
        test_out["OLS_Raw_Score"] = raw_scores

        # Predicted percentile (rank within each date's cross-section)
        test_out["OLS_Predicted_Percentile"] = (
            test_out.groupby("Date")["OLS_Raw_Score"]
            .rank(pct=True)
        )

        all_predictions.append(test_out)

        elapsed = time.perf_counter() - window_start
        print(f"    Completed in {elapsed:.1f}s")

        estimation_records.append(
            {
                "Outer_Window_ID": wid,
                "Test_Quarter": outer_row.Test_Quarter,
                "Train_Observations": len(train),
                "Test_Observations": len(test),
                "In_Sample_R_Squared": float(model.rsquared),
                "Intercept": float(model.params[0]),
                "Elapsed_Seconds": elapsed,
            }
        )

    # -------------------------------------------------------
    # Assemble full prediction panel
    # -------------------------------------------------------
    full_predictions = (
        pd.concat(all_predictions, ignore_index=True)
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )

    daily_performance = compute_daily_performance(full_predictions)
    overall_stats = summarise_performance(full_predictions, daily_performance)

    # Per-window performance
    window_records = []
    for wid in outer_windows["Outer_Window_ID"]:
        pred_w = full_predictions.loc[
            full_predictions["Outer_Window_ID"] == wid
        ]
        daily_w = daily_performance.loc[
            daily_performance["Outer_Window_ID"] == wid
        ]
        tq = outer_windows.loc[
            outer_windows["Outer_Window_ID"] == wid,
            "Test_Quarter",
        ].iloc[0]
        rec = summarise_performance(pred_w, daily_w)
        rec["Outer_Window_ID"] = wid
        rec["Test_Quarter"] = tq
        window_records.append(rec)

    window_performance = pd.DataFrame(window_records)

    # -------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------
    full_predictions.to_csv(OUTPUT_PREDICTIONS, index=False)
    daily_performance.to_csv(OUTPUT_DAILY_PERFORMANCE, index=False)
    window_performance.to_csv(OUTPUT_WINDOW_PERFORMANCE, index=False)
    pd.DataFrame([overall_stats]).to_csv(OUTPUT_OVERALL_PERFORMANCE, index=False)
    pd.concat(all_coefficients, ignore_index=True).to_csv(
        OUTPUT_COEFFICIENTS, index=False
    )
    pd.concat(all_thresholds, ignore_index=True).to_csv(
        OUTPUT_WINSORISATION, index=False
    )
    pd.DataFrame(estimation_records).to_csv(
        OUTPUT_ESTIMATION_SUMMARY, index=False
    )

    total_elapsed = time.perf_counter() - overall_start

    print("\n" + "=" * 78)
    print("  OVERALL OLS PERFORMANCE  (21-Day Target)")
    print("=" * 78)
    print(f"  Out-of-sample observations: {overall_stats['Out_Of_Sample_Observations']:>9,}")
    print(f"  Out-of-sample dates:        {overall_stats['Out_Of_Sample_Dates']:>9,}")
    print(f"  Mean Daily IC:              {overall_stats['Mean_Daily_IC']:>12.6f}")
    print(f"  IC Standard Deviation:      {overall_stats['Daily_IC_Standard_Deviation']:>12.6f}")
    print(f"  IC Information Ratio:       {overall_stats['Daily_IC_Information_Ratio']:>12.6f}")
    print(f"  Positive IC Days (%):       {overall_stats['Positive_IC_Percentage']:>11.2f}%")
    print(f"  Observation MAE:            {overall_stats['Observation_MAE']:>12.6f}")
    print(f"  Observation RMSE:           {overall_stats['Observation_RMSE']:>12.6f}")
    print(f"\n  Total elapsed: {total_elapsed:.1f}s")
    print("\n  Script 07B complete.")


if __name__ == "__main__":
    main()
