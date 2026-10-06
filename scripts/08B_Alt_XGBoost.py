"""
Script 08B: Alternative XGBoost Estimation (21-Day Target, Expanded Grid)
--------------------------------------------------------------------------

Purpose
-------
Runs the exact same nested expanding-window XGBoost estimation as Script 08,
but uses the alternative 21-day-target master panel and walk-forward windows
produced by Scripts 03B and 06B, and expands the hyperparameter grid to also
search over L2 regularisation (reg_lambda in {1.0, 5.0, 10.0}).

Grid size: 2 × 2 × 2 × 2 × 2 × 3 = 96 candidates  (vs 32 in Script 08)

Inputs
------
data/processed/06B_alt_validation/JSE_Alt_Master_Model_Panel.csv
data/processed/06B_alt_validation/JSE_Alt_Walk_Forward_Windows.csv
data/processed/06B_alt_validation/JSE_Alt_Inner_Validation_Windows.csv

Core outputs
------------
results/08_xgboost_alt/JSE_Alt_XGB_Walk_Forward_Predictions.csv
results/08_xgboost_alt/JSE_Alt_XGB_Daily_Performance.csv
results/08_xgboost_alt/JSE_Alt_XGB_Window_Performance.csv
results/08_xgboost_alt/JSE_Alt_XGB_Overall_Performance.csv

Audit outputs
-------------
outputs/audits/08_xgboost_alt/JSE_Alt_XGB_Hyperparameter_Search_Results.csv
outputs/audits/08_xgboost_alt/JSE_Alt_XGB_Selected_Hyperparameters.csv

Run from the project root
-------------------------
python .\\scripts\\08B_Alt_XGBoost.py
"""

from __future__ import annotations

import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALIDATION_DIRECTORY = (
    PROJECT_ROOT / "data" / "processed" / "06B_alt_validation"
)

RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "08_xgboost_alt"
AUDIT_DIRECTORY = PROJECT_ROOT / "outputs" / "audits" / "08_xgboost_alt"

INPUT_MASTER_PANEL = VALIDATION_DIRECTORY / "JSE_Alt_Master_Model_Panel.csv"
INPUT_OUTER_WINDOWS = VALIDATION_DIRECTORY / "JSE_Alt_Walk_Forward_Windows.csv"
INPUT_INNER_WINDOWS = VALIDATION_DIRECTORY / "JSE_Alt_Inner_Validation_Windows.csv"

OUTPUT_PREDICTIONS = RESULTS_DIRECTORY / "JSE_Alt_XGB_Walk_Forward_Predictions.csv"
OUTPUT_DAILY_PERFORMANCE = RESULTS_DIRECTORY / "JSE_Alt_XGB_Daily_Performance.csv"
OUTPUT_WINDOW_PERFORMANCE = RESULTS_DIRECTORY / "JSE_Alt_XGB_Window_Performance.csv"
OUTPUT_OVERALL_PERFORMANCE = RESULTS_DIRECTORY / "JSE_Alt_XGB_Overall_Performance.csv"

OUTPUT_SEARCH_RESULTS = (
    AUDIT_DIRECTORY / "JSE_Alt_XGB_Hyperparameter_Search_Results.csv"
)
OUTPUT_SELECTED_PARAMS = (
    AUDIT_DIRECTORY / "JSE_Alt_XGB_Selected_Hyperparameters.csv"
)
OUTPUT_WINSORISATION = (
    AUDIT_DIRECTORY / "JSE_Alt_XGB_Winsorisation_Thresholds.csv"
)
OUTPUT_ESTIMATION_SUMMARY = (
    AUDIT_DIRECTORY / "JSE_Alt_XGB_Estimation_Summary.csv"
)


# ============================================================
# 2. MODEL CONFIGURATION
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

WINSOR_LOWER_QUANTILE = 0.01
WINSOR_UPPER_QUANTILE = 0.99

RANDOM_STATE = 42
TREE_METHOD = "hist"
OBJECTIVE = "reg:squarederror"
EVAL_METRIC = "rmse"

MAX_BOOSTING_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 50

# *** EXPANDED GRID vs Script 08 ***
# 2 × 2 × 2 × 2 × 2 × 3 = 96 candidates
LEARNING_RATES = [0.02, 0.05]
MAX_DEPTHS = [2, 3]
MIN_CHILD_WEIGHTS = [5.0, 15.0]
SUBSAMPLES = [0.8, 1.0]
COLSAMPLE_BYTREE_VALUES = [0.8, 1.0]
REG_LAMBDA_VALUES = [1.0, 5.0, 10.0]   # <- new dimension

REG_ALPHA = 0.10
GAMMA = 0.0


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
            raise ValueError(f"Non-finite winsorisation for {feature}.")
        records.append(
            {
                "Feature": feature,
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
# 4. XGBOOST HELPERS
# ============================================================

def candidate_configurations() -> list[dict]:
    """Produce all 96 candidate configurations."""
    candidates = []
    grid = product(
        LEARNING_RATES,
        MAX_DEPTHS,
        MIN_CHILD_WEIGHTS,
        SUBSAMPLES,
        COLSAMPLE_BYTREE_VALUES,
        REG_LAMBDA_VALUES,
    )
    for i, (lr, depth, mcw, ss, cst, lam) in enumerate(grid, start=1):
        candidates.append(
            {
                "Candidate_ID": f"C_{i:03d}",
                "learning_rate": float(lr),
                "max_depth": int(depth),
                "min_child_weight": float(mcw),
                "subsample": float(ss),
                "colsample_bytree": float(cst),
                "reg_alpha": REG_ALPHA,
                "reg_lambda": float(lam),
                "gamma": GAMMA,
            }
        )
    return candidates


def build_xgb(
    config: dict,
    n_estimators: int,
    early_stopping: bool,
) -> xgb.XGBRegressor:
    params: dict = {
        "objective": OBJECTIVE,
        "eval_metric": EVAL_METRIC,
        "tree_method": TREE_METHOD,
        "n_estimators": int(n_estimators),
        "learning_rate": float(config["learning_rate"]),
        "max_depth": int(config["max_depth"]),
        "min_child_weight": float(config["min_child_weight"]),
        "subsample": float(config["subsample"]),
        "colsample_bytree": float(config["colsample_bytree"]),
        "reg_alpha": float(config["reg_alpha"]),
        "reg_lambda": float(config["reg_lambda"]),
        "gamma": float(config["gamma"]),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }
    if early_stopping:
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    return xgb.XGBRegressor(**params)


def fit_with_early_stopping(
    config: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> xgb.XGBRegressor:
    model = build_xgb(config, MAX_BOOSTING_ROUNDS, early_stopping=True)
    try:
        model.fit(
            x_train, y_train,
            eval_set=[(x_val, y_val)],
            verbose=False,
        )
        return model
    except TypeError:
        model = build_xgb(config, MAX_BOOSTING_ROUNDS, early_stopping=False)
        model.fit(
            x_train, y_train,
            eval_set=[(x_val, y_val)],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False,
        )
        return model


def best_rounds(model: xgb.XGBRegressor) -> int:
    bi = getattr(model, "best_iteration", None)
    if bi is None:
        bi = getattr(model.get_booster(), "best_iteration", None)
    if bi is None:
        return MAX_BOOSTING_ROUNDS
    return max(1, int(bi) + 1)


# ============================================================
# 5. VALIDATION METRICS
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


def validation_metrics(
    val_data: pd.DataFrame,
    predictions: np.ndarray,
) -> dict:
    scored = val_data[["Date", "Ticker", TARGET_COLUMN]].copy()
    scored["pred"] = predictions
    daily_ic_values = []
    for _, group in scored.groupby("Date", sort=True):
        ic = safe_spearman_ic(group["pred"], group[TARGET_COLUMN])
        daily_ic_values.append(ic)
    daily_arr = np.asarray(daily_ic_values, dtype=float)
    errors = predictions - val_data[TARGET_COLUMN].to_numpy(dtype=float)
    return {
        "Validation_Mean_Daily_IC": float(np.nanmean(daily_arr)),
        "Validation_Median_Daily_IC": float(np.nanmedian(daily_arr)),
        "Validation_IC_Standard_Deviation": float(
            np.nanstd(daily_arr, ddof=1)
        ),
        "Validation_RMSE": float(np.sqrt(np.mean(errors**2))),
        "Validation_MAE": float(np.mean(np.abs(errors))),
    }


# ============================================================
# 6. DAILY AND WINDOW PERFORMANCE
# ============================================================

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
    top_hr = len(
        set(pred_sorted.head(q)["Ticker"]) & set(real_sorted.head(q)["Ticker"])
    ) / q
    bot_hr = len(
        set(pred_sorted.tail(q)["Ticker"]) & set(real_sorted.tail(q)["Ticker"])
    ) / q
    return float(top_hr), float(bot_hr), q


def compute_daily_performance(
    predictions: pd.DataFrame,
    score_col: str = "XGB_Raw_Score",
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
    score_col: str = "XGB_Raw_Score",
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
        "Mean_Bottom_Quartile_Hit_Rate": float(
            daily["Bottom_Quartile_Hit_Rate"].mean()
        ),
    }


# ============================================================
# 7. NESTED HYPERPARAMETER TUNING
# ============================================================

def tune_outer_window(
    master_panel: pd.DataFrame,
    inner_windows: pd.DataFrame,
    outer_window_id: str,
    test_quarter: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate 96 candidates across 3 inner folds. Select by mean IC."""

    candidates = candidate_configurations()
    fold_rows = (
        inner_windows.loc[
            inner_windows["Outer_Window_ID"].eq(outer_window_id)
        ]
        .sort_values("Inner_Fold_Number")
        .reset_index(drop=True)
    )

    if len(fold_rows) != 3:
        raise ValueError(
            f"{outer_window_id} does not contain exactly 3 inner folds."
        )

    detailed_records: list[dict] = []

    total = len(candidates) * len(fold_rows)
    done = 0

    for candidate in candidates:
        for fold in fold_rows.itertuples(index=False):
            inner_train = master_panel.loc[
                master_panel["Date"].between(
                    pd.Timestamp(fold.Inner_Train_Start),
                    pd.Timestamp(fold.Inner_Train_End),
                    inclusive="both",
                )
            ].copy()

            val = master_panel.loc[
                master_panel["Date"].between(
                    pd.Timestamp(fold.Validation_Start),
                    pd.Timestamp(fold.Validation_End),
                    inclusive="both",
                )
            ].copy()

            # Winsorise inner
            thresh = estimate_winsorisation_thresholds(inner_train)
            inner_train_w = apply_winsorisation(inner_train, thresh)
            val_w = apply_winsorisation(val, thresh)

            model = fit_with_early_stopping(
                config=candidate,
                x_train=inner_train_w[FEATURES],
                y_train=inner_train_w[TARGET_COLUMN],
                x_val=val_w[FEATURES],
                y_val=val_w[TARGET_COLUMN],
            )

            preds = model.predict(val_w[FEATURES])
            metrics = validation_metrics(val_w, preds)
            rounds = best_rounds(model)

            done += 1
            if done % 48 == 0:
                print(
                    f"      {done}/{total} fits done for {outer_window_id}"
                )

            detailed_records.append(
                {
                    "Outer_Window_ID": outer_window_id,
                    "Test_Quarter": test_quarter,
                    "Candidate_ID": candidate["Candidate_ID"],
                    "Inner_Fold_ID": fold.Inner_Fold_ID,
                    "Inner_Fold_Number": int(fold.Inner_Fold_Number),
                    **{
                        k: v
                        for k, v in candidate.items()
                        if k != "Candidate_ID"
                    },
                    "Selected_Boosting_Rounds": rounds,
                    **metrics,
                }
            )

    detailed = pd.DataFrame(detailed_records)

    # Aggregate across folds
    agg_cols = [
        "Outer_Window_ID", "Test_Quarter", "Candidate_ID",
        "learning_rate", "max_depth", "min_child_weight",
        "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "gamma",
    ]

    aggregate = (
        detailed
        .groupby(agg_cols, as_index=False)
        .agg(
            Mean_Validation_Daily_IC=("Validation_Mean_Daily_IC", "mean"),
            Median_Validation_Daily_IC=("Validation_Median_Daily_IC", "mean"),
            Mean_Validation_RMSE=("Validation_RMSE", "mean"),
            Median_Selected_Boosting_Rounds=("Selected_Boosting_Rounds", "median"),
            Min_Selected_Boosting_Rounds=("Selected_Boosting_Rounds", "min"),
            Max_Selected_Boosting_Rounds=("Selected_Boosting_Rounds", "max"),
        )
    )

    # Primary criterion: highest mean validation IC
    # Tie-breaks: lower RMSE, shallower trees, lower lambda, fewer rounds
    aggregate = (
        aggregate
        .sort_values(
            [
                "Mean_Validation_Daily_IC",
                "Mean_Validation_RMSE",
                "max_depth",
                "reg_lambda",
                "Median_Selected_Boosting_Rounds",
                "Candidate_ID",
            ],
            ascending=[False, True, True, True, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    aggregate["Candidate_Rank"] = np.arange(1, len(aggregate) + 1)
    aggregate["Selected_Candidate"] = aggregate["Candidate_Rank"].eq(1)

    detailed = detailed.merge(
        aggregate[
            [
                "Candidate_ID",
                "Candidate_Rank",
                "Selected_Candidate",
                "Mean_Validation_Daily_IC",
                "Median_Selected_Boosting_Rounds",
            ]
        ],
        on="Candidate_ID",
        how="left",
        validate="many_to_one",
    )

    return detailed, aggregate


# ============================================================
# 8. MAIN PIPELINE
# ============================================================

def main() -> None:
    print("=" * 78)
    print("SCRIPT 08B: ALT XGBOOST  (21-DAY TARGET, 96-CANDIDATE GRID)")
    print("=" * 78)

    for path in [INPUT_MASTER_PANEL, INPUT_OUTER_WINDOWS, INPUT_INNER_WINDOWS]:
        if not path.exists():
            sys.exit(f"ERROR: Required file missing: {path}")

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    AUDIT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    overall_start = time.perf_counter()

    # -------------------------------------------------------
    # Load
    # -------------------------------------------------------
    master_panel = pd.read_csv(INPUT_MASTER_PANEL)
    master_panel["Date"] = pd.to_datetime(master_panel["Date"])
    if "Next_Date" in master_panel.columns:
        master_panel["Next_Date"] = pd.to_datetime(
            master_panel["Next_Date"], errors="coerce"
        )
    master_panel = (
        master_panel
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )

    outer_windows = pd.read_csv(INPUT_OUTER_WINDOWS)
    for col in ["Train_Start", "Train_End", "Purged_Boundary_Date",
                "Test_Start", "Test_End"]:
        outer_windows[col] = pd.to_datetime(outer_windows[col])
    outer_windows = (
        outer_windows.sort_values("Test_Start").reset_index(drop=True)
    )

    inner_windows = pd.read_csv(INPUT_INNER_WINDOWS)
    for col in [
        "Inner_Train_Start", "Inner_Train_End",
        "Inner_Purged_Boundary_Date",
        "Validation_Start", "Validation_End",
    ]:
        inner_windows[col] = pd.to_datetime(inner_windows[col])
    inner_windows = (
        inner_windows
        .sort_values(["Outer_Window_ID", "Inner_Fold_Number"])
        .reset_index(drop=True)
    )

    print(f"  Master panel: {len(master_panel):,} rows, "
          f"{master_panel['Date'].nunique()} dates")
    print(f"  Outer windows: {len(outer_windows)}")
    print(f"  Inner folds:   {len(inner_windows)}")
    print(f"  Grid size:     {len(candidate_configurations())} candidates")

    all_predictions: list[pd.DataFrame] = []
    all_search_detail: list[pd.DataFrame] = []
    all_selected: list[dict] = []
    all_thresholds: list[pd.DataFrame] = []
    estimation_records: list[dict] = []

    # -------------------------------------------------------
    # Main loop
    # -------------------------------------------------------
    for outer_row in outer_windows.itertuples(index=False):
        window_start = time.perf_counter()
        wid = outer_row.Outer_Window_ID
        print(f"\n  [{wid}]  Test quarter: {outer_row.Test_Quarter}")
        print(
            f"    Running 96-candidate × 3-fold inner search "
            f"(288 fits) ..."
        )

        # ------------------------------------------------
        # Inner hyperparameter search
        # ------------------------------------------------
        detailed_search, aggregate_search = tune_outer_window(
            master_panel=master_panel,
            inner_windows=inner_windows,
            outer_window_id=wid,
            test_quarter=outer_row.Test_Quarter,
        )

        all_search_detail.append(detailed_search)

        selected_row = aggregate_search.loc[
            aggregate_search["Selected_Candidate"]
        ].iloc[0]

        configuration = {
            "learning_rate": float(selected_row["learning_rate"]),
            "max_depth": int(selected_row["max_depth"]),
            "min_child_weight": float(selected_row["min_child_weight"]),
            "subsample": float(selected_row["subsample"]),
            "colsample_bytree": float(selected_row["colsample_bytree"]),
            "reg_alpha": float(selected_row["reg_alpha"]),
            "reg_lambda": float(selected_row["reg_lambda"]),
            "gamma": float(selected_row["gamma"]),
        }

        final_rounds = max(
            1, int(round(float(selected_row["Median_Selected_Boosting_Rounds"])))
        )

        print(
            f"    Selected: {selected_row['Candidate_ID']}  "
            f"lr={configuration['learning_rate']}  "
            f"depth={configuration['max_depth']}  "
            f"lambda={configuration['reg_lambda']}  "
            f"rounds={final_rounds}  "
            f"mean_val_IC={selected_row['Mean_Validation_Daily_IC']:.5f}"
        )

        all_selected.append(
            {
                "Outer_Window_ID": wid,
                "Test_Quarter": outer_row.Test_Quarter,
                "Selected_Candidate_ID": selected_row["Candidate_ID"],
                "Final_Boosting_Rounds": final_rounds,
                "Mean_Validation_Daily_IC": float(
                    selected_row["Mean_Validation_Daily_IC"]
                ),
                **configuration,
            }
        )

        # ------------------------------------------------
        # Outer training + test
        # ------------------------------------------------
        outer_train = master_panel.loc[
            master_panel["Date"].between(
                pd.Timestamp(outer_row.Train_Start),
                pd.Timestamp(outer_row.Train_End),
                inclusive="both",
            )
        ].copy()

        outer_test = master_panel.loc[
            master_panel["Date"].between(
                pd.Timestamp(outer_row.Test_Start),
                pd.Timestamp(outer_row.Test_End),
                inclusive="both",
            )
        ].copy()

        # Winsorise (fit on outer train only)
        outer_thresh = estimate_winsorisation_thresholds(outer_train)
        outer_train_w = apply_winsorisation(outer_train, outer_thresh)
        outer_test_w = apply_winsorisation(outer_test, outer_thresh)

        thresh_df = outer_thresh.copy()
        thresh_df.insert(0, "Outer_Window_ID", wid)
        thresh_df.insert(1, "Test_Quarter", outer_row.Test_Quarter)
        all_thresholds.append(thresh_df)

        # Retrain on full outer training set (no early stopping)
        final_model = build_xgb(
            configuration, final_rounds, early_stopping=False
        )
        final_model.fit(
            outer_train_w[FEATURES],
            outer_train_w[TARGET_COLUMN],
            verbose=False,
        )

        # Predict
        raw_scores = final_model.predict(outer_test_w[FEATURES])

        if not np.isfinite(raw_scores).all():
            raise AssertionError(
                f"Non-finite predictions produced for {wid}."
            )

        test_out = outer_test_w[
            ["Date", "Ticker", TARGET_COLUMN, "Forward_Return",
             "N_t", "Model_N_t"]
        ].copy()
        test_out["Outer_Window_ID"] = wid
        test_out["Test_Quarter"] = outer_row.Test_Quarter
        test_out["XGB_Raw_Score"] = raw_scores
        test_out["XGB_Predicted_Percentile"] = (
            test_out.groupby("Date")["XGB_Raw_Score"].rank(pct=True)
        )

        all_predictions.append(test_out)

        elapsed = time.perf_counter() - window_start
        print(f"    Window completed in {elapsed:.1f}s")

        estimation_records.append(
            {
                "Outer_Window_ID": wid,
                "Test_Quarter": outer_row.Test_Quarter,
                "Train_Observations": len(outer_train),
                "Test_Observations": len(outer_test),
                "Final_Boosting_Rounds": final_rounds,
                "Selected_Candidate_ID": selected_row["Candidate_ID"],
                "Elapsed_Seconds": elapsed,
            }
        )

    # -------------------------------------------------------
    # Assemble and save
    # -------------------------------------------------------
    full_predictions = (
        pd.concat(all_predictions, ignore_index=True)
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )

    daily_performance = compute_daily_performance(full_predictions)
    overall_stats = summarise_performance(full_predictions, daily_performance)

    window_records = []
    for wid in outer_windows["Outer_Window_ID"]:
        pred_w = full_predictions.loc[
            full_predictions["Outer_Window_ID"] == wid
        ]
        daily_w = daily_performance.loc[
            daily_performance["Outer_Window_ID"] == wid
        ]
        tq = outer_windows.loc[
            outer_windows["Outer_Window_ID"] == wid, "Test_Quarter"
        ].iloc[0]
        rec = summarise_performance(pred_w, daily_w)
        rec["Outer_Window_ID"] = wid
        rec["Test_Quarter"] = tq
        window_records.append(rec)
    window_performance = pd.DataFrame(window_records)

    full_predictions.to_csv(OUTPUT_PREDICTIONS, index=False)
    daily_performance.to_csv(OUTPUT_DAILY_PERFORMANCE, index=False)
    window_performance.to_csv(OUTPUT_WINDOW_PERFORMANCE, index=False)
    pd.DataFrame([overall_stats]).to_csv(OUTPUT_OVERALL_PERFORMANCE, index=False)
    pd.concat(all_search_detail, ignore_index=True).to_csv(
        OUTPUT_SEARCH_RESULTS, index=False
    )
    pd.DataFrame(all_selected).to_csv(OUTPUT_SELECTED_PARAMS, index=False)
    pd.concat(all_thresholds, ignore_index=True).to_csv(
        OUTPUT_WINSORISATION, index=False
    )
    pd.DataFrame(estimation_records).to_csv(
        OUTPUT_ESTIMATION_SUMMARY, index=False
    )

    total_elapsed = time.perf_counter() - overall_start

    print("\n" + "=" * 78)
    print("  OVERALL XGB PERFORMANCE  (21-Day Target, Expanded Grid)")
    print("=" * 78)
    print(f"  Out-of-sample observations: {overall_stats['Out_Of_Sample_Observations']:>9,}")
    print(f"  Out-of-sample dates:        {overall_stats['Out_Of_Sample_Dates']:>9,}")
    print(f"  Mean Daily IC:              {overall_stats['Mean_Daily_IC']:>12.6f}")
    print(f"  IC Standard Deviation:      {overall_stats['Daily_IC_Standard_Deviation']:>12.6f}")
    print(f"  IC Information Ratio:       {overall_stats['Daily_IC_Information_Ratio']:>12.6f}")
    print(f"  Positive IC Days (%):       {overall_stats['Positive_IC_Percentage']:>11.2f}%")
    print(f"  Observation MAE:            {overall_stats['Observation_MAE']:>12.6f}")
    print(f"  Observation RMSE:           {overall_stats['Observation_RMSE']:>12.6f}")
    print(f"\n  Total elapsed: {total_elapsed/60:.1f} min")
    print("\n  Script 08B complete.")


if __name__ == "__main__":
    main()
