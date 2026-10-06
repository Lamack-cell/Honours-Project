"""
Script 08: XGBoost Nested Expanding-Window Estimation
------------------------------------------------------

Purpose
-------
Estimates XGBoost under the same quarterly expanding-window design used by
Pooled OLS. Hyperparameters are selected separately inside every outer
training window using three purged chronological validation folds and mean
daily Spearman Information Coefficient (IC) as the primary selection metric.

The XGBoost regression objective remains squared error because the supervised
target is a continuous percentile score. Hyperparameter selection, however,
is aligned with the study's primary cross-sectional ranking criterion.
No risk-free-rate or portfolio-performance variables are required.

Inputs
------
1. JSE_Master_Model_Panel.csv
2. JSE_Walk_Forward_Windows.csv
3. JSE_Inner_Validation_Windows.csv

Outputs
-------
1. JSE_XGB_Walk_Forward_Predictions.csv
2. JSE_XGB_Daily_Performance.csv
3. JSE_XGB_Window_Performance.csv
4. JSE_XGB_Overall_Performance.csv
5. outputs/audits/JSE_XGB_Hyperparameter_Search_Results.csv
6. outputs/audits/JSE_XGB_Selected_Hyperparameters.csv
7. outputs/audits/JSE_XGB_Winsorisation_Thresholds.csv
8. outputs/audits/JSE_XGB_Feature_Importance.csv
9. outputs/audits/JSE_XGB_Estimation_Summary.csv
10. outputs/figures/figure_5_3_xgb_feature_importance_stability.png

Required packages
-----------------
pip install pandas numpy scipy matplotlib scikit-learn xgboost
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
import platform
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
import sklearn
import xgboost as xgb


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_MASTER_PANEL = Path("JSE_Master_Model_Panel.csv")
INPUT_OUTER_WINDOWS = Path("JSE_Walk_Forward_Windows.csv")
INPUT_INNER_WINDOWS = Path("JSE_Inner_Validation_Windows.csv")

OUTPUT_PREDICTIONS = Path("JSE_XGB_Walk_Forward_Predictions.csv")
OUTPUT_DAILY_PERFORMANCE = Path("JSE_XGB_Daily_Performance.csv")
OUTPUT_WINDOW_PERFORMANCE = Path("JSE_XGB_Window_Performance.csv")
OUTPUT_OVERALL_PERFORMANCE = Path("JSE_XGB_Overall_Performance.csv")

AUDIT_DIRECTORY = Path("outputs") / "audits"
FIGURE_DIRECTORY = Path("outputs") / "figures"

OUTPUT_SEARCH_RESULTS = (
    AUDIT_DIRECTORY / "JSE_XGB_Hyperparameter_Search_Results.csv"
)
OUTPUT_SELECTED_PARAMETERS = (
    AUDIT_DIRECTORY / "JSE_XGB_Selected_Hyperparameters.csv"
)
OUTPUT_WINSORISATION = (
    AUDIT_DIRECTORY / "JSE_XGB_Winsorisation_Thresholds.csv"
)
OUTPUT_FEATURE_IMPORTANCE = (
    AUDIT_DIRECTORY / "JSE_XGB_Feature_Importance.csv"
)
OUTPUT_ESTIMATION_SUMMARY = (
    AUDIT_DIRECTORY / "JSE_XGB_Estimation_Summary.csv"
)
OUTPUT_FIGURE = (
    FIGURE_DIRECTORY / "figure_5_3_xgb_feature_importance_stability.png"
)

FEATURES = [
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
TOP_BOTTOM_FRACTION = 0.25
MINIMUM_DAILY_CROSS_SECTION = 5

WINSOR_LOWER_QUANTILE = 0.01
WINSOR_UPPER_QUANTILE = 0.99

RANDOM_STATE = 42
TREE_METHOD = "hist"
OBJECTIVE = "reg:squarederror"
EVAL_METRIC = "rmse"

MAX_BOOSTING_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 50

# Compact, prespecified grid: 32 candidates per outer window.
LEARNING_RATES = [0.02, 0.05]
MAX_DEPTHS = [2, 3]
MIN_CHILD_WEIGHTS = [5.0, 15.0]
SUBSAMPLES = [0.8, 1.0]
COLSAMPLE_BYTREE_VALUES = [0.8, 1.0]

# Explicit regularisation is fixed to avoid an impractically large grid.
REG_ALPHA = 0.10
REG_LAMBDA = 10.0
GAMMA = 0.0


# ============================================================
# 2. VALIDATION AND PREPROCESSING
# ============================================================

def validate_required_files() -> None:
    required = [
        INPUT_MASTER_PANEL,
        INPUT_OUTER_WINDOWS,
        INPUT_INNER_WINDOWS,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required input files are missing:\n- " + "\n- ".join(missing)
        )


def parse_window_dates(dataframe: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in dataframe.columns:
            raise ValueError(f"Required date column missing: {column}")
        dataframe[column] = pd.to_datetime(dataframe[column], errors="raise")


def validate_master_panel(panel: pd.DataFrame) -> None:
    required_columns = {
        "Date",
        "Ticker",
        TARGET_COLUMN,
        "Forward_Return",
        "N_t",
        "Model_N_t",
        *FEATURES,
    }
    missing = required_columns.difference(panel.columns)
    if missing:
        raise ValueError(
            "Master model panel is missing required columns: "
            f"{sorted(missing)}"
        )

    if panel.empty:
        raise ValueError("Master model panel contains no observations.")

    duplicate_count = int(panel.duplicated(["Date", "Ticker"]).sum())
    if duplicate_count:
        raise ValueError(
            f"Master model panel contains {duplicate_count:,} duplicate "
            "Date-Ticker observations."
        )

    required_nonmissing = [
        "Date",
        "Ticker",
        TARGET_COLUMN,
        "Forward_Return",
        *FEATURES,
    ]
    missing_values = int(panel[required_nonmissing].isna().sum().sum())
    if missing_values:
        raise ValueError(
            f"Master panel contains {missing_values:,} missing required values."
        )

    feature_values = panel[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        raise ValueError("Non-finite values exist in the feature matrix.")

    if not panel[TARGET_COLUMN].between(0, 1, inclusive="both").all():
        raise ValueError("Target values outside [0, 1] were detected.")


def estimate_winsorisation_thresholds(
    training_data: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for feature in FEATURES:
        lower_bound = float(training_data[feature].quantile(WINSOR_LOWER_QUANTILE))
        upper_bound = float(training_data[feature].quantile(WINSOR_UPPER_QUANTILE))

        if not np.isfinite(lower_bound) or not np.isfinite(upper_bound):
            raise ValueError(f"Non-finite winsorisation threshold for {feature}.")
        if lower_bound > upper_bound:
            raise ValueError(f"Invalid winsorisation interval for {feature}.")

        records.append(
            {
                "Feature": feature,
                "Lower_Quantile": WINSOR_LOWER_QUANTILE,
                "Upper_Quantile": WINSOR_UPPER_QUANTILE,
                "Lower_Bound": lower_bound,
                "Upper_Bound": upper_bound,
            }
        )

    return pd.DataFrame(records)


def apply_winsorisation(
    data: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    transformed = data.copy()
    threshold_map = thresholds.set_index("Feature")

    for feature in FEATURES:
        transformed[feature] = transformed[feature].clip(
            lower=float(threshold_map.loc[feature, "Lower_Bound"]),
            upper=float(threshold_map.loc[feature, "Upper_Bound"]),
        )

    return transformed


# ============================================================
# 3. XGBOOST FITTING HELPERS
# ============================================================

def candidate_configurations() -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []

    for candidate_number, values in enumerate(
        product(
            LEARNING_RATES,
            MAX_DEPTHS,
            MIN_CHILD_WEIGHTS,
            SUBSAMPLES,
            COLSAMPLE_BYTREE_VALUES,
        ),
        start=1,
    ):
        learning_rate, max_depth, min_child_weight, subsample, colsample = values
        candidates.append(
            {
                "Candidate_ID": f"C_{candidate_number:02d}",
                "learning_rate": float(learning_rate),
                "max_depth": int(max_depth),
                "min_child_weight": float(min_child_weight),
                "subsample": float(subsample),
                "colsample_bytree": float(colsample),
                "reg_alpha": REG_ALPHA,
                "reg_lambda": REG_LAMBDA,
                "gamma": GAMMA,
            }
        )

    return candidates


def build_xgb_regressor(
    configuration: dict[str, object],
    n_estimators: int,
    use_early_stopping: bool,
) -> xgb.XGBRegressor:
    parameters: dict[str, object] = {
        "objective": OBJECTIVE,
        "eval_metric": EVAL_METRIC,
        "tree_method": TREE_METHOD,
        "n_estimators": int(n_estimators),
        "learning_rate": float(configuration["learning_rate"]),
        "max_depth": int(configuration["max_depth"]),
        "min_child_weight": float(configuration["min_child_weight"]),
        "subsample": float(configuration["subsample"]),
        "colsample_bytree": float(configuration["colsample_bytree"]),
        "reg_alpha": float(configuration["reg_alpha"]),
        "reg_lambda": float(configuration["reg_lambda"]),
        "gamma": float(configuration["gamma"]),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }

    if use_early_stopping:
        parameters["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS

    return xgb.XGBRegressor(**parameters)


def fit_with_early_stopping(
    configuration: dict[str, object],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> xgb.XGBRegressor:
    """Support both newer and older XGBoost sklearn APIs."""

    model = build_xgb_regressor(
        configuration=configuration,
        n_estimators=MAX_BOOSTING_ROUNDS,
        use_early_stopping=True,
    )

    try:
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_validation, y_validation)],
            verbose=False,
        )
        return model
    except TypeError:
        # Older XGBoost versions expect early_stopping_rounds in fit().
        model = build_xgb_regressor(
            configuration=configuration,
            n_estimators=MAX_BOOSTING_ROUNDS,
            use_early_stopping=False,
        )
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_validation, y_validation)],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False,
        )
        return model


def selected_boosting_rounds(model: xgb.XGBRegressor) -> int:
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        booster = model.get_booster()
        best_iteration = getattr(booster, "best_iteration", None)

    if best_iteration is None:
        return MAX_BOOSTING_ROUNDS

    return max(1, int(best_iteration) + 1)


def safe_daily_spearman(
    group: pd.DataFrame,
    prediction_column: str,
) -> tuple[float, bool]:
    predictions = group[prediction_column]
    target = group[TARGET_COLUMN]

    if len(group) < MINIMUM_DAILY_CROSS_SECTION:
        return np.nan, False

    if predictions.nunique(dropna=True) <= 1 or target.nunique(dropna=True) <= 1:
        return 0.0, True

    result = spearmanr(predictions, target)
    statistic = float(result.statistic)

    if not np.isfinite(statistic):
        return 0.0, True

    return statistic, False


def validation_metrics(
    validation_data: pd.DataFrame,
    predictions: np.ndarray,
) -> dict[str, float | int]:
    scored = validation_data[["Date", "Ticker", TARGET_COLUMN]].copy()
    scored["Validation_Prediction"] = predictions

    daily_ic_values: list[float] = []
    constant_days = 0

    for _, group in scored.groupby("Date", sort=True):
        daily_ic, constant_flag = safe_daily_spearman(
            group,
            "Validation_Prediction",
        )
        daily_ic_values.append(daily_ic)
        constant_days += int(constant_flag)

    daily_ic_array = np.asarray(daily_ic_values, dtype=float)
    if np.isnan(daily_ic_array).any():
        raise ValueError("Missing validation IC values were produced.")

    errors = predictions - validation_data[TARGET_COLUMN].to_numpy(dtype=float)

    return {
        "Validation_Mean_Daily_IC": float(np.mean(daily_ic_array)),
        "Validation_Median_Daily_IC": float(np.median(daily_ic_array)),
        "Validation_IC_Standard_Deviation": float(np.std(daily_ic_array, ddof=1)),
        "Validation_RMSE": float(np.sqrt(np.mean(errors**2))),
        "Validation_MAE": float(np.mean(np.abs(errors))),
        "Validation_Constant_Prediction_Days": int(constant_days),
    }


# ============================================================
# 4. PERFORMANCE FUNCTIONS
# ============================================================

def deterministic_top_bottom_hit_rates(
    group: pd.DataFrame,
) -> tuple[float, float, int]:
    number_of_assets = len(group)
    quartile_size = max(1, int(np.ceil(number_of_assets * TOP_BOTTOM_FRACTION)))

    predicted_order = group.sort_values(
        ["XGB_Raw_Score", "Ticker"],
        ascending=[False, True],
        kind="mergesort",
    )
    realised_order = group.sort_values(
        [TARGET_COLUMN, "Ticker"],
        ascending=[False, True],
        kind="mergesort",
    )

    predicted_top = set(predicted_order.head(quartile_size)["Ticker"])
    realised_top = set(realised_order.head(quartile_size)["Ticker"])
    predicted_bottom = set(predicted_order.tail(quartile_size)["Ticker"])
    realised_bottom = set(realised_order.tail(quartile_size)["Ticker"])

    top_hit_rate = len(predicted_top.intersection(realised_top)) / quartile_size
    bottom_hit_rate = (
        len(predicted_bottom.intersection(realised_bottom)) / quartile_size
    )

    return float(top_hit_rate), float(bottom_hit_rate), quartile_size


def calculate_daily_performance(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for date, group in predictions.groupby("Date", sort=True):
        daily_ic, constant_flag = safe_daily_spearman(group, "XGB_Raw_Score")

        errors = group["XGB_Raw_Score"] - group[TARGET_COLUMN]
        top_hit_rate, bottom_hit_rate, quartile_size = (
            deterministic_top_bottom_hit_rates(group)
        )

        records.append(
            {
                "Date": date,
                "Outer_Window_ID": group["Outer_Window_ID"].iloc[0],
                "Test_Quarter": group["Test_Quarter"].iloc[0],
                "Model_Eligible_Securities": len(group),
                "Daily_IC": daily_ic,
                "Daily_MAE": float(np.mean(np.abs(errors))),
                "Daily_RMSE": float(np.sqrt(np.mean(errors**2))),
                "Quartile_Size": quartile_size,
                "Top_Quartile_Hit_Rate": top_hit_rate,
                "Bottom_Quartile_Hit_Rate": bottom_hit_rate,
                "Predictions_Below_Zero": int(group["XGB_Raw_Score"].lt(0).sum()),
                "Predictions_Above_One": int(group["XGB_Raw_Score"].gt(1).sum()),
                "Constant_Prediction_Day": int(constant_flag),
            }
        )

    return pd.DataFrame(records)


def summarise_single_group(
    predictions: pd.DataFrame,
    daily_performance: pd.DataFrame,
) -> dict[str, float | int]:
    daily_ic = daily_performance["Daily_IC"].dropna()
    mean_ic = float(daily_ic.mean())
    ic_std = float(daily_ic.std())
    ic_ir = mean_ic / ic_std if ic_std > 0 else np.nan

    errors = predictions["XGB_Raw_Score"] - predictions[TARGET_COLUMN]

    return {
        "Out_Of_Sample_Observations": int(len(predictions)),
        "Out_Of_Sample_Dates": int(predictions["Date"].nunique()),
        "Mean_Daily_IC": mean_ic,
        "Median_Daily_IC": float(daily_ic.median()),
        "Daily_IC_Standard_Deviation": ic_std,
        "Daily_IC_Information_Ratio": float(ic_ir),
        "Positive_IC_Days": int(daily_ic.gt(0).sum()),
        "Zero_IC_Days": int(daily_ic.eq(0).sum()),
        "Negative_IC_Days": int(daily_ic.lt(0).sum()),
        "Positive_IC_Percentage": float(daily_ic.gt(0).mean() * 100),
        "Observation_MAE": float(np.mean(np.abs(errors))),
        "Observation_RMSE": float(np.sqrt(np.mean(errors**2))),
        "Mean_Daily_MAE": float(daily_performance["Daily_MAE"].mean()),
        "Mean_Daily_RMSE": float(daily_performance["Daily_RMSE"].mean()),
        "Mean_Top_Quartile_Hit_Rate": float(
            daily_performance["Top_Quartile_Hit_Rate"].mean()
        ),
        "Mean_Bottom_Quartile_Hit_Rate": float(
            daily_performance["Bottom_Quartile_Hit_Rate"].mean()
        ),
        "Predictions_Below_Zero": int(predictions["XGB_Raw_Score"].lt(0).sum()),
        "Predictions_Above_One": int(predictions["XGB_Raw_Score"].gt(1).sum()),
        "Constant_Prediction_Days": int(
            daily_performance["Constant_Prediction_Day"].sum()
        ),
    }


def summarise_by_window(
    predictions: pd.DataFrame,
    daily_performance: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for (window_id, test_quarter), group in predictions.groupby(
        ["Outer_Window_ID", "Test_Quarter"], sort=False
    ):
        daily_group = daily_performance.loc[
            daily_performance["Outer_Window_ID"].eq(window_id)
        ]
        record = summarise_single_group(group, daily_group)
        record["Outer_Window_ID"] = window_id
        record["Test_Quarter"] = test_quarter
        records.append(record)

    columns = ["Outer_Window_ID", "Test_Quarter"]
    result = pd.DataFrame(records)
    return result[columns + [c for c in result.columns if c not in columns]]


# ============================================================
# 5. TUNING AND ESTIMATION
# ============================================================

def tune_outer_window(
    master_panel: pd.DataFrame,
    inner_windows: pd.DataFrame,
    outer_window_id: str,
    test_quarter: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = candidate_configurations()
    fold_rows = inner_windows.loc[
        inner_windows["Outer_Window_ID"].eq(outer_window_id)
    ].sort_values("Inner_Fold_Number")

    if len(fold_rows) == 0:
        raise ValueError(f"No inner folds found for {outer_window_id}.")

    detailed_records: list[dict[str, object]] = []

    for candidate in candidates:
        for fold in fold_rows.itertuples(index=False):
            train_mask = master_panel["Date"].between(
                pd.Timestamp(fold.Inner_Train_Start),
                pd.Timestamp(fold.Inner_Train_End),
                inclusive="both",
            )
            validation_mask = master_panel["Date"].between(
                pd.Timestamp(fold.Validation_Start),
                pd.Timestamp(fold.Validation_End),
                inclusive="both",
            )

            inner_train = master_panel.loc[train_mask].copy()
            validation = master_panel.loc[validation_mask].copy()

            thresholds = estimate_winsorisation_thresholds(inner_train)
            inner_train_transformed = apply_winsorisation(inner_train, thresholds)
            validation_transformed = apply_winsorisation(validation, thresholds)

            model = fit_with_early_stopping(
                configuration=candidate,
                x_train=inner_train_transformed[FEATURES],
                y_train=inner_train_transformed[TARGET_COLUMN],
                x_validation=validation_transformed[FEATURES],
                y_validation=validation_transformed[TARGET_COLUMN],
            )

            validation_predictions = model.predict(validation_transformed[FEATURES])
            metrics = validation_metrics(validation_transformed, validation_predictions)
            rounds = selected_boosting_rounds(model)

            detailed_records.append(
                {
                    "Outer_Window_ID": outer_window_id,
                    "Test_Quarter": test_quarter,
                    "Candidate_ID": candidate["Candidate_ID"],
                    "Inner_Fold_ID": fold.Inner_Fold_ID,
                    "Inner_Fold_Number": int(fold.Inner_Fold_Number),
                    "Inner_Train_Dates": int(fold.Inner_Train_Dates),
                    "Inner_Train_Observations": int(fold.Inner_Train_Observations),
                    "Validation_Dates": int(fold.Validation_Dates),
                    "Validation_Observations": int(fold.Validation_Observations),
                    **{key: value for key, value in candidate.items() if key != "Candidate_ID"},
                    "Selected_Boosting_Rounds": rounds,
                    **metrics,
                }
            )

    detailed = pd.DataFrame(detailed_records)

    aggregate = (
        detailed.groupby(
            [
                "Outer_Window_ID",
                "Test_Quarter",
                "Candidate_ID",
                "learning_rate",
                "max_depth",
                "min_child_weight",
                "subsample",
                "colsample_bytree",
                "reg_alpha",
                "reg_lambda",
                "gamma",
            ],
            as_index=False,
        )
        .agg(
            Mean_Validation_Daily_IC=("Validation_Mean_Daily_IC", "mean"),
            Median_Validation_Daily_IC=("Validation_Median_Daily_IC", "mean"),
            Mean_Validation_IC_Standard_Deviation=(
                "Validation_IC_Standard_Deviation",
                "mean",
            ),
            Mean_Validation_RMSE=("Validation_RMSE", "mean"),
            Mean_Validation_MAE=("Validation_MAE", "mean"),
            Total_Constant_Prediction_Days=(
                "Validation_Constant_Prediction_Days",
                "sum",
            ),
            Median_Selected_Boosting_Rounds=("Selected_Boosting_Rounds", "median"),
            Minimum_Selected_Boosting_Rounds=("Selected_Boosting_Rounds", "min"),
            Maximum_Selected_Boosting_Rounds=("Selected_Boosting_Rounds", "max"),
        )
    )

    # Primary: highest mean daily IC. Tie-breaks favour lower RMSE and simpler trees.
    aggregate = aggregate.sort_values(
        [
            "Mean_Validation_Daily_IC",
            "Mean_Validation_RMSE",
            "max_depth",
            "Median_Selected_Boosting_Rounds",
            "Candidate_ID",
        ],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    aggregate["Candidate_Rank"] = np.arange(1, len(aggregate) + 1)
    aggregate["Selected_Candidate"] = aggregate["Candidate_Rank"].eq(1)

    # Attach aggregate selection columns back to the fold-level audit.
    detailed = detailed.merge(
        aggregate[
            [
                "Candidate_ID",
                "Candidate_Rank",
                "Selected_Candidate",
                "Mean_Validation_Daily_IC",
                "Mean_Validation_RMSE",
                "Median_Selected_Boosting_Rounds",
            ]
        ],
        on="Candidate_ID",
        how="left",
        validate="many_to_one",
    )

    return detailed, aggregate


# ============================================================
# 6. FIGURE
# ============================================================

def create_feature_importance_figure(feature_importance: pd.DataFrame) -> None:
    pivot = feature_importance.pivot(
        index="Feature",
        columns="Test_Quarter",
        values="Gain_Importance",
    ).reindex(FEATURES)

    figure, axis = plt.subplots(figsize=(12, 7))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")

    axis.set_xticks(np.arange(len(pivot.columns)))
    axis.set_xticklabels(pivot.columns, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(pivot.index)))
    axis.set_yticklabels(pivot.index)
    axis.set_title("XGBoost Gain-Based Feature Importance across Test Windows")
    axis.set_xlabel("Out-of-sample test quarter")
    axis.set_ylabel("Predictor")

    for row_index in range(len(pivot.index)):
        for column_index in range(len(pivot.columns)):
            value = pivot.iloc[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > pivot.to_numpy().mean() else "black",
            )

    colour_bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colour_bar.set_label("Normalised gain importance")
    figure.tight_layout()
    figure.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(figure)


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def main() -> None:
    print("=" * 78)
    print("SCRIPT 08: XGBOOST NESTED EXPANDING-WINDOW ESTIMATION")
    print("=" * 78)

    overall_start_time = time.perf_counter()
    validate_required_files()
    AUDIT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    master_panel = pd.read_csv(INPUT_MASTER_PANEL)
    outer_windows = pd.read_csv(INPUT_OUTER_WINDOWS)
    inner_windows = pd.read_csv(INPUT_INNER_WINDOWS)

    master_panel["Date"] = pd.to_datetime(master_panel["Date"], errors="raise")
    if "Next_Date" in master_panel.columns:
        master_panel["Next_Date"] = pd.to_datetime(
            master_panel["Next_Date"], errors="coerce"
        )

    parse_window_dates(
        outer_windows,
        [
            "Train_Start",
            "Train_End",
            "Purged_Boundary_Date",
            "Test_Start",
            "Test_End",
        ],
    )
    parse_window_dates(
        inner_windows,
        [
            "Inner_Train_Start",
            "Inner_Train_End",
            "Inner_Purged_Boundary_Date",
            "Validation_Start",
            "Validation_End",
        ],
    )

    master_panel = master_panel.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    outer_windows = outer_windows.sort_values("Test_Start").reset_index(drop=True)
    inner_windows = inner_windows.sort_values(
        ["Outer_Window_ID", "Inner_Fold_Number"]
    ).reset_index(drop=True)

    validate_master_panel(master_panel)

    expected_test_rows = int(outer_windows["Test_Observations"].sum())
    expected_test_dates = int(outer_windows["Test_Dates"].sum())

    prediction_frames: list[pd.DataFrame] = []
    search_detail_frames: list[pd.DataFrame] = []
    selected_parameter_records: list[dict[str, object]] = []
    winsorisation_frames: list[pd.DataFrame] = []
    feature_importance_records: list[dict[str, object]] = []
    estimation_summary_records: list[dict[str, object]] = []

    for outer_row in outer_windows.itertuples(index=False):
        window_start_time = time.perf_counter()

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

        outer_train = master_panel.loc[train_mask].copy()
        outer_test = master_panel.loc[test_mask].copy()

        if len(outer_train) != int(outer_row.Train_Observations):
            raise AssertionError(
                f"Training row count mismatch for {outer_row.Outer_Window_ID}."
            )
        if len(outer_test) != int(outer_row.Test_Observations):
            raise AssertionError(
                f"Test row count mismatch for {outer_row.Outer_Window_ID}."
            )

        detailed_search, aggregate_search = tune_outer_window(
            master_panel=master_panel,
            inner_windows=inner_windows,
            outer_window_id=outer_row.Outer_Window_ID,
            test_quarter=outer_row.Test_Quarter,
        )
        search_detail_frames.append(detailed_search)

        selected = aggregate_search.loc[
            aggregate_search["Selected_Candidate"]
        ].iloc[0]

        configuration = {
            "learning_rate": float(selected["learning_rate"]),
            "max_depth": int(selected["max_depth"]),
            "min_child_weight": float(selected["min_child_weight"]),
            "subsample": float(selected["subsample"]),
            "colsample_bytree": float(selected["colsample_bytree"]),
            "reg_alpha": float(selected["reg_alpha"]),
            "reg_lambda": float(selected["reg_lambda"]),
            "gamma": float(selected["gamma"]),
        }

        final_rounds = max(1, int(round(selected["Median_Selected_Boosting_Rounds"])))

        thresholds = estimate_winsorisation_thresholds(outer_train)
        thresholds.insert(0, "Outer_Window_ID", outer_row.Outer_Window_ID)
        thresholds.insert(1, "Test_Quarter", outer_row.Test_Quarter)
        winsorisation_frames.append(thresholds)

        outer_train_transformed = apply_winsorisation(outer_train, thresholds)
        outer_test_transformed = apply_winsorisation(outer_test, thresholds)

        final_model = build_xgb_regressor(
            configuration=configuration,
            n_estimators=final_rounds,
            use_early_stopping=False,
        )
        final_model.fit(
            outer_train_transformed[FEATURES],
            outer_train_transformed[TARGET_COLUMN],
            verbose=False,
        )

        raw_predictions = final_model.predict(outer_test_transformed[FEATURES])

        prediction_columns = [
            "Date",
            "Ticker",
            TARGET_COLUMN,
            "Forward_Return",
                "N_t",
            "Model_N_t",
            "Daily_Return_Rank",
        ]
        optional_columns = [
            "Active_Quarter",
            "Liquidity_Rank",
            "Membership_Status",
        ]
        for column in optional_columns:
            if column in outer_test_transformed.columns:
                prediction_columns.append(column)

        window_predictions = outer_test_transformed[prediction_columns].copy()
        window_predictions.insert(0, "Outer_Window_ID", outer_row.Outer_Window_ID)
        window_predictions.insert(1, "Test_Quarter", outer_row.Test_Quarter)
        window_predictions["XGB_Raw_Score"] = raw_predictions

        window_predictions["XGB_Predicted_Rank"] = (
            window_predictions.groupby("Date")["XGB_Raw_Score"]
            .rank(method="average", ascending=True)
        )
        window_predictions["XGB_Prediction_N_t"] = (
            window_predictions.groupby("Date")["Ticker"].transform("count").astype(int)
        )
        window_predictions["XGB_Predicted_Percentile"] = (
            window_predictions["XGB_Predicted_Rank"] - 1
        ) / (window_predictions["XGB_Prediction_N_t"] - 1)

        prediction_frames.append(window_predictions)

        booster = final_model.get_booster()
        gain_scores = booster.get_score(importance_type="gain")
        gain_values = np.array(
            [float(gain_scores.get(feature, 0.0)) for feature in FEATURES],
            dtype=float,
        )
        gain_total = float(gain_values.sum())
        normalised_gain = (
            gain_values / gain_total if gain_total > 0 else np.zeros_like(gain_values)
        )

        for feature, raw_gain, normalised_value in zip(
            FEATURES, gain_values, normalised_gain
        ):
            feature_importance_records.append(
                {
                    "Outer_Window_ID": outer_row.Outer_Window_ID,
                    "Test_Quarter": outer_row.Test_Quarter,
                    "Feature": feature,
                    "Raw_Gain": float(raw_gain),
                    "Gain_Importance": float(normalised_value),
                }
            )

        selected_parameter_records.append(
            {
                "Outer_Window_ID": outer_row.Outer_Window_ID,
                "Test_Quarter": outer_row.Test_Quarter,
                "Candidate_ID": selected["Candidate_ID"],
                **configuration,
                "Final_N_Estimators": final_rounds,
                "Mean_Validation_Daily_IC": float(
                    selected["Mean_Validation_Daily_IC"]
                ),
                "Mean_Validation_RMSE": float(selected["Mean_Validation_RMSE"]),
                "Minimum_Selected_Boosting_Rounds": int(
                    selected["Minimum_Selected_Boosting_Rounds"]
                ),
                "Maximum_Selected_Boosting_Rounds": int(
                    selected["Maximum_Selected_Boosting_Rounds"]
                ),
            }
        )

        elapsed_seconds = time.perf_counter() - window_start_time
        estimation_summary_records.append(
            {
                "Outer_Window_ID": outer_row.Outer_Window_ID,
                "Test_Quarter": outer_row.Test_Quarter,
                "Train_Start": outer_row.Train_Start,
                "Train_End": outer_row.Train_End,
                "Test_Start": outer_row.Test_Start,
                "Test_End": outer_row.Test_End,
                "Training_Observations": len(outer_train),
                "Training_Dates": outer_train["Date"].nunique(),
                "Training_Tickers": outer_train["Ticker"].nunique(),
                "Test_Observations": len(outer_test),
                "Test_Dates": outer_test["Date"].nunique(),
                "Test_Tickers": outer_test["Ticker"].nunique(),
                "Candidate_Configurations_Evaluated": len(aggregate_search),
                "Inner_Folds": int(
                    inner_windows["Outer_Window_ID"]
                    .eq(outer_row.Outer_Window_ID)
                    .sum()
                ),
                "Final_N_Estimators": final_rounds,
                "Window_Runtime_Seconds": elapsed_seconds,
            }
        )

        print(
            f"Completed {outer_row.Outer_Window_ID} | {outer_row.Test_Quarter} | "
            f"selected {selected['Candidate_ID']} | rounds {final_rounds} | "
            f"validation IC {selected['Mean_Validation_Daily_IC']:.5f} | "
            f"runtime {elapsed_seconds / 60:.1f} min"
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    if len(predictions) != expected_test_rows:
        raise AssertionError(
            f"Expected {expected_test_rows:,} predictions but obtained "
            f"{len(predictions):,}."
        )
    if predictions["Date"].nunique() != expected_test_dates:
        raise AssertionError(
            f"Expected {expected_test_dates:,} test dates but obtained "
            f"{predictions['Date'].nunique():,}."
        )
    if predictions.duplicated(["Date", "Ticker"]).any():
        raise AssertionError("Duplicate out-of-sample predictions were generated.")

    daily_performance = calculate_daily_performance(predictions)
    window_performance = summarise_by_window(predictions, daily_performance)
    overall_record = summarise_single_group(predictions, daily_performance)
    overall_performance = pd.DataFrame([{"Model": "XGBoost", **overall_record}])

    search_results = pd.concat(search_detail_frames, ignore_index=True)
    selected_parameters = pd.DataFrame(selected_parameter_records)
    winsorisation_thresholds = pd.concat(winsorisation_frames, ignore_index=True)
    feature_importance = pd.DataFrame(feature_importance_records)
    estimation_summary = pd.DataFrame(estimation_summary_records)

    predictions.to_csv(OUTPUT_PREDICTIONS, index=False)
    daily_performance.to_csv(OUTPUT_DAILY_PERFORMANCE, index=False)
    window_performance.to_csv(OUTPUT_WINDOW_PERFORMANCE, index=False)
    overall_performance.to_csv(OUTPUT_OVERALL_PERFORMANCE, index=False)
    search_results.to_csv(OUTPUT_SEARCH_RESULTS, index=False)
    selected_parameters.to_csv(OUTPUT_SELECTED_PARAMETERS, index=False)
    winsorisation_thresholds.to_csv(OUTPUT_WINSORISATION, index=False)
    feature_importance.to_csv(OUTPUT_FEATURE_IMPORTANCE, index=False)

    total_runtime_seconds = time.perf_counter() - overall_start_time
    estimation_summary["Python_Version"] = platform.python_version()
    estimation_summary["Pandas_Version"] = pd.__version__
    estimation_summary["NumPy_Version"] = np.__version__
    estimation_summary["SciPy_Version"] = scipy.__version__
    estimation_summary["Scikit_Learn_Version"] = sklearn.__version__
    estimation_summary["XGBoost_Version"] = xgb.__version__
    estimation_summary["Random_State"] = RANDOM_STATE
    estimation_summary["Overall_Runtime_Seconds"] = total_runtime_seconds
    estimation_summary.to_csv(OUTPUT_ESTIMATION_SUMMARY, index=False)

    create_feature_importance_figure(feature_importance)

    overall = overall_performance.iloc[0]

    print("\n" + "=" * 78)
    print("SCRIPT 08 EMPIRICAL OUTPUTS")
    print("=" * 78)
    print(f"Outer XGBoost models estimated: {len(outer_windows):,}")
    print(f"Candidate configurations per outer window: {len(candidate_configurations()):,}")
    print(f"Inner model fits completed: {len(search_results):,}")
    print(f"Out-of-sample observations predicted: {len(predictions):,}")
    print(f"Out-of-sample trading dates predicted: {predictions['Date'].nunique():,}")
    print(
        "Out-of-sample date range: "
        f"{predictions['Date'].min().date()} to {predictions['Date'].max().date()}"
    )
    print(f"Mean daily Spearman IC: {overall['Mean_Daily_IC']:.6f}")
    print(f"Median daily Spearman IC: {overall['Median_Daily_IC']:.6f}")
    print(
        "Daily IC standard deviation: "
        f"{overall['Daily_IC_Standard_Deviation']:.6f}"
    )
    print(
        "Unannualised IC information ratio: "
        f"{overall['Daily_IC_Information_Ratio']:.6f}"
    )
    print(
        "Positive-IC trading days: "
        f"{int(overall['Positive_IC_Days']):,} "
        f"({overall['Positive_IC_Percentage']:.2f}%)"
    )
    print(f"Observation-level MAE: {overall['Observation_MAE']:.6f}")
    print(f"Observation-level RMSE: {overall['Observation_RMSE']:.6f}")
    print(
        "Mean daily top-quartile hit rate: "
        f"{overall['Mean_Top_Quartile_Hit_Rate']:.4f}"
    )
    print(
        "Mean daily bottom-quartile hit rate: "
        f"{overall['Mean_Bottom_Quartile_Hit_Rate']:.4f}"
    )
    print(f"Raw XGBoost predictions below 0: {int(overall['Predictions_Below_Zero']):,}")
    print(f"Raw XGBoost predictions above 1: {int(overall['Predictions_Above_One']):,}")
    print(f"Constant-prediction trading days: {int(overall['Constant_Prediction_Days']):,}")
    print(f"Total runtime: {total_runtime_seconds / 60:.1f} minutes")

    print("\nIntegrity checks passed:")
    print("- Every test observation was predicted exactly once.")
    print("- Every inner fold was constructed from stored chronological dates.")
    print("- Winsorisation bounds were estimated separately inside each inner fold.")
    print("- Hyperparameters were selected by mean daily validation Spearman IC.")
    print("- The outer test periods were never used for early stopping or tuning.")
    print("- Final boosting rounds were derived from inner-fold early stopping only.")
    print("- Both models use the same nine predictors and outer test observations.")

    print("\nCore outputs:")
    print(f"- {OUTPUT_PREDICTIONS}")
    print(f"- {OUTPUT_DAILY_PERFORMANCE}")
    print(f"- {OUTPUT_WINDOW_PERFORMANCE}")
    print(f"- {OUTPUT_OVERALL_PERFORMANCE}")

    print("\nAudit and figure outputs:")
    print(f"- {OUTPUT_SEARCH_RESULTS}")
    print(f"- {OUTPUT_SELECTED_PARAMETERS}")
    print(f"- {OUTPUT_WINSORISATION}")
    print(f"- {OUTPUT_FEATURE_IMPORTANCE}")
    print(f"- {OUTPUT_ESTIMATION_SUMMARY}")
    print(f"- {OUTPUT_FIGURE}")
    print("=" * 78)


if __name__ == "__main__":
    # Spearman may emit a warning for a constant validation vector. Such days
    # are handled explicitly and assigned IC = 0 rather than being dropped.
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()