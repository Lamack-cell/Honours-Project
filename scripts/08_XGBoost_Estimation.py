"""
Script 08: XGBoost Nested Expanding-Window Estimation
------------------------------------------------------

Purpose
-------
Estimates XGBoost under the same quarterly expanding-window design used
by Pooled OLS.

Hyperparameters are selected separately within every outer training
window using three purged chronological validation folds.

Important distinction
---------------------
- RMSE controls early stopping within each inner fold.
- Mean daily Spearman Information Coefficient selects the best
  hyperparameter candidate.
- The outer test period is never used for tuning or early stopping.

Inputs
------
data/processed/06_validation/JSE_Master_Model_Panel.csv
data/processed/06_validation/JSE_Walk_Forward_Windows.csv
data/processed/06_validation/JSE_Inner_Validation_Windows.csv

Results
-------
results/08_xgboost/JSE_XGB_Walk_Forward_Predictions.csv
results/08_xgboost/JSE_XGB_Daily_Performance.csv
results/08_xgboost/JSE_XGB_Window_Performance.csv
results/08_xgboost/JSE_XGB_Overall_Performance.csv

Audit outputs
-------------
outputs/audits/08_xgboost/JSE_XGB_Hyperparameter_Search_Results.csv
outputs/audits/08_xgboost/JSE_XGB_Selected_Hyperparameters.csv
outputs/audits/08_xgboost/JSE_XGB_Winsorisation_Thresholds.csv
outputs/audits/08_xgboost/JSE_XGB_Feature_Importance.csv
outputs/audits/08_xgboost/JSE_XGB_Estimation_Summary.csv

Figure output
-------------
outputs/figures/chapter_5/
figure_5_3_xgb_feature_importance_stability.png

Required packages
-----------------
pip install pandas numpy scipy matplotlib scikit-learn xgboost

Run from the project root
-------------------------
python .\\scripts\\08_XGBoost_Estimation.py
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
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALIDATION_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "06_validation"
)

RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "08_xgboost"
)

AUDIT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "audits"
    / "08_xgboost"
)

FIGURE_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "chapter_5"
)


INPUT_MASTER_PANEL = (
    VALIDATION_DIRECTORY
    / "JSE_Master_Model_Panel.csv"
)

INPUT_OUTER_WINDOWS = (
    VALIDATION_DIRECTORY
    / "JSE_Walk_Forward_Windows.csv"
)

INPUT_INNER_WINDOWS = (
    VALIDATION_DIRECTORY
    / "JSE_Inner_Validation_Windows.csv"
)


OUTPUT_PREDICTIONS = (
    RESULTS_DIRECTORY
    / "JSE_XGB_Walk_Forward_Predictions.csv"
)

OUTPUT_DAILY_PERFORMANCE = (
    RESULTS_DIRECTORY
    / "JSE_XGB_Daily_Performance.csv"
)

OUTPUT_WINDOW_PERFORMANCE = (
    RESULTS_DIRECTORY
    / "JSE_XGB_Window_Performance.csv"
)

OUTPUT_OVERALL_PERFORMANCE = (
    RESULTS_DIRECTORY
    / "JSE_XGB_Overall_Performance.csv"
)


OUTPUT_SEARCH_RESULTS = (
    AUDIT_DIRECTORY
    / "JSE_XGB_Hyperparameter_Search_Results.csv"
)

OUTPUT_SELECTED_PARAMETERS = (
    AUDIT_DIRECTORY
    / "JSE_XGB_Selected_Hyperparameters.csv"
)

OUTPUT_WINSORISATION = (
    AUDIT_DIRECTORY
    / "JSE_XGB_Winsorisation_Thresholds.csv"
)

OUTPUT_FEATURE_IMPORTANCE = (
    AUDIT_DIRECTORY
    / "JSE_XGB_Feature_Importance.csv"
)

OUTPUT_ESTIMATION_SUMMARY = (
    AUDIT_DIRECTORY
    / "JSE_XGB_Estimation_Summary.csv"
)

OUTPUT_FIGURE = (
    FIGURE_DIRECTORY
    / "figure_5_3_xgb_feature_importance_stability.png"
)


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

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


# 2 × 2 × 2 × 2 × 2 = 32 candidates.
LEARNING_RATES = [
    0.02,
    0.05,
]

MAX_DEPTHS = [
    2,
    3,
]

MIN_CHILD_WEIGHTS = [
    5.0,
    15.0,
]

SUBSAMPLES = [
    0.8,
    1.0,
]

COLSAMPLE_BYTREE_VALUES = [
    0.8,
    1.0,
]


# Fixed regularisation parameters.
REG_ALPHA = 0.10
REG_LAMBDA = 10.0
GAMMA = 0.0


# ============================================================
# 3. GENERAL HELPERS
# ============================================================

def relative_path(
    path: Path,
) -> str:
    """Return a readable path relative to the project root."""

    return str(
        path.relative_to(
            PROJECT_ROOT
        )
    )


def parse_window_dates(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> None:
    """Parse and validate stored window-date columns."""

    for column in columns:
        if column not in dataframe.columns:
            raise ValueError(
                f"Required date column missing: {column}"
            )

        dataframe[column] = pd.to_datetime(
            dataframe[column],
            errors="raise",
        )


# ============================================================
# 4. INPUT VALIDATION
# ============================================================

def validate_required_files() -> None:
    """Confirm that all Script 06 outputs exist."""

    required_files = [
        INPUT_MASTER_PANEL,
        INPUT_OUTER_WINDOWS,
        INPUT_INNER_WINDOWS,
    ]

    missing_files = [
        str(
            path.resolve()
        )
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required input files are missing:\n- "
            + "\n- ".join(
                missing_files
            )
        )


def validate_master_panel(
    panel: pd.DataFrame,
) -> None:
    """Validate the common model panel."""

    required_columns = {
        "Date",
        "Ticker",
        TARGET_COLUMN,
        "Forward_Return",
        "N_t",
        "Model_N_t",
        *FEATURES,
    }

    missing_columns = required_columns.difference(
        panel.columns
    )

    if missing_columns:
        raise ValueError(
            "Master model panel is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if panel.empty:
        raise ValueError(
            "Master model panel contains no observations."
        )

    duplicate_count = int(
        panel.duplicated(
            [
                "Date",
                "Ticker",
            ]
        ).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"Master model panel contains {duplicate_count:,} "
            "duplicate Date-Ticker observations."
        )

    required_nonmissing = [
        "Date",
        "Ticker",
        TARGET_COLUMN,
        "Forward_Return",
        "N_t",
        "Model_N_t",
        *FEATURES,
    ]

    missing_values = int(
        panel[
            required_nonmissing
        ]
        .isna()
        .sum()
        .sum()
    )

    if missing_values > 0:
        raise ValueError(
            f"Master panel contains {missing_values:,} "
            "missing required values."
        )

    numeric_values = panel[
        [
            TARGET_COLUMN,
            "Forward_Return",
            *FEATURES,
        ]
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        numeric_values
    ).all():
        raise ValueError(
            "Non-finite values exist in the model matrix."
        )

    if not panel[
        TARGET_COLUMN
    ].between(
        0,
        1,
        inclusive="both",
    ).all():
        raise ValueError(
            "Target values outside [0, 1] were detected."
        )

    observed_model_size = (
        panel
        .groupby(
            "Date"
        )[
            "Ticker"
        ]
        .transform(
            "nunique"
        )
        .astype(
            int
        )
    )

    if not panel[
        "Model_N_t"
    ].astype(
        int
    ).eq(
        observed_model_size
    ).all():
        raise ValueError(
            "Stored Model_N_t does not match the observed "
            "daily cross-sectional size."
        )


def validate_outer_windows(
    windows: pd.DataFrame,
) -> None:
    """Validate the stored outer schedule."""

    required_columns = {
        "Outer_Window_ID",
        "Test_Quarter",
        "Train_Start",
        "Train_End",
        "Purged_Boundary_Date",
        "Test_Start",
        "Test_End",
        "Train_Dates",
        "Train_Observations",
        "Test_Dates",
        "Test_Observations",
    }

    missing_columns = required_columns.difference(
        windows.columns
    )

    if missing_columns:
        raise ValueError(
            "Outer-window file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if windows.empty:
        raise ValueError(
            "No outer windows were supplied."
        )

    if windows[
        "Outer_Window_ID"
    ].duplicated().any():
        raise ValueError(
            "Duplicate outer-window identifiers were detected."
        )

    if not (
        windows[
            "Train_End"
        ]
        < windows[
            "Purged_Boundary_Date"
        ]
    ).all():
        raise ValueError(
            "At least one outer training block reaches "
            "its purge date."
        )

    if not (
        windows[
            "Purged_Boundary_Date"
        ]
        < windows[
            "Test_Start"
        ]
    ).all():
        raise ValueError(
            "At least one outer purge date is not earlier "
            "than its test block."
        )


def validate_inner_windows(
    inner_windows: pd.DataFrame,
    outer_windows: pd.DataFrame,
) -> None:
    """Validate the nested chronological validation schedule."""

    required_columns = {
        "Outer_Window_ID",
        "Test_Quarter",
        "Inner_Fold_ID",
        "Inner_Fold_Number",
        "Inner_Train_Start",
        "Inner_Train_End",
        "Inner_Purged_Boundary_Date",
        "Validation_Start",
        "Validation_End",
        "Inner_Train_Dates",
        "Inner_Train_Observations",
        "Validation_Dates",
        "Validation_Observations",
    }

    missing_columns = required_columns.difference(
        inner_windows.columns
    )

    if missing_columns:
        raise ValueError(
            "Inner-window file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if inner_windows.empty:
        raise ValueError(
            "No inner validation windows were supplied."
        )

    if inner_windows[
        "Inner_Fold_ID"
    ].duplicated().any():
        raise ValueError(
            "Duplicate inner-fold identifiers were detected."
        )

    if not (
        inner_windows[
            "Inner_Train_End"
        ]
        < inner_windows[
            "Inner_Purged_Boundary_Date"
        ]
    ).all():
        raise ValueError(
            "At least one inner training block reaches "
            "its purge date."
        )

    if not (
        inner_windows[
            "Inner_Purged_Boundary_Date"
        ]
        < inner_windows[
            "Validation_Start"
        ]
    ).all():
        raise ValueError(
            "At least one inner purge date is not earlier "
            "than validation."
        )

    expected_outer_ids = set(
        outer_windows[
            "Outer_Window_ID"
        ]
    )

    observed_outer_ids = set(
        inner_windows[
            "Outer_Window_ID"
        ]
    )

    if expected_outer_ids != observed_outer_ids:
        raise ValueError(
            "The inner-fold outer-window identifiers do not "
            "match the outer schedule."
        )

    folds_per_outer = (
        inner_windows
        .groupby(
            "Outer_Window_ID"
        )[
            "Inner_Fold_ID"
        ]
        .nunique()
    )

    if not folds_per_outer.eq(
        3
    ).all():
        raise ValueError(
            "Every outer window must contain exactly "
            "three inner validation folds."
        )


# ============================================================
# 5. WINSORISATION
# ============================================================

def estimate_winsorisation_thresholds(
    training_data: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate 1st and 99th percentile bounds from training data."""

    records: list[
        dict[str, object]
    ] = []

    for feature in FEATURES:
        lower_bound = float(
            training_data[
                feature
            ].quantile(
                WINSOR_LOWER_QUANTILE
            )
        )

        upper_bound = float(
            training_data[
                feature
            ].quantile(
                WINSOR_UPPER_QUANTILE
            )
        )

        if not np.isfinite(
            lower_bound
        ):
            raise ValueError(
                f"Non-finite lower winsorisation "
                f"threshold for {feature}."
            )

        if not np.isfinite(
            upper_bound
        ):
            raise ValueError(
                f"Non-finite upper winsorisation "
                f"threshold for {feature}."
            )

        if lower_bound > upper_bound:
            raise ValueError(
                f"Invalid winsorisation interval for {feature}."
            )

        records.append(
            {
                "Feature": feature,
                "Lower_Quantile": (
                    WINSOR_LOWER_QUANTILE
                ),
                "Upper_Quantile": (
                    WINSOR_UPPER_QUANTILE
                ),
                "Lower_Bound": lower_bound,
                "Upper_Bound": upper_bound,
            }
        )

    return pd.DataFrame(
        records
    )


def apply_winsorisation(
    data: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    """Apply previously estimated clipping thresholds."""

    transformed = data.copy()

    threshold_map = thresholds.set_index(
        "Feature"
    )

    for feature in FEATURES:
        transformed[
            feature
        ] = transformed[
            feature
        ].clip(
            lower=float(
                threshold_map.loc[
                    feature,
                    "Lower_Bound",
                ]
            ),
            upper=float(
                threshold_map.loc[
                    feature,
                    "Upper_Bound",
                ]
            ),
        )

    return transformed


# ============================================================
# 6. XGBOOST CONFIGURATION AND FITTING
# ============================================================

def candidate_configurations() -> list[
    dict[str, object]
]:
    """Construct the prespecified 32-candidate grid."""

    candidates: list[
        dict[str, object]
    ] = []

    grid = product(
        LEARNING_RATES,
        MAX_DEPTHS,
        MIN_CHILD_WEIGHTS,
        SUBSAMPLES,
        COLSAMPLE_BYTREE_VALUES,
    )

    for candidate_number, values in enumerate(
        grid,
        start=1,
    ):
        (
            learning_rate,
            max_depth,
            min_child_weight,
            subsample,
            colsample_bytree,
        ) = values

        candidates.append(
            {
                "Candidate_ID": (
                    f"C_{candidate_number:02d}"
                ),
                "learning_rate": float(
                    learning_rate
                ),
                "max_depth": int(
                    max_depth
                ),
                "min_child_weight": float(
                    min_child_weight
                ),
                "subsample": float(
                    subsample
                ),
                "colsample_bytree": float(
                    colsample_bytree
                ),
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
    """Construct an XGBoost regressor from a candidate configuration."""

    parameters: dict[
        str,
        object,
    ] = {
        "objective": OBJECTIVE,
        "eval_metric": EVAL_METRIC,
        "tree_method": TREE_METHOD,
        "n_estimators": int(
            n_estimators
        ),
        "learning_rate": float(
            configuration[
                "learning_rate"
            ]
        ),
        "max_depth": int(
            configuration[
                "max_depth"
            ]
        ),
        "min_child_weight": float(
            configuration[
                "min_child_weight"
            ]
        ),
        "subsample": float(
            configuration[
                "subsample"
            ]
        ),
        "colsample_bytree": float(
            configuration[
                "colsample_bytree"
            ]
        ),
        "reg_alpha": float(
            configuration[
                "reg_alpha"
            ]
        ),
        "reg_lambda": float(
            configuration[
                "reg_lambda"
            ]
        ),
        "gamma": float(
            configuration[
                "gamma"
            ]
        ),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }

    if use_early_stopping:
        parameters[
            "early_stopping_rounds"
        ] = EARLY_STOPPING_ROUNDS

    return xgb.XGBRegressor(
        **parameters
    )


def fit_with_early_stopping(
    configuration: dict[str, object],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> xgb.XGBRegressor:
    """
    Fit with early stopping while supporting both newer and older
    XGBoost sklearn APIs.
    """

    model = build_xgb_regressor(
        configuration=configuration,
        n_estimators=MAX_BOOSTING_ROUNDS,
        use_early_stopping=True,
    )

    try:
        model.fit(
            x_train,
            y_train,
            eval_set=[
                (
                    x_validation,
                    y_validation,
                )
            ],
            verbose=False,
        )

        return model

    except TypeError:
        model = build_xgb_regressor(
            configuration=configuration,
            n_estimators=MAX_BOOSTING_ROUNDS,
            use_early_stopping=False,
        )

        model.fit(
            x_train,
            y_train,
            eval_set=[
                (
                    x_validation,
                    y_validation,
                )
            ],
            early_stopping_rounds=(
                EARLY_STOPPING_ROUNDS
            ),
            verbose=False,
        )

        return model


def selected_boosting_rounds(
    model: xgb.XGBRegressor,
) -> int:
    """Return the number of boosting rounds selected by early stopping."""

    best_iteration = getattr(
        model,
        "best_iteration",
        None,
    )

    if best_iteration is None:
        booster = model.get_booster()

        best_iteration = getattr(
            booster,
            "best_iteration",
            None,
        )

    if best_iteration is None:
        return MAX_BOOSTING_ROUNDS

    return max(
        1,
        int(
            best_iteration
        )
        + 1,
    )


# ============================================================
# 7. SPEARMAN IC AND PERFORMANCE HELPERS
# ============================================================

def safe_daily_spearman(
    group: pd.DataFrame,
    prediction_column: str,
) -> tuple[float, bool]:
    """
    Calculate daily Spearman IC.

    Constant prediction vectors contain no ranking information and are
    assigned IC = 0.
    """

    predictions = group[
        prediction_column
    ]

    realised_target = group[
        TARGET_COLUMN
    ]

    if len(
        group
    ) < MINIMUM_DAILY_CROSS_SECTION:
        return (
            np.nan,
            False,
        )

    if (
        predictions.nunique(
            dropna=True
        )
        <= 1
    ):
        return (
            0.0,
            True,
        )

    if (
        realised_target.nunique(
            dropna=True
        )
        <= 1
    ):
        return (
            0.0,
            True,
        )

    result = spearmanr(
        predictions,
        realised_target,
    )

    statistic = float(
        result.statistic
    )

    if not np.isfinite(
        statistic
    ):
        return (
            0.0,
            True,
        )

    return (
        statistic,
        False,
    )


def validation_metrics(
    validation_data: pd.DataFrame,
    predictions: np.ndarray,
) -> dict[str, float | int]:
    """Calculate inner-fold validation metrics."""

    scored = validation_data[
        [
            "Date",
            "Ticker",
            TARGET_COLUMN,
        ]
    ].copy()

    scored[
        "Validation_Prediction"
    ] = predictions

    daily_ic_values: list[
        float
    ] = []

    constant_prediction_days = 0

    for _, group in scored.groupby(
        "Date",
        sort=True,
    ):
        daily_ic, constant_flag = (
            safe_daily_spearman(
                group=group,
                prediction_column=(
                    "Validation_Prediction"
                ),
            )
        )

        daily_ic_values.append(
            daily_ic
        )

        constant_prediction_days += int(
            constant_flag
        )

    daily_ic_array = np.asarray(
        daily_ic_values,
        dtype=float,
    )

    if np.isnan(
        daily_ic_array
    ).any():
        raise ValueError(
            "Missing validation IC values were produced."
        )

    errors = (
        predictions
        - validation_data[
            TARGET_COLUMN
        ].to_numpy(
            dtype=float
        )
    )

    return {
        "Validation_Mean_Daily_IC": float(
            np.mean(
                daily_ic_array
            )
        ),
        "Validation_Median_Daily_IC": float(
            np.median(
                daily_ic_array
            )
        ),
        "Validation_IC_Standard_Deviation": float(
            np.std(
                daily_ic_array,
                ddof=1,
            )
        ),
        "Validation_RMSE": float(
            np.sqrt(
                np.mean(
                    errors ** 2
                )
            )
        ),
        "Validation_MAE": float(
            np.mean(
                np.abs(
                    errors
                )
            )
        ),
        "Validation_Constant_Prediction_Days": int(
            constant_prediction_days
        ),
    }


def deterministic_top_bottom_hit_rates(
    group: pd.DataFrame,
) -> tuple[float, float, int]:
    """Calculate deterministic top- and bottom-quartile overlap."""

    number_of_assets = len(
        group
    )

    quartile_size = max(
        1,
        int(
            np.ceil(
                number_of_assets
                * TOP_BOTTOM_FRACTION
            )
        ),
    )

    predicted_order = group.sort_values(
        [
            "XGB_Raw_Score",
            "Ticker",
        ],
        ascending=[
            False,
            True,
        ],
        kind="mergesort",
    )

    realised_order = group.sort_values(
        [
            TARGET_COLUMN,
            "Ticker",
        ],
        ascending=[
            False,
            True,
        ],
        kind="mergesort",
    )

    predicted_top = set(
        predicted_order.head(
            quartile_size
        )[
            "Ticker"
        ]
    )

    realised_top = set(
        realised_order.head(
            quartile_size
        )[
            "Ticker"
        ]
    )

    predicted_bottom = set(
        predicted_order.tail(
            quartile_size
        )[
            "Ticker"
        ]
    )

    realised_bottom = set(
        realised_order.tail(
            quartile_size
        )[
            "Ticker"
        ]
    )

    top_hit_rate = (
        len(
            predicted_top.intersection(
                realised_top
            )
        )
        / quartile_size
    )

    bottom_hit_rate = (
        len(
            predicted_bottom.intersection(
                realised_bottom
            )
        )
        / quartile_size
    )

    return (
        float(
            top_hit_rate
        ),
        float(
            bottom_hit_rate
        ),
        quartile_size,
    )


def calculate_daily_performance(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate daily XGBoost predictive diagnostics."""

    records: list[
        dict[str, object]
    ] = []

    for date, group in predictions.groupby(
        "Date",
        sort=True,
    ):
        daily_ic, constant_flag = (
            safe_daily_spearman(
                group=group,
                prediction_column="XGB_Raw_Score",
            )
        )

        errors = (
            group[
                "XGB_Raw_Score"
            ]
            - group[
                TARGET_COLUMN
            ]
        )

        (
            top_hit_rate,
            bottom_hit_rate,
            quartile_size,
        ) = deterministic_top_bottom_hit_rates(
            group
        )

        records.append(
            {
                "Date": date,
                "Outer_Window_ID": group[
                    "Outer_Window_ID"
                ].iloc[
                    0
                ],
                "Test_Quarter": group[
                    "Test_Quarter"
                ].iloc[
                    0
                ],
                "Model_Eligible_Securities": len(
                    group
                ),
                "Daily_IC": daily_ic,
                "Daily_MAE": float(
                    np.mean(
                        np.abs(
                            errors
                        )
                    )
                ),
                "Daily_RMSE": float(
                    np.sqrt(
                        np.mean(
                            errors ** 2
                        )
                    )
                ),
                "Quartile_Size": quartile_size,
                "Top_Quartile_Hit_Rate": (
                    top_hit_rate
                ),
                "Bottom_Quartile_Hit_Rate": (
                    bottom_hit_rate
                ),
                "Predictions_Below_Zero": int(
                    group[
                        "XGB_Raw_Score"
                    ].lt(
                        0
                    ).sum()
                ),
                "Predictions_Above_One": int(
                    group[
                        "XGB_Raw_Score"
                    ].gt(
                        1
                    ).sum()
                ),
                "Constant_Prediction_Day": int(
                    constant_flag
                ),
            }
        )

    return pd.DataFrame(
        records
    )


def summarise_single_group(
    predictions: pd.DataFrame,
    daily_performance: pd.DataFrame,
) -> dict[str, float | int]:
    """Summarise one prediction group."""

    daily_ic = daily_performance[
        "Daily_IC"
    ].dropna()

    if daily_ic.empty:
        mean_ic = np.nan
        median_ic = np.nan
        ic_standard_deviation = np.nan
        ic_information_ratio = np.nan

    else:
        mean_ic = float(
            daily_ic.mean()
        )

        median_ic = float(
            daily_ic.median()
        )

        ic_standard_deviation = float(
            daily_ic.std(
                ddof=1
            )
        )

        ic_information_ratio = (
            mean_ic
            / ic_standard_deviation
            if (
                np.isfinite(
                    ic_standard_deviation
                )
                and ic_standard_deviation > 0
            )
            else np.nan
        )

    errors = (
        predictions[
            "XGB_Raw_Score"
        ]
        - predictions[
            TARGET_COLUMN
        ]
    )

    return {
        "Out_Of_Sample_Observations": int(
            len(
                predictions
            )
        ),
        "Out_Of_Sample_Dates": int(
            predictions[
                "Date"
            ].nunique()
        ),
        "Mean_Daily_IC": mean_ic,
        "Median_Daily_IC": median_ic,
        "Daily_IC_Standard_Deviation": (
            ic_standard_deviation
        ),
        "Daily_IC_Information_Ratio": (
            ic_information_ratio
        ),
        "Positive_IC_Days": int(
            daily_ic.gt(
                0
            ).sum()
        ),
        "Zero_IC_Days": int(
            daily_ic.eq(
                0
            ).sum()
        ),
        "Negative_IC_Days": int(
            daily_ic.lt(
                0
            ).sum()
        ),
        "Positive_IC_Percentage": float(
            daily_ic.gt(
                0
            ).mean()
            * 100.0
        ),
        "Observation_MAE": float(
            np.mean(
                np.abs(
                    errors
                )
            )
        ),
        "Observation_RMSE": float(
            np.sqrt(
                np.mean(
                    errors ** 2
                )
            )
        ),
        "Mean_Daily_MAE": float(
            daily_performance[
                "Daily_MAE"
            ].mean()
        ),
        "Mean_Daily_RMSE": float(
            daily_performance[
                "Daily_RMSE"
            ].mean()
        ),
        "Mean_Top_Quartile_Hit_Rate": float(
            daily_performance[
                "Top_Quartile_Hit_Rate"
            ].mean()
        ),
        "Mean_Bottom_Quartile_Hit_Rate": float(
            daily_performance[
                "Bottom_Quartile_Hit_Rate"
            ].mean()
        ),
        "Predictions_Below_Zero": int(
            predictions[
                "XGB_Raw_Score"
            ].lt(
                0
            ).sum()
        ),
        "Predictions_Above_One": int(
            predictions[
                "XGB_Raw_Score"
            ].gt(
                1
            ).sum()
        ),
        "Constant_Prediction_Days": int(
            daily_performance[
                "Constant_Prediction_Day"
            ].sum()
        ),
    }


def summarise_by_window(
    predictions: pd.DataFrame,
    daily_performance: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise performance separately for every outer window."""

    records: list[
        dict[str, object]
    ] = []

    grouped_predictions = predictions.groupby(
        [
            "Outer_Window_ID",
            "Test_Quarter",
        ],
        sort=False,
    )

    for (
        window_id,
        test_quarter,
    ), group in grouped_predictions:
        daily_group = daily_performance.loc[
            daily_performance[
                "Outer_Window_ID"
            ].eq(
                window_id
            )
        ].copy()

        record = summarise_single_group(
            predictions=group,
            daily_performance=daily_group,
        )

        record[
            "Outer_Window_ID"
        ] = window_id

        record[
            "Test_Quarter"
        ] = test_quarter

        records.append(
            record
        )

    result = pd.DataFrame(
        records
    )

    leading_columns = [
        "Outer_Window_ID",
        "Test_Quarter",
    ]

    return result[
        leading_columns
        + [
            column
            for column in result.columns
            if column
            not in leading_columns
        ]
    ]


# ============================================================
# 8. NESTED HYPERPARAMETER TUNING
# ============================================================

def tune_outer_window(
    master_panel: pd.DataFrame,
    inner_windows: pd.DataFrame,
    outer_window_id: str,
    test_quarter: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate every candidate on the three stored inner folds.

    Candidate selection is based primarily on the highest mean daily
    validation Spearman IC.
    """

    candidates = candidate_configurations()

    fold_rows = (
        inner_windows.loc[
            inner_windows[
                "Outer_Window_ID"
            ].eq(
                outer_window_id
            )
        ]
        .sort_values(
            "Inner_Fold_Number"
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        fold_rows
    ) != 3:
        raise ValueError(
            f"{outer_window_id} does not contain exactly "
            "three inner folds."
        )

    detailed_records: list[
        dict[str, object]
    ] = []

    for candidate in candidates:
        for fold in fold_rows.itertuples(
            index=False
        ):
            inner_train_mask = master_panel[
                "Date"
            ].between(
                pd.Timestamp(
                    fold.Inner_Train_Start
                ),
                pd.Timestamp(
                    fold.Inner_Train_End
                ),
                inclusive="both",
            )

            validation_mask = master_panel[
                "Date"
            ].between(
                pd.Timestamp(
                    fold.Validation_Start
                ),
                pd.Timestamp(
                    fold.Validation_End
                ),
                inclusive="both",
            )

            inner_train = master_panel.loc[
                inner_train_mask
            ].copy()

            validation = master_panel.loc[
                validation_mask
            ].copy()

            if len(
                inner_train
            ) != int(
                fold.Inner_Train_Observations
            ):
                raise AssertionError(
                    f"Inner training-row mismatch for "
                    f"{fold.Inner_Fold_ID}."
                )

            if len(
                validation
            ) != int(
                fold.Validation_Observations
            ):
                raise AssertionError(
                    f"Validation-row mismatch for "
                    f"{fold.Inner_Fold_ID}."
                )

            if inner_train[
                "Date"
            ].nunique() != int(
                fold.Inner_Train_Dates
            ):
                raise AssertionError(
                    f"Inner training-date mismatch for "
                    f"{fold.Inner_Fold_ID}."
                )

            if validation[
                "Date"
            ].nunique() != int(
                fold.Validation_Dates
            ):
                raise AssertionError(
                    f"Validation-date mismatch for "
                    f"{fold.Inner_Fold_ID}."
                )

            if inner_train[
                "Date"
            ].max() >= validation[
                "Date"
            ].min():
                raise AssertionError(
                    f"Chronological overlap detected in "
                    f"{fold.Inner_Fold_ID}."
                )

            if "Next_Date" in inner_train.columns:
                leakage_count = int(
                    (
                        inner_train[
                            "Next_Date"
                        ]
                        >= pd.Timestamp(
                            fold.Validation_Start
                        )
                    ).sum()
                )

                if leakage_count > 0:
                    raise AssertionError(
                        f"{leakage_count:,} target-boundary "
                        f"leakage rows detected in "
                        f"{fold.Inner_Fold_ID}."
                    )

            thresholds = (
                estimate_winsorisation_thresholds(
                    inner_train
                )
            )

            inner_train_transformed = (
                apply_winsorisation(
                    data=inner_train,
                    thresholds=thresholds,
                )
            )

            validation_transformed = (
                apply_winsorisation(
                    data=validation,
                    thresholds=thresholds,
                )
            )

            model = fit_with_early_stopping(
                configuration=candidate,
                x_train=inner_train_transformed[
                    FEATURES
                ],
                y_train=inner_train_transformed[
                    TARGET_COLUMN
                ],
                x_validation=validation_transformed[
                    FEATURES
                ],
                y_validation=validation_transformed[
                    TARGET_COLUMN
                ],
            )

            validation_predictions = model.predict(
                validation_transformed[
                    FEATURES
                ]
            )

            if not np.isfinite(
                validation_predictions
            ).all():
                raise AssertionError(
                    f"Non-finite validation predictions "
                    f"were produced in {fold.Inner_Fold_ID}."
                )

            metrics = validation_metrics(
                validation_data=validation_transformed,
                predictions=validation_predictions,
            )

            rounds = selected_boosting_rounds(
                model
            )

            detailed_records.append(
                {
                    "Outer_Window_ID": outer_window_id,
                    "Test_Quarter": test_quarter,
                    "Candidate_ID": candidate[
                        "Candidate_ID"
                    ],
                    "Inner_Fold_ID": fold.Inner_Fold_ID,
                    "Inner_Fold_Number": int(
                        fold.Inner_Fold_Number
                    ),
                    "Inner_Train_Dates": int(
                        fold.Inner_Train_Dates
                    ),
                    "Inner_Train_Observations": int(
                        fold.Inner_Train_Observations
                    ),
                    "Validation_Dates": int(
                        fold.Validation_Dates
                    ),
                    "Validation_Observations": int(
                        fold.Validation_Observations
                    ),
                    **{
                        key: value
                        for key, value in candidate.items()
                        if key != "Candidate_ID"
                    },
                    "Selected_Boosting_Rounds": rounds,
                    **metrics,
                }
            )

    detailed = pd.DataFrame(
        detailed_records
    )

    expected_rows = (
        len(
            candidates
        )
        * len(
            fold_rows
        )
    )

    if len(
        detailed
    ) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows:,} inner fits for "
            f"{outer_window_id}, but obtained {len(detailed):,}."
        )

    aggregate = (
        detailed
        .groupby(
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
            Mean_Validation_Daily_IC=(
                "Validation_Mean_Daily_IC",
                "mean",
            ),
            Median_Validation_Daily_IC=(
                "Validation_Median_Daily_IC",
                "mean",
            ),
            Mean_Validation_IC_Standard_Deviation=(
                "Validation_IC_Standard_Deviation",
                "mean",
            ),
            Mean_Validation_RMSE=(
                "Validation_RMSE",
                "mean",
            ),
            Mean_Validation_MAE=(
                "Validation_MAE",
                "mean",
            ),
            Total_Constant_Prediction_Days=(
                "Validation_Constant_Prediction_Days",
                "sum",
            ),
            Median_Selected_Boosting_Rounds=(
                "Selected_Boosting_Rounds",
                "median",
            ),
            Minimum_Selected_Boosting_Rounds=(
                "Selected_Boosting_Rounds",
                "min",
            ),
            Maximum_Selected_Boosting_Rounds=(
                "Selected_Boosting_Rounds",
                "max",
            ),
        )
    )

    # Primary criterion: highest mean daily validation IC.
    # Tie-breaks: lower RMSE, shallower trees, fewer rounds,
    # then deterministic Candidate_ID ordering.
    aggregate = (
        aggregate.sort_values(
            [
                "Mean_Validation_Daily_IC",
                "Mean_Validation_RMSE",
                "max_depth",
                "Median_Selected_Boosting_Rounds",
                "Candidate_ID",
            ],
            ascending=[
                False,
                True,
                True,
                True,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    aggregate[
        "Candidate_Rank"
    ] = np.arange(
        1,
        len(
            aggregate
        )
        + 1,
    )

    aggregate[
        "Selected_Candidate"
    ] = aggregate[
        "Candidate_Rank"
    ].eq(
        1
    )

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

    return (
        detailed,
        aggregate,
    )


# ============================================================
# 9. FEATURE-IMPORTANCE FIGURE
# ============================================================

def create_feature_importance_figure(
    feature_importance: pd.DataFrame,
) -> None:
    """Create a gain-importance stability heatmap."""

    pivot = (
        feature_importance.pivot(
            index="Feature",
            columns="Test_Quarter",
            values="Gain_Importance",
        )
        .reindex(
            FEATURES
        )
    )

    if pivot.isna().any().any():
        raise ValueError(
            "Missing values exist in the feature-importance matrix."
        )

    figure_width = max(
        11,
        len(
            pivot.columns
        )
        * 1.05,
    )

    figure_height = max(
        6.5,
        len(
            pivot.index
        )
        * 0.65,
    )

    figure, axis = plt.subplots(
        figsize=(
            figure_width,
            figure_height,
        )
    )

    image = axis.imshow(
        pivot.to_numpy(
            dtype=float
        ),
        aspect="auto",
        cmap="viridis",
    )

    axis.set_xticks(
        np.arange(
            len(
                pivot.columns
            )
        )
    )

    axis.set_xticklabels(
        pivot.columns,
        rotation=45,
        ha="right",
    )

    axis.set_yticks(
        np.arange(
            len(
                pivot.index
            )
        )
    )

    axis.set_yticklabels(
        pivot.index
    )

    axis.set_title(
        "XGBoost Gain-Based Feature Importance "
        "across Expanding Training Windows"
    )

    axis.set_xlabel(
        "Out-of-sample test quarter"
    )

    axis.set_ylabel(
        "Predictor"
    )

    matrix_mean = float(
        pivot.to_numpy(
            dtype=float
        ).mean()
    )

    for row_index in range(
        len(
            pivot.index
        )
    ):
        for column_index in range(
            len(
                pivot.columns
            )
        ):
            value = float(
                pivot.iloc[
                    row_index,
                    column_index,
                ]
            )

            axis.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    "white"
                    if value > matrix_mean
                    else "black"
                ),
            )

    colour_bar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )

    colour_bar.set_label(
        "Normalised gain importance"
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# 10. MAIN PIPELINE
# ============================================================

def main() -> None:
    print("=" * 78)
    print("SCRIPT 08: XGBOOST NESTED EXPANDING-WINDOW ESTIMATION")
    print("=" * 78)

    overall_start_time = time.perf_counter()

    validate_required_files()

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 10.1 Load inputs
    # --------------------------------------------------------

    master_panel = pd.read_csv(
        INPUT_MASTER_PANEL
    )

    outer_windows = pd.read_csv(
        INPUT_OUTER_WINDOWS
    )

    inner_windows = pd.read_csv(
        INPUT_INNER_WINDOWS
    )

    master_panel[
        "Date"
    ] = pd.to_datetime(
        master_panel[
            "Date"
        ],
        errors="raise",
    )

    if "Next_Date" in master_panel.columns:
        master_panel[
            "Next_Date"
        ] = pd.to_datetime(
            master_panel[
                "Next_Date"
            ],
            errors="coerce",
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

    master_panel = (
        master_panel
        .sort_values(
            [
                "Date",
                "Ticker",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    outer_windows = (
        outer_windows
        .sort_values(
            "Test_Start"
        )
        .reset_index(
            drop=True
        )
    )

    inner_windows = (
        inner_windows
        .sort_values(
            [
                "Outer_Window_ID",
                "Inner_Fold_Number",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    validate_master_panel(
        master_panel
    )

    validate_outer_windows(
        outer_windows
    )

    validate_inner_windows(
        inner_windows=inner_windows,
        outer_windows=outer_windows,
    )

    expected_test_rows = int(
        outer_windows[
            "Test_Observations"
        ].sum()
    )

    expected_test_dates = int(
        outer_windows[
            "Test_Dates"
        ].sum()
    )

    prediction_frames: list[
        pd.DataFrame
    ] = []

    search_detail_frames: list[
        pd.DataFrame
    ] = []

    selected_parameter_records: list[
        dict[str, object]
    ] = []

    winsorisation_frames: list[
        pd.DataFrame
    ] = []

    feature_importance_records: list[
        dict[str, object]
    ] = []

    estimation_summary_records: list[
        dict[str, object]
    ] = []

    # --------------------------------------------------------
    # 10.2 Estimate each outer XGBoost model
    # --------------------------------------------------------

    for outer_row in outer_windows.itertuples(
        index=False
    ):
        window_start_time = time.perf_counter()

        outer_train_mask = master_panel[
            "Date"
        ].between(
            pd.Timestamp(
                outer_row.Train_Start
            ),
            pd.Timestamp(
                outer_row.Train_End
            ),
            inclusive="both",
        )

        outer_test_mask = master_panel[
            "Date"
        ].between(
            pd.Timestamp(
                outer_row.Test_Start
            ),
            pd.Timestamp(
                outer_row.Test_End
            ),
            inclusive="both",
        )

        outer_train = master_panel.loc[
            outer_train_mask
        ].copy()

        outer_test = master_panel.loc[
            outer_test_mask
        ].copy()

        if len(
            outer_train
        ) != int(
            outer_row.Train_Observations
        ):
            raise AssertionError(
                f"Training-row count mismatch for "
                f"{outer_row.Outer_Window_ID}."
            )

        if len(
            outer_test
        ) != int(
            outer_row.Test_Observations
        ):
            raise AssertionError(
                f"Test-row count mismatch for "
                f"{outer_row.Outer_Window_ID}."
            )

        if outer_train[
            "Date"
        ].nunique() != int(
            outer_row.Train_Dates
        ):
            raise AssertionError(
                f"Training-date count mismatch for "
                f"{outer_row.Outer_Window_ID}."
            )

        if outer_test[
            "Date"
        ].nunique() != int(
            outer_row.Test_Dates
        ):
            raise AssertionError(
                f"Test-date count mismatch for "
                f"{outer_row.Outer_Window_ID}."
            )

        if outer_train[
            "Date"
        ].max() >= outer_test[
            "Date"
        ].min():
            raise AssertionError(
                f"Training and test dates overlap in "
                f"{outer_row.Outer_Window_ID}."
            )

        if "Next_Date" in outer_train.columns:
            outer_leakage_count = int(
                (
                    outer_train[
                        "Next_Date"
                    ]
                    >= pd.Timestamp(
                        outer_row.Test_Start
                    )
                ).sum()
            )

            if outer_leakage_count > 0:
                raise AssertionError(
                    f"{outer_leakage_count:,} target-boundary "
                    f"leakage rows detected in "
                    f"{outer_row.Outer_Window_ID}."
                )

        # ----------------------------------------------------
        # Nested candidate search
        # ----------------------------------------------------

        (
            detailed_search,
            aggregate_search,
        ) = tune_outer_window(
            master_panel=master_panel,
            inner_windows=inner_windows,
            outer_window_id=(
                outer_row.Outer_Window_ID
            ),
            test_quarter=(
                outer_row.Test_Quarter
            ),
        )

        search_detail_frames.append(
            detailed_search
        )

        selected_rows = aggregate_search.loc[
            aggregate_search[
                "Selected_Candidate"
            ]
        ]

        if len(
            selected_rows
        ) != 1:
            raise AssertionError(
                f"Expected one selected candidate for "
                f"{outer_row.Outer_Window_ID}."
            )

        selected = selected_rows.iloc[
            0
        ]

        configuration = {
            "learning_rate": float(
                selected[
                    "learning_rate"
                ]
            ),
            "max_depth": int(
                selected[
                    "max_depth"
                ]
            ),
            "min_child_weight": float(
                selected[
                    "min_child_weight"
                ]
            ),
            "subsample": float(
                selected[
                    "subsample"
                ]
            ),
            "colsample_bytree": float(
                selected[
                    "colsample_bytree"
                ]
            ),
            "reg_alpha": float(
                selected[
                    "reg_alpha"
                ]
            ),
            "reg_lambda": float(
                selected[
                    "reg_lambda"
                ]
            ),
            "gamma": float(
                selected[
                    "gamma"
                ]
            ),
        }

        final_rounds = max(
            1,
            int(
                round(
                    float(
                        selected[
                            "Median_Selected_Boosting_Rounds"
                        ]
                    )
                )
            ),
        )

        # ----------------------------------------------------
        # Outer training-only winsorisation
        # ----------------------------------------------------

        outer_thresholds = (
            estimate_winsorisation_thresholds(
                outer_train
            )
        )

        threshold_values = outer_thresholds.copy()

        outer_thresholds.insert(
            0,
            "Outer_Window_ID",
            outer_row.Outer_Window_ID,
        )

        outer_thresholds.insert(
            1,
            "Test_Quarter",
            outer_row.Test_Quarter,
        )

        outer_thresholds.insert(
            2,
            "Train_Start",
            pd.Timestamp(
                outer_row.Train_Start
            ),
        )

        outer_thresholds.insert(
            3,
            "Train_End",
            pd.Timestamp(
                outer_row.Train_End
            ),
        )

        winsorisation_frames.append(
            outer_thresholds
        )

        outer_train_transformed = (
            apply_winsorisation(
                data=outer_train,
                thresholds=threshold_values,
            )
        )

        outer_test_transformed = (
            apply_winsorisation(
                data=outer_test,
                thresholds=threshold_values,
            )
        )

        # ----------------------------------------------------
        # Fit final outer model
        # ----------------------------------------------------

        final_model = build_xgb_regressor(
            configuration=configuration,
            n_estimators=final_rounds,
            use_early_stopping=False,
        )

        final_model.fit(
            outer_train_transformed[
                FEATURES
            ],
            outer_train_transformed[
                TARGET_COLUMN
            ],
            verbose=False,
        )

        raw_predictions = final_model.predict(
            outer_test_transformed[
                FEATURES
            ]
        )

        if not np.isfinite(
            raw_predictions
        ).all():
            raise AssertionError(
                f"Non-finite test predictions were produced "
                f"in {outer_row.Outer_Window_ID}."
            )

        # ----------------------------------------------------
        # Store predictions and daily predicted ranks
        # ----------------------------------------------------

        prediction_columns = [
            "Date",
            "Ticker",
            TARGET_COLUMN,
            "Forward_Return",
            "N_t",
            "Model_N_t",
        ]

        optional_columns = [
            "Daily_Return_Rank",
            "Active_Quarter",
            "Selection_Quarter",
            "Market_Quarter_End",
            "Liquidity_Rank",
            "Membership_Status",
            "Model_Date_ID",
            "Calendar_Quarter",
            "Calendar_Year",
        ]

        prediction_columns.extend(
            [
                column
                for column in optional_columns
                if (
                    column
                    in outer_test_transformed.columns
                    and column
                    not in prediction_columns
                )
            ]
        )

        window_predictions = (
            outer_test_transformed[
                prediction_columns
            ].copy()
        )

        window_predictions.insert(
            0,
            "Outer_Window_ID",
            outer_row.Outer_Window_ID,
        )

        window_predictions.insert(
            1,
            "Test_Quarter",
            outer_row.Test_Quarter,
        )

        window_predictions[
            "XGB_Raw_Score"
        ] = raw_predictions

        window_predictions[
            "XGB_Predicted_Rank"
        ] = (
            window_predictions
            .groupby(
                "Date"
            )[
                "XGB_Raw_Score"
            ]
            .rank(
                method="average",
                ascending=True,
            )
        )

        window_predictions[
            "XGB_Prediction_N_t"
        ] = (
            window_predictions
            .groupby(
                "Date"
            )[
                "Ticker"
            ]
            .transform(
                "count"
            )
            .astype(
                int
            )
        )

        prediction_denominator = (
            window_predictions[
                "XGB_Prediction_N_t"
            ]
            - 1
        )

        window_predictions[
            "XGB_Predicted_Percentile"
        ] = np.where(
            prediction_denominator.gt(
                0
            ),
            (
                window_predictions[
                    "XGB_Predicted_Rank"
                ]
                - 1.0
            )
            / prediction_denominator,
            0.5,
        )

        if not window_predictions[
            "XGB_Predicted_Percentile"
        ].between(
            0,
            1,
            inclusive="both",
        ).all():
            raise AssertionError(
                f"Predicted percentiles outside [0, 1] "
                f"were produced in {outer_row.Outer_Window_ID}."
            )

        prediction_frames.append(
            window_predictions
        )

        # ----------------------------------------------------
        # Gain-based feature importance
        # ----------------------------------------------------

        booster = final_model.get_booster()

        gain_scores = booster.get_score(
            importance_type="gain"
        )

        gain_values = np.array(
            [
                float(
                    gain_scores.get(
                        feature,
                        gain_scores.get(
                            f"f{feature_index}",
                            0.0,
                        ),
                    )
                )
                for feature_index, feature in enumerate(
                    FEATURES
                )
            ],
            dtype=float,
        )

        gain_total = float(
            gain_values.sum()
        )

        normalised_gain = (
            gain_values
            / gain_total
            if gain_total > 0
            else np.zeros_like(
                gain_values
            )
        )

        for (
            feature,
            raw_gain,
            normalised_value,
        ) in zip(
            FEATURES,
            gain_values,
            normalised_gain,
        ):
            feature_importance_records.append(
                {
                    "Outer_Window_ID": (
                        outer_row.Outer_Window_ID
                    ),
                    "Test_Quarter": (
                        outer_row.Test_Quarter
                    ),
                    "Feature": feature,
                    "Raw_Gain": float(
                        raw_gain
                    ),
                    "Gain_Importance": float(
                        normalised_value
                    ),
                }
            )

        # ----------------------------------------------------
        # Selected-parameter and runtime audits
        # ----------------------------------------------------

        selected_parameter_records.append(
            {
                "Outer_Window_ID": (
                    outer_row.Outer_Window_ID
                ),
                "Test_Quarter": (
                    outer_row.Test_Quarter
                ),
                "Candidate_ID": selected[
                    "Candidate_ID"
                ],
                **configuration,
                "Final_N_Estimators": final_rounds,
                "Mean_Validation_Daily_IC": float(
                    selected[
                        "Mean_Validation_Daily_IC"
                    ]
                ),
                "Mean_Validation_RMSE": float(
                    selected[
                        "Mean_Validation_RMSE"
                    ]
                ),
                "Minimum_Selected_Boosting_Rounds": int(
                    selected[
                        "Minimum_Selected_Boosting_Rounds"
                    ]
                ),
                "Maximum_Selected_Boosting_Rounds": int(
                    selected[
                        "Maximum_Selected_Boosting_Rounds"
                    ]
                ),
            }
        )

        elapsed_seconds = (
            time.perf_counter()
            - window_start_time
        )

        estimation_summary_records.append(
            {
                "Outer_Window_ID": (
                    outer_row.Outer_Window_ID
                ),
                "Test_Quarter": (
                    outer_row.Test_Quarter
                ),
                "Train_Start": (
                    outer_row.Train_Start
                ),
                "Train_End": (
                    outer_row.Train_End
                ),
                "Purged_Boundary_Date": (
                    outer_row.Purged_Boundary_Date
                ),
                "Test_Start": (
                    outer_row.Test_Start
                ),
                "Test_End": (
                    outer_row.Test_End
                ),
                "Training_Observations": len(
                    outer_train
                ),
                "Training_Dates": outer_train[
                    "Date"
                ].nunique(),
                "Training_Tickers": outer_train[
                    "Ticker"
                ].nunique(),
                "Test_Observations": len(
                    outer_test
                ),
                "Test_Dates": outer_test[
                    "Date"
                ].nunique(),
                "Test_Tickers": outer_test[
                    "Ticker"
                ].nunique(),
                "Candidate_Configurations_Evaluated": len(
                    aggregate_search
                ),
                "Inner_Folds": int(
                    inner_windows[
                        "Outer_Window_ID"
                    ]
                    .eq(
                        outer_row.Outer_Window_ID
                    )
                    .sum()
                ),
                "Selected_Candidate_ID": selected[
                    "Candidate_ID"
                ],
                "Final_N_Estimators": final_rounds,
                "Window_Runtime_Seconds": (
                    elapsed_seconds
                ),
            }
        )

        print(
            f"Completed {outer_row.Outer_Window_ID} | "
            f"{outer_row.Test_Quarter} | "
            f"selected {selected['Candidate_ID']} | "
            f"rounds {final_rounds} | "
            f"validation IC "
            f"{selected['Mean_Validation_Daily_IC']:.5f} | "
            f"runtime {elapsed_seconds / 60:.1f} min"
        )

    # --------------------------------------------------------
    # 10.3 Combine and validate predictions
    # --------------------------------------------------------

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    predictions = (
        predictions
        .sort_values(
            [
                "Date",
                "Ticker",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        predictions
    ) != expected_test_rows:
        raise AssertionError(
            f"Expected {expected_test_rows:,} predictions "
            f"but obtained {len(predictions):,}."
        )

    if predictions[
        "Date"
    ].nunique() != expected_test_dates:
        raise AssertionError(
            f"Expected {expected_test_dates:,} test dates "
            f"but obtained "
            f"{predictions['Date'].nunique():,}."
        )

    if predictions.duplicated(
        [
            "Date",
            "Ticker",
        ]
    ).any():
        raise AssertionError(
            "Duplicate out-of-sample predictions were generated."
        )

    # --------------------------------------------------------
    # 10.4 Performance outputs
    # --------------------------------------------------------

    daily_performance = (
        calculate_daily_performance(
            predictions
        )
    )

    window_performance = summarise_by_window(
        predictions=predictions,
        daily_performance=daily_performance,
    )

    overall_record = summarise_single_group(
        predictions=predictions,
        daily_performance=daily_performance,
    )

    overall_performance = pd.DataFrame(
        [
            {
                "Model": "XGBoost",
                **overall_record,
            }
        ]
    )

    search_results = pd.concat(
        search_detail_frames,
        ignore_index=True,
    )

    selected_parameters = pd.DataFrame(
        selected_parameter_records
    )

    winsorisation_thresholds = pd.concat(
        winsorisation_frames,
        ignore_index=True,
    )

    feature_importance = pd.DataFrame(
        feature_importance_records
    )

    estimation_summary = pd.DataFrame(
        estimation_summary_records
    )

    # --------------------------------------------------------
    # 10.5 Final output-count assertions
    # --------------------------------------------------------

    expected_search_rows = (
        len(
            outer_windows
        )
        * len(
            candidate_configurations()
        )
        * 3
    )

    if len(
        search_results
    ) != expected_search_rows:
        raise AssertionError(
            f"Expected {expected_search_rows:,} search rows "
            f"but obtained {len(search_results):,}."
        )

    if len(
        selected_parameters
    ) != len(
        outer_windows
    ):
        raise AssertionError(
            "Selected-parameter output does not contain "
            "one record per outer window."
        )

    expected_winsorisation_rows = (
        len(
            outer_windows
        )
        * len(
            FEATURES
        )
    )

    if len(
        winsorisation_thresholds
    ) != expected_winsorisation_rows:
        raise AssertionError(
            "Unexpected number of outer winsorisation rows."
        )

    expected_importance_rows = (
        len(
            outer_windows
        )
        * len(
            FEATURES
        )
    )

    if len(
        feature_importance
    ) != expected_importance_rows:
        raise AssertionError(
            "Unexpected number of feature-importance rows."
        )

    selected_candidate_counts = (
        search_results.loc[
            search_results[
                "Selected_Candidate"
            ]
        ]
        .groupby(
            [
                "Outer_Window_ID",
                "Candidate_ID",
            ]
        )[
            "Inner_Fold_ID"
        ]
        .nunique()
    )

    if not selected_candidate_counts.eq(
        3
    ).all():
        raise AssertionError(
            "A selected candidate is not represented in all "
            "three inner folds."
        )

    # --------------------------------------------------------
    # 10.6 Save outputs
    # --------------------------------------------------------

    predictions.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    daily_performance.to_csv(
        OUTPUT_DAILY_PERFORMANCE,
        index=False,
    )

    window_performance.to_csv(
        OUTPUT_WINDOW_PERFORMANCE,
        index=False,
    )

    overall_performance.to_csv(
        OUTPUT_OVERALL_PERFORMANCE,
        index=False,
    )

    search_results.to_csv(
        OUTPUT_SEARCH_RESULTS,
        index=False,
    )

    selected_parameters.to_csv(
        OUTPUT_SELECTED_PARAMETERS,
        index=False,
    )

    winsorisation_thresholds.to_csv(
        OUTPUT_WINSORISATION,
        index=False,
    )

    feature_importance.to_csv(
        OUTPUT_FEATURE_IMPORTANCE,
        index=False,
    )

    total_runtime_seconds = (
        time.perf_counter()
        - overall_start_time
    )

    estimation_summary[
        "Python_Version"
    ] = platform.python_version()

    estimation_summary[
        "Pandas_Version"
    ] = pd.__version__

    estimation_summary[
        "NumPy_Version"
    ] = np.__version__

    estimation_summary[
        "SciPy_Version"
    ] = scipy.__version__

    estimation_summary[
        "Scikit_Learn_Version"
    ] = sklearn.__version__

    estimation_summary[
        "XGBoost_Version"
    ] = xgb.__version__

    estimation_summary[
        "Random_State"
    ] = RANDOM_STATE

    estimation_summary[
        "Overall_Runtime_Seconds"
    ] = total_runtime_seconds

    estimation_summary.to_csv(
        OUTPUT_ESTIMATION_SUMMARY,
        index=False,
    )

    create_feature_importance_figure(
        feature_importance
    )

    # --------------------------------------------------------
    # 10.7 Terminal summary
    # --------------------------------------------------------

    overall = overall_performance.iloc[
        0
    ]

    print(
        "\n"
        + "=" * 78
    )

    print(
        "SCRIPT 08 EMPIRICAL OUTPUTS"
    )

    print(
        "=" * 78
    )

    print(
        "Outer XGBoost models estimated: "
        f"{len(outer_windows):,}"
    )

    print(
        "Candidate configurations per outer window: "
        f"{len(candidate_configurations()):,}"
    )

    print(
        "Inner model fits completed: "
        f"{len(search_results):,}"
    )

    print(
        "Out-of-sample observations predicted: "
        f"{len(predictions):,}"
    )

    print(
        "Out-of-sample trading dates predicted: "
        f"{predictions['Date'].nunique():,}"
    )

    print(
        "Out-of-sample date range: "
        f"{predictions['Date'].min().date()} to "
        f"{predictions['Date'].max().date()}"
    )

    print(
        "Mean daily Spearman IC: "
        f"{overall['Mean_Daily_IC']:.6f}"
    )

    print(
        "Median daily Spearman IC: "
        f"{overall['Median_Daily_IC']:.6f}"
    )

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

    print(
        "Zero-IC trading days: "
        f"{int(overall['Zero_IC_Days']):,}"
    )

    print(
        "Negative-IC trading days: "
        f"{int(overall['Negative_IC_Days']):,}"
    )

    print(
        "Observation-level MAE: "
        f"{overall['Observation_MAE']:.6f}"
    )

    print(
        "Observation-level RMSE: "
        f"{overall['Observation_RMSE']:.6f}"
    )

    print(
        "Mean daily top-quartile hit rate: "
        f"{overall['Mean_Top_Quartile_Hit_Rate']:.4f}"
    )

    print(
        "Mean daily bottom-quartile hit rate: "
        f"{overall['Mean_Bottom_Quartile_Hit_Rate']:.4f}"
    )

    print(
        "Raw XGBoost predictions below 0: "
        f"{int(overall['Predictions_Below_Zero']):,}"
    )

    print(
        "Raw XGBoost predictions above 1: "
        f"{int(overall['Predictions_Above_One']):,}"
    )

    print(
        "Constant-prediction trading days: "
        f"{int(overall['Constant_Prediction_Days']):,}"
    )

    print(
        "Total runtime: "
        f"{total_runtime_seconds / 60:.1f} minutes"
    )

    print(
        "\nIntegrity checks passed:"
    )

    print(
        "- Every test observation was predicted exactly once."
    )

    print(
        "- Every inner fold was reconstructed from stored "
        "chronological dates."
    )

    print(
        "- Every inner training-validation boundary retained "
        "the stored purge date."
    )

    print(
        "- Winsorisation bounds were estimated independently "
        "inside each inner fold."
    )

    print(
        "- RMSE controlled early stopping within each "
        "inner model fit."
    )

    print(
        "- Hyperparameters were selected by mean daily "
        "validation Spearman IC."
    )

    print(
        "- Outer test periods were never used for tuning or "
        "early stopping."
    )

    print(
        "- Final boosting rounds were derived from the median "
        "inner-fold early-stopping round."
    )

    print(
        "- Outer training-only winsorisation was applied "
        "unchanged to the corresponding test observations."
    )

    print(
        "- XGBoost used the same nine predictors and outer "
        "test observations as Pooled OLS."
    )

    print(
        "- No risk-free-rate or portfolio-performance "
        "variables were required."
    )

    print(
        "\nResults:"
    )

    for output_file in [
        OUTPUT_PREDICTIONS,
        OUTPUT_DAILY_PERFORMANCE,
        OUTPUT_WINDOW_PERFORMANCE,
        OUTPUT_OVERALL_PERFORMANCE,
    ]:
        print(
            f"- {relative_path(output_file)}"
        )

    print(
        "\nAudit and figure outputs:"
    )

    for output_file in [
        OUTPUT_SEARCH_RESULTS,
        OUTPUT_SELECTED_PARAMETERS,
        OUTPUT_WINSORISATION,
        OUTPUT_FEATURE_IMPORTANCE,
        OUTPUT_ESTIMATION_SUMMARY,
        OUTPUT_FIGURE,
    ]:
        print(
            f"- {relative_path(output_file)}"
        )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    main()