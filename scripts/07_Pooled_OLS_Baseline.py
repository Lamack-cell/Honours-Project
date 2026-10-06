"""
Script 07: Pooled OLS Expanding-Window Estimation
--------------------------------------------------

Purpose
-------
Estimates the primary nine-feature Pooled Cross-Sectional OLS benchmark
across the quarterly expanding-window schedule produced by Script 06.

For each outer window, the script:

1. loads the stored chronological training and test observations;
2. estimates 1st and 99th percentile winsorisation thresholds using
   only the current training sample;
3. applies those thresholds unchanged to the current test sample;
4. estimates one pooled OLS scoring function;
5. calculates two-way cluster-robust inference by date and ticker;
6. generates out-of-sample predictions;
7. ranks predictions independently within each test date; and
8. calculates IC, MAE, RMSE and quartile hit-rate diagnostics.

No test observation is used to estimate preprocessing thresholds,
regression coefficients or covariance estimates.

Inputs
------
data/processed/06_validation/JSE_Master_Model_Panel.csv
data/processed/06_validation/JSE_Walk_Forward_Windows.csv
outputs/audits/06_validation/JSE_Model_Feature_Configuration.csv

Results
-------
results/07_ols/JSE_OLS_Walk_Forward_Predictions.csv
results/07_ols/JSE_OLS_Daily_Performance.csv
results/07_ols/JSE_OLS_Window_Performance.csv
results/07_ols/JSE_OLS_Overall_Performance.csv

Audit outputs
-------------
outputs/audits/07_ols/JSE_OLS_Coefficient_Estimates.csv
outputs/audits/07_ols/JSE_OLS_Winsorisation_Thresholds.csv
outputs/audits/07_ols/JSE_OLS_Estimation_Summary.csv

Figure output
-------------
outputs/figures/chapter_5/figure_5_2_ols_coefficient_stability.png

Run from the project root
-------------------------
python .\\scripts\\07_Pooled_OLS_Baseline.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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
    PROJECT_ROOT
    / "data"
    / "processed"
    / "06_validation"
)

RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "07_ols"
)

AUDIT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "audits"
    / "07_ols"
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

INPUT_FEATURE_CONFIGURATION = (
    PROJECT_ROOT
    / "outputs"
    / "audits"
    / "06_validation"
    / "JSE_Model_Feature_Configuration.csv"
)


OUTPUT_PREDICTIONS = (
    RESULTS_DIRECTORY
    / "JSE_OLS_Walk_Forward_Predictions.csv"
)

OUTPUT_DAILY_PERFORMANCE = (
    RESULTS_DIRECTORY
    / "JSE_OLS_Daily_Performance.csv"
)

OUTPUT_WINDOW_PERFORMANCE = (
    RESULTS_DIRECTORY
    / "JSE_OLS_Window_Performance.csv"
)

OUTPUT_OVERALL_PERFORMANCE = (
    RESULTS_DIRECTORY
    / "JSE_OLS_Overall_Performance.csv"
)

OUTPUT_COEFFICIENTS = (
    AUDIT_DIRECTORY
    / "JSE_OLS_Coefficient_Estimates.csv"
)

OUTPUT_WINSORISATION = (
    AUDIT_DIRECTORY
    / "JSE_OLS_Winsorisation_Thresholds.csv"
)

OUTPUT_ESTIMATION_SUMMARY = (
    AUDIT_DIRECTORY
    / "JSE_OLS_Estimation_Summary.csv"
)

OUTPUT_FIGURE = (
    FIGURE_DIRECTORY
    / "figure_5_2_ols_coefficient_stability.png"
)


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

PRIMARY_FEATURES = [
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

DEFAULT_WINSOR_LOWER_QUANTILE = 0.01
DEFAULT_WINSOR_UPPER_QUANTILE = 0.99

MINIMUM_DAILY_CROSS_SECTION = 5
TOP_BOTTOM_FRACTION = 0.25

NORMAL_CRITICAL_VALUE = float(
    norm.ppf(0.975)
)


# ============================================================
# 3. GENERAL HELPERS
# ============================================================

def relative_path(path: Path) -> str:
    """Return a readable project-relative path."""

    return str(
        path.relative_to(
            PROJECT_ROOT
        )
    )


def parse_boolean_series(
    series: pd.Series,
) -> pd.Series:
    """Convert common CSV boolean representations to Boolean values."""

    if pd.api.types.is_bool_dtype(
        series
    ):
        return series

    normalised = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }

    converted = normalised.map(
        mapping
    )

    if converted.isna().any():
        invalid_values = sorted(
            normalised.loc[
                converted.isna()
            ].unique()
        )

        raise ValueError(
            "Unrecognised Boolean values in the feature configuration: "
            f"{invalid_values}"
        )

    return converted.astype(
        bool
    )


# ============================================================
# 4. INPUT VALIDATION
# ============================================================

def validate_required_files() -> None:
    """Confirm that all Script 06 outputs exist."""

    required_files = [
        INPUT_MASTER_PANEL,
        INPUT_OUTER_WINDOWS,
        INPUT_FEATURE_CONFIGURATION,
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
    """Validate the common modelling panel."""

    required_columns = {
        "Date",
        "Ticker",
        "Forward_Return",
        "N_t",
        "Model_N_t",
        TARGET_COLUMN,
        *PRIMARY_FEATURES,
    }

    missing_columns = required_columns.difference(
        panel.columns
    )

    if missing_columns:
        raise ValueError(
            "The master model panel is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if panel.empty:
        raise ValueError(
            "The master model panel contains no observations."
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
            f"The master model panel contains {duplicate_count:,} "
            "duplicate Date-Ticker observations."
        )

    required_values = [
        "Date",
        "Ticker",
        "Forward_Return",
        "N_t",
        "Model_N_t",
        TARGET_COLUMN,
        *PRIMARY_FEATURES,
    ]

    missing_values = int(
        panel[
            required_values
        ]
        .isna()
        .sum()
        .sum()
    )

    if missing_values > 0:
        raise ValueError(
            f"The master model panel contains {missing_values:,} "
            "missing required values."
        )

    numeric_matrix = panel[
        [
            "Forward_Return",
            TARGET_COLUMN,
            *PRIMARY_FEATURES,
        ]
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        numeric_matrix
    ).all():
        raise ValueError(
            "The master model panel contains non-finite model values."
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

    observed_model_sizes = (
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
        observed_model_sizes
    ).all():
        raise ValueError(
            "Stored Model_N_t does not match the observed "
            "daily model cross-sectional size."
        )


def validate_outer_windows(
    windows: pd.DataFrame,
) -> None:
    """Validate the stored outer walk-forward schedule."""

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
            "The outer-window file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if windows.empty:
        raise ValueError(
            "No outer walk-forward windows were supplied."
        )

    if windows[
        "Outer_Window_ID"
    ].duplicated().any():
        raise ValueError(
            "Duplicate Outer_Window_ID values were detected."
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
            "At least one training window reaches its purge date."
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
            "At least one purge date is not earlier than "
            "its test start."
        )

    ordered = (
        windows
        .sort_values(
            "Test_Start"
        )
        .reset_index(
            drop=True
        )
    )

    for index in range(
        1,
        len(
            ordered
        ),
    ):
        previous_test_end = pd.Timestamp(
            ordered.loc[
                index - 1,
                "Test_End",
            ]
        )

        current_test_start = pd.Timestamp(
            ordered.loc[
                index,
                "Test_Start",
            ]
        )

        if current_test_start <= previous_test_end:
            raise ValueError(
                "Outer test windows overlap."
            )


# ============================================================
# 5. FEATURE CONFIGURATION
# ============================================================

def read_winsorisation_configuration() -> tuple[float, float]:
    """Read and validate the stored winsorisation quantiles."""

    configuration = pd.read_csv(
        INPUT_FEATURE_CONFIGURATION
    )

    required_columns = {
        "Feature",
        "Used_In_Primary_OLS",
        "Winsor_Lower_Quantile",
        "Winsor_Upper_Quantile",
    }

    missing_columns = required_columns.difference(
        configuration.columns
    )

    if missing_columns:
        raise ValueError(
            "The feature-configuration file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    configuration[
        "Used_In_Primary_OLS"
    ] = parse_boolean_series(
        configuration[
            "Used_In_Primary_OLS"
        ]
    )

    configured_features = configuration.loc[
        configuration[
            "Used_In_Primary_OLS"
        ],
        "Feature",
    ].tolist()

    if configured_features != PRIMARY_FEATURES:
        raise ValueError(
            "The primary OLS feature order does not match Script 07.\n"
            f"Configured: {configured_features}\n"
            f"Expected: {PRIMARY_FEATURES}"
        )

    lower_values = (
        configuration.loc[
            configuration[
                "Used_In_Primary_OLS"
            ],
            "Winsor_Lower_Quantile",
        ]
        .dropna()
        .astype(
            float
        )
        .unique()
    )

    upper_values = (
        configuration.loc[
            configuration[
                "Used_In_Primary_OLS"
            ],
            "Winsor_Upper_Quantile",
        ]
        .dropna()
        .astype(
            float
        )
        .unique()
    )

    if len(
        lower_values
    ) == 0:
        lower_quantile = (
            DEFAULT_WINSOR_LOWER_QUANTILE
        )

    elif len(
        lower_values
    ) == 1:
        lower_quantile = float(
            lower_values[
                0
            ]
        )

    else:
        raise ValueError(
            "Multiple lower winsorisation quantiles were supplied."
        )

    if len(
        upper_values
    ) == 0:
        upper_quantile = (
            DEFAULT_WINSOR_UPPER_QUANTILE
        )

    elif len(
        upper_values
    ) == 1:
        upper_quantile = float(
            upper_values[
                0
            ]
        )

    else:
        raise ValueError(
            "Multiple upper winsorisation quantiles were supplied."
        )

    if not (
        0.0
        <= lower_quantile
        < upper_quantile
        <= 1.0
    ):
        raise ValueError(
            "Invalid winsorisation quantiles were supplied."
        )

    return (
        lower_quantile,
        upper_quantile,
    )


# ============================================================
# 6. WINSORISATION
# ============================================================

def estimate_winsorisation_thresholds(
    training_data: pd.DataFrame,
    features: list[str],
    lower_quantile: float,
    upper_quantile: float,
) -> pd.DataFrame:
    """Estimate clipping thresholds using training observations only."""

    records: list[
        dict[str, float | str]
    ] = []

    for feature in features:
        lower_bound = float(
            training_data[
                feature
            ].quantile(
                lower_quantile
            )
        )

        upper_bound = float(
            training_data[
                feature
            ].quantile(
                upper_quantile
            )
        )

        if not np.isfinite(
            lower_bound
        ):
            raise ValueError(
                f"Non-finite lower winsorisation threshold "
                f"for {feature}."
            )

        if not np.isfinite(
            upper_bound
        ):
            raise ValueError(
                f"Non-finite upper winsorisation threshold "
                f"for {feature}."
            )

        if lower_bound > upper_bound:
            raise ValueError(
                f"Winsorisation bounds are reversed for {feature}."
            )

        records.append(
            {
                "Feature": feature,
                "Lower_Quantile": lower_quantile,
                "Upper_Quantile": upper_quantile,
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
    """Apply stored thresholds without re-estimating them."""

    transformed = data.copy()

    for threshold in thresholds.itertuples(
        index=False
    ):
        transformed[
            threshold.Feature
        ] = transformed[
            threshold.Feature
        ].clip(
            lower=threshold.Lower_Bound,
            upper=threshold.Upper_Bound,
        )

    return transformed


# ============================================================
# 7. TWO-WAY CLUSTERED INFERENCE
# ============================================================

def calculate_two_way_cluster_inference(
    fitted_model: sm.regression.linear_model.RegressionResultsWrapper,
    training_data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate date-and-ticker two-way cluster-robust inference."""

    date_groups = pd.factorize(
        training_data[
            "Date"
        ],
        sort=True,
    )[0]

    ticker_groups = pd.factorize(
        training_data[
            "Ticker"
        ],
        sort=True,
    )[0]

    try:
        covariance_both, _, _ = cov_cluster_2groups(
            fitted_model,
            date_groups,
            ticker_groups,
            use_correction=True,
        )

    except Exception as error:
        raise RuntimeError(
            "Two-way clustered covariance estimation failed."
        ) from error

    covariance_diagonal = np.diag(
        covariance_both
    )

    if np.any(
        covariance_diagonal
        < -1e-12
    ):
        raise ValueError(
            "The two-way clustered covariance matrix contains "
            "materially negative diagonal elements."
        )

    standard_errors = np.sqrt(
        np.clip(
            covariance_diagonal,
            0.0,
            None,
        )
    )

    parameters = fitted_model.params.to_numpy(
        dtype=float
    )

    z_statistics = np.divide(
        parameters,
        standard_errors,
        out=np.full_like(
            parameters,
            np.nan,
        ),
        where=standard_errors > 0,
    )

    p_values = (
        2.0
        * norm.sf(
            np.abs(
                z_statistics
            )
        )
    )

    lower_confidence = (
        parameters
        - NORMAL_CRITICAL_VALUE
        * standard_errors
    )

    upper_confidence = (
        parameters
        + NORMAL_CRITICAL_VALUE
        * standard_errors
    )

    return pd.DataFrame(
        {
            "Term": fitted_model.params.index,
            "Coefficient": parameters,
            "Two_Way_Cluster_SE": standard_errors,
            "Z_Statistic": z_statistics,
            "P_Value_Normal_Approximation": p_values,
            "CI_95_Lower": lower_confidence,
            "CI_95_Upper": upper_confidence,
        }
    )


# ============================================================
# 8. DAILY PERFORMANCE FUNCTIONS
# ============================================================

def safe_daily_spearman_ic(
    predicted_scores: pd.Series,
    realised_target: pd.Series,
) -> float:
    """
    Calculate a safe daily Spearman IC.

    A constant prediction vector contains no ranking information and is
    assigned IC = 0 rather than producing a missing correlation.
    """

    if len(
        predicted_scores
    ) < MINIMUM_DAILY_CROSS_SECTION:
        return np.nan

    if (
        predicted_scores.nunique(
            dropna=True
        )
        <= 1
    ):
        return 0.0

    if (
        realised_target.nunique(
            dropna=True
        )
        <= 1
    ):
        return 0.0

    result = spearmanr(
        predicted_scores,
        realised_target,
    )

    statistic = float(
        result.statistic
    )

    if not np.isfinite(
        statistic
    ):
        return 0.0

    return statistic


def deterministic_top_bottom_hit_rates(
    group: pd.DataFrame,
) -> tuple[float, float, int]:
    """Calculate deterministic daily top- and bottom-quartile overlap."""

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
            "OLS_Raw_Score",
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
    """Calculate date-level OLS predictive diagnostics."""

    records: list[
        dict[str, object]
    ] = []

    for date, group in predictions.groupby(
        "Date",
        sort=True,
    ):
        number_of_assets = len(
            group
        )

        daily_ic = safe_daily_spearman_ic(
            predicted_scores=group[
                "OLS_Raw_Score"
            ],
            realised_target=group[
                TARGET_COLUMN
            ],
        )

        errors = (
            group[
                "OLS_Raw_Score"
            ]
            - group[
                TARGET_COLUMN
            ]
        )

        daily_mae = float(
            np.mean(
                np.abs(
                    errors
                )
            )
        )

        daily_rmse = float(
            np.sqrt(
                np.mean(
                    errors ** 2
                )
            )
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
                "Model_Eligible_Securities": number_of_assets,
                "Daily_IC": daily_ic,
                "Daily_MAE": daily_mae,
                "Daily_RMSE": daily_rmse,
                "Quartile_Size": quartile_size,
                "Top_Quartile_Hit_Rate": top_hit_rate,
                "Bottom_Quartile_Hit_Rate": bottom_hit_rate,
                "Predictions_Below_Zero": int(
                    group[
                        "OLS_Raw_Score"
                    ].lt(
                        0
                    ).sum()
                ),
                "Predictions_Above_One": int(
                    group[
                        "OLS_Raw_Score"
                    ].gt(
                        1
                    ).sum()
                ),
            }
        )

    return pd.DataFrame(
        records
    )


def summarise_performance(
    predictions: pd.DataFrame,
    daily_performance: pd.DataFrame,
    grouping_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Summarise observation-level and daily metrics."""

    if grouping_columns is None:
        grouped_predictions = [
            (
                "Overall",
                predictions,
            )
        ]

        grouped_daily = {
            "Overall": daily_performance
        }

    else:
        grouped_predictions = list(
            predictions.groupby(
                grouping_columns,
                sort=False,
                dropna=False,
            )
        )

        grouped_daily = {
            key: group
            for key, group in daily_performance.groupby(
                grouping_columns,
                sort=False,
                dropna=False,
            )
        }

    records: list[
        dict[str, object]
    ] = []

    for key, prediction_group in grouped_predictions:
        daily_group = grouped_daily[
            key
        ]

        valid_ic = daily_group[
            "Daily_IC"
        ].dropna()

        if valid_ic.empty:
            mean_ic = np.nan
            median_ic = np.nan
            ic_standard_deviation = np.nan
            ic_information_ratio = np.nan

        else:
            mean_ic = float(
                valid_ic.mean()
            )

            median_ic = float(
                valid_ic.median()
            )

            ic_standard_deviation = float(
                valid_ic.std(
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
            prediction_group[
                "OLS_Raw_Score"
            ]
            - prediction_group[
                TARGET_COLUMN
            ]
        )

        record: dict[
            str,
            object,
        ] = {
            "Out_Of_Sample_Observations": len(
                prediction_group
            ),
            "Out_Of_Sample_Dates": prediction_group[
                "Date"
            ].nunique(),
            "Mean_Daily_IC": mean_ic,
            "Median_Daily_IC": median_ic,
            "Daily_IC_Standard_Deviation": (
                ic_standard_deviation
            ),
            "Daily_IC_Information_Ratio": (
                ic_information_ratio
            ),
            "Positive_IC_Days": int(
                valid_ic.gt(
                    0
                ).sum()
            ),
            "Zero_IC_Days": int(
                valid_ic.eq(
                    0
                ).sum()
            ),
            "Negative_IC_Days": int(
                valid_ic.lt(
                    0
                ).sum()
            ),
            "Positive_IC_Percentage": float(
                valid_ic.gt(
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
                daily_group[
                    "Daily_MAE"
                ].mean()
            ),
            "Mean_Daily_RMSE": float(
                daily_group[
                    "Daily_RMSE"
                ].mean()
            ),
            "Mean_Top_Quartile_Hit_Rate": float(
                daily_group[
                    "Top_Quartile_Hit_Rate"
                ].mean()
            ),
            "Mean_Bottom_Quartile_Hit_Rate": float(
                daily_group[
                    "Bottom_Quartile_Hit_Rate"
                ].mean()
            ),
            "Predictions_Below_Zero": int(
                prediction_group[
                    "OLS_Raw_Score"
                ].lt(
                    0
                ).sum()
            ),
            "Predictions_Above_One": int(
                prediction_group[
                    "OLS_Raw_Score"
                ].gt(
                    1
                ).sum()
            ),
        }

        if grouping_columns is None:
            record[
                "Model"
            ] = "Pooled_OLS"

        else:
            if not isinstance(
                key,
                tuple,
            ):
                key = (
                    key,
                )

            for column, value in zip(
                grouping_columns,
                key,
            ):
                record[
                    column
                ] = value

        records.append(
            record
        )

    output = pd.DataFrame(
        records
    )

    if grouping_columns is None:
        ordered_columns = [
            "Model",
            *[
                column
                for column in output.columns
                if column != "Model"
            ],
        ]

        output = output[
            ordered_columns
        ]

    else:
        ordered_columns = [
            *grouping_columns,
            *[
                column
                for column in output.columns
                if column
                not in grouping_columns
            ],
        ]

        output = output[
            ordered_columns
        ]

    return output


# ============================================================
# 9. COEFFICIENT FIGURE
# ============================================================

def create_coefficient_stability_figure(
    coefficient_table: pd.DataFrame,
) -> None:
    """Create a feature-by-quarter coefficient heatmap."""

    plotting_data = coefficient_table.loc[
        coefficient_table[
            "Term"
        ].ne(
            "const"
        )
    ].copy()

    coefficient_matrix = plotting_data.pivot(
        index="Term",
        columns="Test_Quarter",
        values="Coefficient",
    )

    coefficient_matrix = coefficient_matrix.reindex(
        PRIMARY_FEATURES
    )

    maximum_absolute_coefficient = float(
        np.nanmax(
            np.abs(
                coefficient_matrix.to_numpy(
                    dtype=float
                )
            )
        )
    )

    if (
        not np.isfinite(
            maximum_absolute_coefficient
        )
        or maximum_absolute_coefficient == 0
    ):
        maximum_absolute_coefficient = 1.0

    figure_width = max(
        10.0,
        len(
            coefficient_matrix.columns
        )
        * 1.05,
    )

    figure_height = max(
        6.0,
        len(
            coefficient_matrix.index
        )
        * 0.62,
    )

    figure, axis = plt.subplots(
        figsize=(
            figure_width,
            figure_height,
        )
    )

    image = axis.imshow(
        coefficient_matrix.to_numpy(
            dtype=float
        ),
        aspect="auto",
        cmap="coolwarm",
        vmin=-maximum_absolute_coefficient,
        vmax=maximum_absolute_coefficient,
    )

    axis.set_xticks(
        np.arange(
            len(
                coefficient_matrix.columns
            )
        )
    )

    axis.set_yticks(
        np.arange(
            len(
                coefficient_matrix.index
            )
        )
    )

    axis.set_xticklabels(
        coefficient_matrix.columns,
        rotation=45,
        ha="right",
    )

    axis.set_yticklabels(
        coefficient_matrix.index
    )

    axis.set_title(
        "Pooled OLS Coefficient Stability across "
        "Expanding Training Windows"
    )

    axis.set_xlabel(
        "Out-of-sample test quarter"
    )

    axis.set_ylabel(
        "Standardised predictor"
    )

    for row_index in range(
        coefficient_matrix.shape[
            0
        ]
    ):
        for column_index in range(
            coefficient_matrix.shape[
                1
            ]
        ):
            value = coefficient_matrix.iloc[
                row_index,
                column_index,
            ]

            axis.text(
                column_index,
                row_index,
                f"{value:.4f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    colour_bar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )

    colour_bar.set_label(
        "OLS coefficient"
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
    print("SCRIPT 07: POOLED OLS EXPANDING-WINDOW ESTIMATION")
    print("=" * 78)

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

    (
        lower_quantile,
        upper_quantile,
    ) = read_winsorisation_configuration()

    # --------------------------------------------------------
    # 10.1 Load the common model panel
    # --------------------------------------------------------

    master_panel = pd.read_csv(
        INPUT_MASTER_PANEL
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

    validate_master_panel(
        master_panel
    )

    # --------------------------------------------------------
    # 10.2 Load outer walk-forward schedule
    # --------------------------------------------------------

    outer_windows = pd.read_csv(
        INPUT_OUTER_WINDOWS
    )

    date_columns = [
        "Train_Start",
        "Train_End",
        "Purged_Boundary_Date",
        "Test_Start",
        "Test_End",
    ]

    for column in date_columns:
        outer_windows[
            column
        ] = pd.to_datetime(
            outer_windows[
                column
            ],
            errors="raise",
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

    validate_outer_windows(
        outer_windows
    )

    prediction_frames: list[
        pd.DataFrame
    ] = []

    coefficient_frames: list[
        pd.DataFrame
    ] = []

    winsorisation_frames: list[
        pd.DataFrame
    ] = []

    estimation_records: list[
        dict[str, object]
    ] = []

    # --------------------------------------------------------
    # 10.3 Estimate each expanding-window OLS model
    # --------------------------------------------------------

    for window in outer_windows.itertuples(
        index=False
    ):
        train_start = pd.Timestamp(
            window.Train_Start
        )

        train_end = pd.Timestamp(
            window.Train_End
        )

        test_start = pd.Timestamp(
            window.Test_Start
        )

        test_end = pd.Timestamp(
            window.Test_End
        )

        train_mask = master_panel[
            "Date"
        ].between(
            train_start,
            train_end,
            inclusive="both",
        )

        test_mask = master_panel[
            "Date"
        ].between(
            test_start,
            test_end,
            inclusive="both",
        )

        training_data = master_panel.loc[
            train_mask
        ].copy()

        test_data = master_panel.loc[
            test_mask
        ].copy()

        if training_data.empty:
            raise ValueError(
                f"{window.Outer_Window_ID} contains no training rows."
            )

        if test_data.empty:
            raise ValueError(
                f"{window.Outer_Window_ID} contains no test rows."
            )

        if len(
            training_data
        ) != int(
            window.Train_Observations
        ):
            raise AssertionError(
                f"{window.Outer_Window_ID} training-row count does "
                "not match the stored schedule."
            )

        if len(
            test_data
        ) != int(
            window.Test_Observations
        ):
            raise AssertionError(
                f"{window.Outer_Window_ID} test-row count does "
                "not match the stored schedule."
            )

        if training_data[
            "Date"
        ].nunique() != int(
            window.Train_Dates
        ):
            raise AssertionError(
                f"{window.Outer_Window_ID} training-date count does "
                "not match the stored schedule."
            )

        if test_data[
            "Date"
        ].nunique() != int(
            window.Test_Dates
        ):
            raise AssertionError(
                f"{window.Outer_Window_ID} test-date count does "
                "not match the stored schedule."
            )

        if training_data[
            "Date"
        ].max() >= test_data[
            "Date"
        ].min():
            raise AssertionError(
                f"{window.Outer_Window_ID} training and test dates overlap."
            )

        if "Next_Date" in training_data.columns:
            leakage_count = int(
                (
                    training_data[
                        "Next_Date"
                    ]
                    >= test_start
                ).sum()
            )

            if leakage_count > 0:
                raise AssertionError(
                    f"{window.Outer_Window_ID} contains "
                    f"{leakage_count:,} target-boundary leakage rows."
                )

        # ----------------------------------------------------
        # Estimate training-only winsorisation thresholds
        # ----------------------------------------------------

        thresholds = estimate_winsorisation_thresholds(
            training_data=training_data,
            features=PRIMARY_FEATURES,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )

        threshold_values = thresholds.copy()

        thresholds.insert(
            0,
            "Outer_Window_ID",
            window.Outer_Window_ID,
        )

        thresholds.insert(
            1,
            "Test_Quarter",
            window.Test_Quarter,
        )

        thresholds.insert(
            2,
            "Train_Start",
            train_start,
        )

        thresholds.insert(
            3,
            "Train_End",
            train_end,
        )

        winsorisation_frames.append(
            thresholds
        )

        training_transformed = apply_winsorisation(
            data=training_data,
            thresholds=threshold_values,
        )

        test_transformed = apply_winsorisation(
            data=test_data,
            thresholds=threshold_values,
        )

        # ----------------------------------------------------
        # Estimate pooled OLS
        # ----------------------------------------------------

        x_train = sm.add_constant(
            training_transformed[
                PRIMARY_FEATURES
            ],
            has_constant="add",
        )

        y_train = training_transformed[
            TARGET_COLUMN
        ]

        x_test = sm.add_constant(
            test_transformed[
                PRIMARY_FEATURES
            ],
            has_constant="add",
        )

        x_test = x_test[
            x_train.columns
        ]

        fitted_model = sm.OLS(
            endog=y_train,
            exog=x_train,
            missing="raise",
        ).fit()

        # ----------------------------------------------------
        # Two-way clustered coefficient inference
        # ----------------------------------------------------

        inference = calculate_two_way_cluster_inference(
            fitted_model=fitted_model,
            training_data=training_transformed,
        )

        inference.insert(
            0,
            "Outer_Window_ID",
            window.Outer_Window_ID,
        )

        inference.insert(
            1,
            "Test_Quarter",
            window.Test_Quarter,
        )

        inference.insert(
            2,
            "Train_Start",
            train_start,
        )

        inference.insert(
            3,
            "Train_End",
            train_end,
        )

        coefficient_frames.append(
            inference
        )

        # ----------------------------------------------------
        # Generate out-of-sample predictions
        # ----------------------------------------------------

        raw_predictions = fitted_model.predict(
            x_test
        )

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
                    in test_data.columns
                    and column
                    not in prediction_columns
                )
            ]
        )

        window_predictions = test_data[
            prediction_columns
        ].copy()

        window_predictions.insert(
            0,
            "Outer_Window_ID",
            window.Outer_Window_ID,
        )

        window_predictions.insert(
            1,
            "Test_Quarter",
            window.Test_Quarter,
        )

        window_predictions[
            "OLS_Raw_Score"
        ] = raw_predictions.to_numpy(
            dtype=float
        )

        if not np.isfinite(
            window_predictions[
                "OLS_Raw_Score"
            ].to_numpy(
                dtype=float
            )
        ).all():
            raise AssertionError(
                f"{window.Outer_Window_ID} produced non-finite "
                "OLS predictions."
            )

        window_predictions[
            "OLS_Predicted_Rank"
        ] = (
            window_predictions
            .groupby(
                "Date"
            )[
                "OLS_Raw_Score"
            ]
            .rank(
                method="average",
                ascending=True,
            )
        )

        window_predictions[
            "OLS_Prediction_N_t"
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

        denominator = (
            window_predictions[
                "OLS_Prediction_N_t"
            ]
            - 1
        )

        window_predictions[
            "OLS_Predicted_Percentile"
        ] = np.where(
            denominator.gt(
                0
            ),
            (
                window_predictions[
                    "OLS_Predicted_Rank"
                ]
                - 1.0
            )
            / denominator,
            0.5,
        )

        if not window_predictions[
            "OLS_Predicted_Percentile"
        ].between(
            0,
            1,
            inclusive="both",
        ).all():
            raise AssertionError(
                f"{window.Outer_Window_ID} produced predicted "
                "percentiles outside [0, 1]."
            )

        prediction_frames.append(
            window_predictions
        )

        estimation_records.append(
            {
                "Outer_Window_ID": window.Outer_Window_ID,
                "Test_Quarter": window.Test_Quarter,
                "Train_Start": train_start,
                "Train_End": train_end,
                "Purged_Boundary_Date": pd.Timestamp(
                    window.Purged_Boundary_Date
                ),
                "Test_Start": test_start,
                "Test_End": test_end,
                "Training_Observations": len(
                    training_data
                ),
                "Training_Dates": training_data[
                    "Date"
                ].nunique(),
                "Training_Tickers": training_data[
                    "Ticker"
                ].nunique(),
                "Test_Observations": len(
                    test_data
                ),
                "Test_Dates": test_data[
                    "Date"
                ].nunique(),
                "Test_Tickers": test_data[
                    "Ticker"
                ].nunique(),
                "OLS_R_Squared_Training": float(
                    fitted_model.rsquared
                ),
                "OLS_Adjusted_R_Squared_Training": float(
                    fitted_model.rsquared_adj
                ),
                "Residual_Degrees_Of_Freedom": float(
                    fitted_model.df_resid
                ),
                "Date_Clusters": training_data[
                    "Date"
                ].nunique(),
                "Ticker_Clusters": training_data[
                    "Ticker"
                ].nunique(),
                "Winsor_Lower_Quantile": lower_quantile,
                "Winsor_Upper_Quantile": upper_quantile,
                "Raw_Prediction_Minimum": float(
                    window_predictions[
                        "OLS_Raw_Score"
                    ].min()
                ),
                "Raw_Prediction_Maximum": float(
                    window_predictions[
                        "OLS_Raw_Score"
                    ].max()
                ),
                "Constant_Prediction_Dates": int(
                    window_predictions
                    .groupby(
                        "Date"
                    )[
                        "OLS_Raw_Score"
                    ]
                    .nunique()
                    .le(
                        1
                    )
                    .sum()
                ),
            }
        )

        print(
            f"Completed {window.Outer_Window_ID} | "
            f"{window.Test_Quarter} | "
            f"train rows {len(training_data):,} | "
            f"test rows {len(test_data):,}"
        )

    # --------------------------------------------------------
    # 10.4 Combine all out-of-sample predictions
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

    if predictions.duplicated(
        [
            "Date",
            "Ticker",
        ]
    ).any():
        raise AssertionError(
            "A test observation was predicted more than once."
        )

    expected_test_observations = int(
        outer_windows[
            "Test_Observations"
        ].sum()
    )

    expected_test_dates = int(
        outer_windows[
            "Test_Dates"
        ].sum()
    )

    if len(
        predictions
    ) != expected_test_observations:
        raise AssertionError(
            "The OLS prediction file does not cover every stored "
            "out-of-sample observation."
        )

    if predictions[
        "Date"
    ].nunique() != expected_test_dates:
        raise AssertionError(
            "The OLS prediction file does not cover every stored "
            "out-of-sample date."
        )

    # --------------------------------------------------------
    # 10.5 Performance outputs
    # --------------------------------------------------------

    daily_performance = calculate_daily_performance(
        predictions
    )

    window_performance = summarise_performance(
        predictions=predictions,
        daily_performance=daily_performance,
        grouping_columns=[
            "Outer_Window_ID",
            "Test_Quarter",
        ],
    )

    overall_performance = summarise_performance(
        predictions=predictions,
        daily_performance=daily_performance,
        grouping_columns=None,
    )

    coefficients = pd.concat(
        coefficient_frames,
        ignore_index=True,
    )

    winsorisation_thresholds = pd.concat(
        winsorisation_frames,
        ignore_index=True,
    )

    estimation_summary = pd.DataFrame(
        estimation_records
    )

    expected_coefficient_rows = (
        len(
            outer_windows
        )
        * (
            len(
                PRIMARY_FEATURES
            )
            + 1
        )
    )

    if len(
        coefficients
    ) != expected_coefficient_rows:
        raise AssertionError(
            "Unexpected number of OLS coefficient rows."
        )

    expected_winsor_rows = (
        len(
            outer_windows
        )
        * len(
            PRIMARY_FEATURES
        )
    )

    if len(
        winsorisation_thresholds
    ) != expected_winsor_rows:
        raise AssertionError(
            "Unexpected number of OLS winsorisation rows."
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

    coefficients.to_csv(
        OUTPUT_COEFFICIENTS,
        index=False,
    )

    winsorisation_thresholds.to_csv(
        OUTPUT_WINSORISATION,
        index=False,
    )

    estimation_summary.to_csv(
        OUTPUT_ESTIMATION_SUMMARY,
        index=False,
    )

    create_coefficient_stability_figure(
        coefficients
    )

    # --------------------------------------------------------
    # 10.7 Terminal summary
    # --------------------------------------------------------

    overall = overall_performance.iloc[
        0
    ]

    constant_prediction_dates = int(
        predictions
        .groupby(
            "Date"
        )[
            "OLS_Raw_Score"
        ]
        .nunique()
        .le(
            1
        )
        .sum()
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "SCRIPT 07 EMPIRICAL OUTPUTS"
    )

    print(
        "=" * 78
    )

    print(
        "Outer OLS models estimated: "
        f"{len(outer_windows):,}"
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
        "Raw OLS predictions below 0: "
        f"{int(overall['Predictions_Below_Zero']):,}"
    )

    print(
        "Raw OLS predictions above 1: "
        f"{int(overall['Predictions_Above_One']):,}"
    )

    print(
        "Trading dates with constant OLS predictions: "
        f"{constant_prediction_dates:,}"
    )

    print(
        "\nIntegrity checks passed:"
    )

    print(
        "- All nine configured predictors were used in the "
        "stored order."
    )

    print(
        "- Winsorisation thresholds were estimated separately "
        "inside each outer training window."
    )

    print(
        "- Test predictors were clipped using training thresholds "
        "without re-estimation."
    )

    print(
        "- Regression coefficients were estimated using training "
        "observations only."
    )

    print(
        "- Coefficient inference uses two-way clustering by "
        "trading date and ticker."
    )

    print(
        "- Every stored out-of-sample observation was predicted "
        "exactly once."
    )

    print(
        "- Predicted percentiles were constructed independently "
        "within each test-date cross-section."
    )

    print(
        "- No risk-free-rate or portfolio-performance fields "
        "were required."
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
        OUTPUT_COEFFICIENTS,
        OUTPUT_WINSORISATION,
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
    main()