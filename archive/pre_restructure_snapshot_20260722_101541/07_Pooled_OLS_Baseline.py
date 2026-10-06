"""
Script 07: Pooled OLS Expanding-Window Estimation
--------------------------------------------------

Purpose
-------
Estimates the primary nine-feature Pooled Cross-Sectional OLS benchmark
across the common quarterly expanding-window schedule created by Script 06.

For every outer window, the script:
1. selects the stored training and test dates;
2. estimates 1st and 99th percentile winsorisation thresholds using only
   the current training sample;
3. applies those thresholds unchanged to the current test sample;
4. estimates one pooled OLS scoring function;
5. calculates two-way cluster-robust covariance by trading date and ticker;
6. predicts the quarterly out-of-sample cross-section;
7. ranks predictions independently within each test date; and
8. records daily and quarterly predictive-performance diagnostics.

The target is never used to transform the predictors. No test observation
is used to estimate preprocessing thresholds or regression coefficients.
The risk-free rate is not required because the study is evaluated as a
statistical comparison rather than a portfolio-performance backtest.

Inputs
------
1. JSE_Master_Model_Panel.csv
2. JSE_Walk_Forward_Windows.csv
3. JSE_Model_Feature_Configuration.csv
   or outputs/audits/JSE_Model_Feature_Configuration.csv

Outputs
-------
1. JSE_OLS_Walk_Forward_Predictions.csv
2. JSE_OLS_Daily_Performance.csv
3. JSE_OLS_Window_Performance.csv
4. JSE_OLS_Overall_Performance.csv
5. outputs/audits/JSE_OLS_Coefficient_Estimates.csv
6. outputs/audits/JSE_OLS_Winsorisation_Thresholds.csv
7. outputs/audits/JSE_OLS_Estimation_Summary.csv
8. outputs/figures/figure_5_2_ols_coefficient_stability.png
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
# 1. CONFIGURATION
# ============================================================

INPUT_MASTER_PANEL = Path("JSE_Master_Model_Panel.csv")
INPUT_OUTER_WINDOWS = Path("JSE_Walk_Forward_Windows.csv")

ROOT_FEATURE_CONFIGURATION = Path(
    "JSE_Model_Feature_Configuration.csv"
)

AUDIT_FEATURE_CONFIGURATION = (
    Path("outputs")
    / "audits"
    / "JSE_Model_Feature_Configuration.csv"
)

OUTPUT_PREDICTIONS = Path(
    "JSE_OLS_Walk_Forward_Predictions.csv"
)

OUTPUT_DAILY_PERFORMANCE = Path(
    "JSE_OLS_Daily_Performance.csv"
)

OUTPUT_WINDOW_PERFORMANCE = Path(
    "JSE_OLS_Window_Performance.csv"
)

OUTPUT_OVERALL_PERFORMANCE = Path(
    "JSE_OLS_Overall_Performance.csv"
)

AUDIT_DIRECTORY = Path("outputs") / "audits"
FIGURE_DIRECTORY = Path("outputs") / "figures"

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
NORMAL_CRITICAL_VALUE = 1.959963984540054


# ============================================================
# 2. VALIDATION AND UTILITY FUNCTIONS
# ============================================================


def locate_feature_configuration() -> Path:
    """Locate the feature-configuration file created by Script 06."""

    if ROOT_FEATURE_CONFIGURATION.exists():
        return ROOT_FEATURE_CONFIGURATION

    if AUDIT_FEATURE_CONFIGURATION.exists():
        return AUDIT_FEATURE_CONFIGURATION

    raise FileNotFoundError(
        "JSE_Model_Feature_Configuration.csv was not found in the "
        "project root or outputs/audits."
    )


def validate_required_files(
    feature_configuration_path: Path,
) -> None:
    """Confirm that all required Script 06 outputs exist."""

    required_files = [
        INPUT_MASTER_PANEL,
        INPUT_OUTER_WINDOWS,
        feature_configuration_path,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required input files are missing:\n- "
            + "\n- ".join(missing_files)
        )


def validate_master_panel(
    panel: pd.DataFrame,
) -> None:
    """Validate the common modelling panel."""

    required_columns = {
        "Date",
        "Ticker",
        "Forward_Return",
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
            ["Date", "Ticker"]
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
        TARGET_COLUMN,
        *PRIMARY_FEATURES,
    ]

    missing_values = int(
        panel[required_values]
        .isna()
        .sum()
        .sum()
    )

    if missing_values > 0:
        raise ValueError(
            f"The master panel contains {missing_values:,} "
            "missing required model values."
        )

    feature_array = panel[
        PRIMARY_FEATURES
    ].to_numpy(dtype=float)

    if not np.isfinite(feature_array).all():
        raise ValueError(
            "The master panel contains non-finite predictor values."
        )

    if not panel[TARGET_COLUMN].between(
        0,
        1,
        inclusive="both",
    ).all():
        raise ValueError(
            "Target values outside [0, 1] were detected."
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

    if windows["Outer_Window_ID"].duplicated().any():
        raise ValueError(
            "Duplicate Outer_Window_ID values were detected."
        )

    if not (
        windows["Train_End"]
        < windows["Purged_Boundary_Date"]
    ).all():
        raise ValueError(
            "At least one training window reaches its purge date."
        )

    if not (
        windows["Purged_Boundary_Date"]
        < windows["Test_Start"]
    ).all():
        raise ValueError(
            "At least one purge date is not earlier than its test start."
        )


def read_winsorisation_configuration(
    configuration_path: Path,
) -> tuple[float, float]:
    """Read and validate the stored training-window clipping quantiles."""

    configuration = pd.read_csv(
        configuration_path
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

    configured_features = configuration.loc[
        configuration["Used_In_Primary_OLS"].eq(True),
        "Feature",
    ].tolist()

    if configured_features != PRIMARY_FEATURES:
        raise ValueError(
            "The primary OLS feature order does not match Script 07.\n"
            f"Configured: {configured_features}\n"
            f"Expected: {PRIMARY_FEATURES}"
        )

    lower_values = configuration[
        "Winsor_Lower_Quantile"
    ].dropna().unique()

    upper_values = configuration[
        "Winsor_Upper_Quantile"
    ].dropna().unique()

    if len(lower_values) != 1 or len(upper_values) != 1:
        raise ValueError(
            "Winsorisation quantiles are not uniquely defined."
        )

    lower_quantile = float(lower_values[0])
    upper_quantile = float(upper_values[0])

    if not (
        0.0
        <= lower_quantile
        < upper_quantile
        <= 1.0
    ):
        raise ValueError(
            "Invalid winsorisation quantiles were supplied."
        )

    return lower_quantile, upper_quantile


def estimate_winsorisation_thresholds(
    training_data: pd.DataFrame,
    features: list[str],
    lower_quantile: float,
    upper_quantile: float,
) -> pd.DataFrame:
    """Estimate feature-level clipping thresholds from training rows only."""

    records: list[dict[str, float | str]] = []

    for feature in features:
        lower_bound = float(
            training_data[feature].quantile(
                lower_quantile
            )
        )

        upper_bound = float(
            training_data[feature].quantile(
                upper_quantile
            )
        )

        if not np.isfinite(lower_bound):
            raise ValueError(
                f"Non-finite lower winsorisation threshold for {feature}."
            )

        if not np.isfinite(upper_bound):
            raise ValueError(
                f"Non-finite upper winsorisation threshold for {feature}."
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

    return pd.DataFrame(records)


def apply_winsorisation(
    data: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    """Apply stored training-window thresholds without re-estimation."""

    transformed = data.copy()

    for row in thresholds.itertuples(
        index=False
    ):
        transformed[row.Feature] = transformed[
            row.Feature
        ].clip(
            lower=row.Lower_Bound,
            upper=row.Upper_Bound,
        )

    return transformed


def calculate_two_way_cluster_inference(
    fitted_model: sm.regression.linear_model.RegressionResultsWrapper,
    training_data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate date-and-ticker two-way cluster-robust inference."""

    date_groups = pd.factorize(
        training_data["Date"],
        sort=True,
    )[0]

    ticker_groups = pd.factorize(
        training_data["Ticker"],
        sort=True,
    )[0]

    try:
        covariance_both, _, _ = cov_cluster_2groups(
            fitted_model,
            date_groups,
            ticker_groups,
            use_correction=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Two-way clustered covariance estimation failed. "
            "Ensure statsmodels supports cov_cluster_2groups."
        ) from exc

    covariance_diagonal = np.diag(
        covariance_both
    )

    # Small negative values can arise from numerical rounding in the
    # two-way covariance combination. Materially negative variances fail.
    if np.any(covariance_diagonal < -1e-12):
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
        out=np.full_like(parameters, np.nan),
        where=standard_errors > 0,
    )

    p_values = 2.0 * norm.sf(
        np.abs(z_statistics)
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


def deterministic_top_bottom_hit_rates(
    group: pd.DataFrame,
) -> tuple[float, float, int]:
    """Calculate daily top- and bottom-quartile overlap rates."""

    number_of_assets = len(group)

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
        ["OLS_Raw_Score", "Ticker"],
        ascending=[False, True],
        kind="mergesort",
    )

    realised_order = group.sort_values(
        [TARGET_COLUMN, "Ticker"],
        ascending=[False, True],
        kind="mergesort",
    )

    predicted_top = set(
        predicted_order.head(
            quartile_size
        )["Ticker"]
    )

    realised_top = set(
        realised_order.head(
            quartile_size
        )["Ticker"]
    )

    predicted_bottom = set(
        predicted_order.tail(
            quartile_size
        )["Ticker"]
    )

    realised_bottom = set(
        realised_order.tail(
            quartile_size
        )["Ticker"]
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
        float(top_hit_rate),
        float(bottom_hit_rate),
        quartile_size,
    )


def safe_daily_spearman(
    group: pd.DataFrame,
    prediction_column: str,
) -> tuple[float, bool]:
    """
    Calculate a daily Spearman Information Coefficient safely.

    A constant prediction vector contains no cross-sectional ranking
    information. It is therefore assigned an IC of zero rather than an
    undefined value. This is the same rule used for XGBoost so that the
    later paired model comparison contains the same evaluation dates.
    """

    if len(group) < MINIMUM_DAILY_CROSS_SECTION:
        return np.nan, False

    prediction_values = group[prediction_column]
    target_values = group[TARGET_COLUMN]

    if (
        prediction_values.nunique(dropna=True) <= 1
        or target_values.nunique(dropna=True) <= 1
    ):
        return 0.0, True

    result = spearmanr(
        prediction_values,
        target_values,
    )

    statistic = float(result.statistic)

    if not np.isfinite(statistic):
        return 0.0, True

    return statistic, False


def calculate_daily_performance(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate date-level OLS predictive diagnostics."""

    records: list[dict[str, object]] = []

    for date, group in predictions.groupby(
        "Date",
        sort=True,
    ):
        number_of_assets = len(group)

        daily_ic, constant_prediction_flag = (
            safe_daily_spearman(
                group=group,
                prediction_column="OLS_Raw_Score",
            )
        )

        prediction_errors = (
            group["OLS_Raw_Score"]
            - group[TARGET_COLUMN]
        )

        daily_mae = float(
            np.mean(
                np.abs(prediction_errors)
            )
        )

        daily_rmse = float(
            np.sqrt(
                np.mean(
                    prediction_errors ** 2
                )
            )
        )

        top_hit_rate, bottom_hit_rate, quartile_size = (
            deterministic_top_bottom_hit_rates(
                group
            )
        )

        records.append(
            {
                "Date": date,
                "Outer_Window_ID": group[
                    "Outer_Window_ID"
                ].iloc[0],
                "Test_Quarter": group[
                    "Test_Quarter"
                ].iloc[0],
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
                    ].lt(0).sum()
                ),
                "Predictions_Above_One": int(
                    group[
                        "OLS_Raw_Score"
                    ].gt(1).sum()
                ),
                "Constant_Prediction_Day": int(
                    constant_prediction_flag
                ),
            }
        )

    return pd.DataFrame(records)


def summarise_performance(
    predictions: pd.DataFrame,
    daily_performance: pd.DataFrame,
    grouping_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Summarise observation-level and daily metrics."""

    if grouping_columns is None:
        grouped_predictions = [
            ("Overall", predictions)
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

    records: list[dict[str, object]] = []

    for key, prediction_group in grouped_predictions:
        daily_group = grouped_daily[key]

        valid_ic = daily_group[
            "Daily_IC"
        ].dropna()

        mean_ic = float(
            valid_ic.mean()
        )

        median_ic = float(
            valid_ic.median()
        )

        ic_standard_deviation = float(
            valid_ic.std(ddof=1)
        )

        ic_information_ratio = (
            mean_ic / ic_standard_deviation
            if ic_standard_deviation > 0
            else np.nan
        )

        errors = (
            prediction_group["OLS_Raw_Score"]
            - prediction_group[TARGET_COLUMN]
        )

        record: dict[str, object] = {
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
                valid_ic.gt(0).sum()
            ),
            "Zero_IC_Days": int(
                valid_ic.eq(0).sum()
            ),
            "Negative_IC_Days": int(
                valid_ic.lt(0).sum()
            ),
            "Positive_IC_Percentage": float(
                valid_ic.gt(0).mean()
                * 100.0
            ),
            "Observation_MAE": float(
                np.mean(
                    np.abs(errors)
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
                daily_group["Daily_MAE"].mean()
            ),
            "Mean_Daily_RMSE": float(
                daily_group["Daily_RMSE"].mean()
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
                ].lt(0).sum()
            ),
            "Predictions_Above_One": int(
                prediction_group[
                    "OLS_Raw_Score"
                ].gt(1).sum()
            ),
            "Constant_Prediction_Days": int(
                daily_group[
                    "Constant_Prediction_Day"
                ].sum()
            ),
        }

        if grouping_columns is None:
            record["Model"] = "Pooled_OLS"
        else:
            if not isinstance(key, tuple):
                key = (key,)

            for column, value in zip(
                grouping_columns,
                key,
            ):
                record[column] = value

        records.append(record)

    output = pd.DataFrame(records)

    if grouping_columns is None:
        ordered_columns = [
            "Model",
            *[
                column
                for column in output.columns
                if column != "Model"
            ],
        ]
        output = output[ordered_columns]

    return output


def create_coefficient_stability_figure(
    coefficient_table: pd.DataFrame,
) -> None:
    """Create a feature-by-quarter coefficient heatmap."""

    plotting_data = coefficient_table.loc[
        coefficient_table["Term"].ne("const")
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

    if maximum_absolute_coefficient == 0:
        maximum_absolute_coefficient = 1.0

    figure_width = max(
        10.0,
        len(coefficient_matrix.columns) * 1.05,
    )

    figure_height = max(
        6.0,
        len(coefficient_matrix.index) * 0.62,
    )

    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height)
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
            len(coefficient_matrix.columns)
        )
    )

    axis.set_yticks(
        np.arange(
            len(coefficient_matrix.index)
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
        coefficient_matrix.shape[0]
    ):
        for column_index in range(
            coefficient_matrix.shape[1]
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

    plt.close(figure)


# ============================================================
# 3. MAIN PIPELINE
# ============================================================


def main() -> None:
    print("=" * 78)
    print("SCRIPT 07: POOLED OLS EXPANDING-WINDOW ESTIMATION")
    print("=" * 78)

    feature_configuration_path = (
        locate_feature_configuration()
    )

    validate_required_files(
        feature_configuration_path
    )

    AUDIT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    lower_quantile, upper_quantile = (
        read_winsorisation_configuration(
            feature_configuration_path
        )
    )

    # --------------------------------------------------------
    # 3.1 Load the common modelling panel and outer schedule
    # --------------------------------------------------------

    master_panel = pd.read_csv(
        INPUT_MASTER_PANEL
    )

    master_panel["Date"] = pd.to_datetime(
        master_panel["Date"],
        errors="raise",
    )

    if "Next_Date" in master_panel.columns:
        master_panel["Next_Date"] = pd.to_datetime(
            master_panel["Next_Date"],
            errors="coerce",
        )

    master_panel = (
        master_panel
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )

    validate_master_panel(
        master_panel
    )

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
        outer_windows[column] = pd.to_datetime(
            outer_windows[column],
            errors="raise",
        )

    outer_windows = (
        outer_windows
        .sort_values("Test_Start")
        .reset_index(drop=True)
    )

    validate_outer_windows(
        outer_windows
    )

    # --------------------------------------------------------
    # 3.2 Estimate every expanding-window OLS model
    # --------------------------------------------------------

    prediction_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    winsorisation_frames: list[pd.DataFrame] = []
    estimation_records: list[dict[str, object]] = []

    for window in outer_windows.itertuples(
        index=False
    ):
        train_mask = master_panel["Date"].between(
            pd.Timestamp(window.Train_Start),
            pd.Timestamp(window.Train_End),
            inclusive="both",
        )

        test_mask = master_panel["Date"].between(
            pd.Timestamp(window.Test_Start),
            pd.Timestamp(window.Test_End),
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

        if len(training_data) != int(
            window.Train_Observations
        ):
            raise AssertionError(
                f"{window.Outer_Window_ID} training-row count does "
                "not match the stored schedule."
            )

        if len(test_data) != int(
            window.Test_Observations
        ):
            raise AssertionError(
                f"{window.Outer_Window_ID} test-row count does "
                "not match the stored schedule."
            )

        if training_data["Date"].max() >= test_data["Date"].min():
            raise AssertionError(
                f"{window.Outer_Window_ID} training and test dates overlap."
            )

        if "Next_Date" in training_data.columns:
            leakage_count = int(
                (
                    training_data["Next_Date"]
                    >= pd.Timestamp(window.Test_Start)
                ).sum()
            )

            if leakage_count > 0:
                raise AssertionError(
                    f"{window.Outer_Window_ID} contains "
                    f"{leakage_count:,} target-boundary leakage rows."
                )

        thresholds = estimate_winsorisation_thresholds(
            training_data=training_data,
            features=PRIMARY_FEATURES,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )

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

        winsorisation_frames.append(
            thresholds
        )

        threshold_values = thresholds.drop(
            columns=[
                "Outer_Window_ID",
                "Test_Quarter",
            ]
        )

        training_transformed = apply_winsorisation(
            data=training_data,
            thresholds=threshold_values,
        )

        test_transformed = apply_winsorisation(
            data=test_data,
            thresholds=threshold_values,
        )

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
            y_train,
            x_train,
            missing="raise",
        ).fit()

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
            pd.Timestamp(window.Train_Start),
        )

        inference.insert(
            3,
            "Train_End",
            pd.Timestamp(window.Train_End),
        )

        coefficient_frames.append(
            inference
        )

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
            "Universe_Selection_Quarter",
            "Liquidity_Rank",
            "Membership_Status",
        ]

        prediction_columns.extend(
            [
                column
                for column in optional_columns
                if column in test_data.columns
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

        window_predictions["OLS_Raw_Score"] = (
            raw_predictions.to_numpy(
                dtype=float
            )
        )

        window_predictions[
            "OLS_Predicted_Rank"
        ] = (
            window_predictions
            .groupby("Date")["OLS_Raw_Score"]
            .rank(
                method="average",
                ascending=True,
            )
        )

        window_predictions[
            "OLS_Prediction_N_t"
        ] = (
            window_predictions
            .groupby("Date")["Ticker"]
            .transform("count")
            .astype(int)
        )

        window_predictions[
            "OLS_Predicted_Percentile"
        ] = (
            window_predictions[
                "OLS_Predicted_Rank"
            ]
            - 1.0
        ) / (
            window_predictions[
                "OLS_Prediction_N_t"
            ]
            - 1.0
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
                "Train_Start": pd.Timestamp(
                    window.Train_Start
                ),
                "Train_End": pd.Timestamp(
                    window.Train_End
                ),
                "Test_Start": pd.Timestamp(
                    window.Test_Start
                ),
                "Test_End": pd.Timestamp(
                    window.Test_End
                ),
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
            }
        )

        print(
            f"Completed {window.Outer_Window_ID} | "
            f"{window.Test_Quarter} | "
            f"train rows {len(training_data):,} | "
            f"test rows {len(test_data):,}"
        )

    # --------------------------------------------------------
    # 3.3 Combine predictions and verify out-of-sample coverage
    # --------------------------------------------------------

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    predictions = (
        predictions
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )

    if predictions.duplicated(
        ["Date", "Ticker"]
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

    if len(predictions) != expected_test_observations:
        raise AssertionError(
            "The OLS prediction file does not cover every stored "
            "out-of-sample observation."
        )

    if predictions["Date"].nunique() != expected_test_dates:
        raise AssertionError(
            "The OLS prediction file does not cover every stored "
            "out-of-sample date."
        )

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

    # --------------------------------------------------------
    # 3.4 Save all accepted OLS outputs
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
    # 3.5 Terminal results
    # --------------------------------------------------------

    overall = overall_performance.iloc[0]

    print("\n" + "=" * 78)
    print("SCRIPT 07 EMPIRICAL OUTPUTS")
    print("=" * 78)

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
        "Constant-prediction trading days: "
        f"{int(overall['Constant_Prediction_Days']):,}"
    )

    print("\nIntegrity checks passed:")
    print(
        "- Every test observation was predicted exactly once."
    )
    print(
        "- Winsorisation thresholds were estimated separately "
        "inside each expanding training window."
    )
    print(
        "- Test predictors were clipped only to the corresponding "
        "training-window bounds."
    )
    print(
        "- Every OLS model used the same nine prespecified predictors."
    )
    print(
        "- Coefficient inference uses two-way clustering by date "
        "and ticker."
    )
    print(
        "- Daily ranks and percentiles were formed only within each "
        "out-of-sample cross-section."
    )
    print(
        "- Constant daily prediction vectors were assigned an IC of "
        "zero, matching the XGBoost evaluation rule."
    )

    print("\nCore outputs:")
    print(f"- {OUTPUT_PREDICTIONS}")
    print(f"- {OUTPUT_DAILY_PERFORMANCE}")
    print(f"- {OUTPUT_WINDOW_PERFORMANCE}")
    print(f"- {OUTPUT_OVERALL_PERFORMANCE}")

    print("\nAudit and figure outputs:")
    print(f"- {OUTPUT_COEFFICIENTS}")
    print(f"- {OUTPUT_WINSORISATION}")
    print(f"- {OUTPUT_ESTIMATION_SUMMARY}")
    print(f"- {OUTPUT_FIGURE}")

    print("=" * 78)


if __name__ == "__main__":
    main()