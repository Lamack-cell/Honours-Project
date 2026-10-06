"""
Script 05B: Final Machine-Learning Panel Diagnostics
----------------------------------------------------

Purpose
-------
Performs structural, distributional and multicollinearity diagnostics on
the final feature-engineered modelling panel.

The script does not modify the modelling data. It assesses:

1. stock-date uniqueness;
2. missing and infinite values;
3. daily cross-sectional coverage;
4. observations excluded by feature-history requirements;
5. standardisation accuracy;
6. extreme standardised observations;
7. Pearson and Spearman correlation;
8. Variance Inflation Factors;
9. condition diagnostics; and
10. annual panel coverage.

Inputs
------
data/processed/05_features/JSE_Panel_Features_Engineered.csv
data/interim/03_target/JSE_Panel_Target_Engineered.csv

Audit outputs
-------------
outputs/audits/05_features/JSE_ML_Panel_Diagnostics_Summary.csv
outputs/audits/05_features/JSE_ML_Panel_Daily_Cross_Section.csv
outputs/audits/05_features/JSE_Model_Eligibility_Exclusions.csv
outputs/audits/05_features/JSE_Feature_VIF_Diagnostics.csv
outputs/audits/05_features/JSE_High_Correlation_Pairs.csv
outputs/audits/05_features/JSE_Feature_Target_Correlations.csv
outputs/audits/05_features/JSE_Feature_Extreme_Value_Diagnostics.csv
outputs/audits/05_features/JSE_Annual_Model_Panel_Coverage.csv
outputs/audits/05_features/JSE_Feature_Condition_Diagnostics.csv

Figure output
-------------
outputs/figures/chapter_4/figure_4_5_model_cross_section_size.png

Run from the project root
-------------------------
python .\\scripts\\05B_ML_Panel_Diagnostics.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FEATURE_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "05_features"
    / "JSE_Panel_Features_Engineered.csv"
)

INPUT_TARGET_PANEL = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "03_target"
    / "JSE_Panel_Target_Engineered.csv"
)

AUDIT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "audits"
    / "05_features"
)

FIGURE_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "chapter_4"
)


# ============================================================
# 2. OUTPUT FILES
# ============================================================

OUTPUT_SUMMARY = (
    AUDIT_DIRECTORY
    / "JSE_ML_Panel_Diagnostics_Summary.csv"
)

OUTPUT_DAILY_COUNTS = (
    AUDIT_DIRECTORY
    / "JSE_ML_Panel_Daily_Cross_Section.csv"
)

OUTPUT_EXCLUSIONS = (
    AUDIT_DIRECTORY
    / "JSE_Model_Eligibility_Exclusions.csv"
)

OUTPUT_VIF = (
    AUDIT_DIRECTORY
    / "JSE_Feature_VIF_Diagnostics.csv"
)

OUTPUT_HIGH_CORRELATIONS = (
    AUDIT_DIRECTORY
    / "JSE_High_Correlation_Pairs.csv"
)

OUTPUT_TARGET_CORRELATIONS = (
    AUDIT_DIRECTORY
    / "JSE_Feature_Target_Correlations.csv"
)

OUTPUT_EXTREME_VALUES = (
    AUDIT_DIRECTORY
    / "JSE_Feature_Extreme_Value_Diagnostics.csv"
)

OUTPUT_ANNUAL_COVERAGE = (
    AUDIT_DIRECTORY
    / "JSE_Annual_Model_Panel_Coverage.csv"
)

OUTPUT_CONDITION_DIAGNOSTICS = (
    AUDIT_DIRECTORY
    / "JSE_Feature_Condition_Diagnostics.csv"
)

OUTPUT_FIGURE = (
    FIGURE_DIRECTORY
    / "figure_4_5_model_cross_section_size.png"
)


# ============================================================
# 3. CONFIGURATION
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

TARGET = "Target_Y"

HIGH_CORRELATION_THRESHOLD = 0.80


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def relative_path(path: Path) -> str:
    """Return a readable path relative to the project root."""

    return str(
        path.relative_to(
            PROJECT_ROOT
        )
    )


def validate_files() -> None:
    """Confirm that all required files exist."""

    required_files = [
        INPUT_FEATURE_PANEL,
        INPUT_TARGET_PANEL,
    ]

    missing_files = [
        str(path.resolve())
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


def validate_panel(
    panel: pd.DataFrame,
    required_columns: set[str],
    panel_name: str,
) -> None:
    """Run structural validation on a panel."""

    missing_columns = required_columns.difference(
        panel.columns
    )

    if missing_columns:
        raise ValueError(
            f"{panel_name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if panel.empty:
        raise ValueError(
            f"{panel_name} contains no observations."
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
            f"{panel_name} contains {duplicate_count:,} duplicate "
            "Date-Ticker records."
        )

    if panel[
        [
            "Date",
            "Ticker",
        ]
    ].isna().any().any():
        raise ValueError(
            f"{panel_name} contains missing Date or Ticker values."
        )


def calculate_vif(
    feature_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factors using auxiliary OLS regressions.

    VIF_j = 1 / (1 - R_j^2)
    """

    matrix = feature_matrix.to_numpy(
        dtype=float
    )

    records: list[dict[str, object]] = []

    for feature_index, feature_name in enumerate(
        feature_matrix.columns
    ):
        dependent = matrix[
            :,
            feature_index,
        ]

        explanatory = np.delete(
            matrix,
            feature_index,
            axis=1,
        )

        explanatory = np.column_stack(
            [
                np.ones(
                    len(
                        explanatory
                    )
                ),
                explanatory,
            ]
        )

        coefficients, _, _, _ = np.linalg.lstsq(
            explanatory,
            dependent,
            rcond=None,
        )

        fitted = (
            explanatory
            @ coefficients
        )

        residual_sum_squares = float(
            np.sum(
                (
                    dependent
                    - fitted
                )
                ** 2
            )
        )

        total_sum_squares = float(
            np.sum(
                (
                    dependent
                    - dependent.mean()
                )
                ** 2
            )
        )

        if total_sum_squares <= 0:
            r_squared = np.nan
            vif = np.inf

        else:
            r_squared = (
                1.0
                - residual_sum_squares
                / total_sum_squares
            )

            r_squared = min(
                max(
                    r_squared,
                    0.0,
                ),
                1.0,
            )

            vif = (
                np.inf
                if np.isclose(
                    1.0
                    - r_squared,
                    0.0,
                )
                else (
                    1.0
                    / (
                        1.0
                        - r_squared
                    )
                )
            )

        tolerance = (
            0.0
            if not np.isfinite(
                vif
            )
            else (
                1.0
                / vif
            )
        )

        if not np.isfinite(
            vif
        ):
            severity = (
                "Perfect collinearity"
            )

        elif vif >= 10:
            severity = "High"

        elif vif >= 5:
            severity = "Moderate"

        else:
            severity = "Low"

        records.append(
            {
                "Feature": feature_name,
                "Auxiliary_R_Squared": r_squared,
                "Tolerance": tolerance,
                "VIF": vif,
                "Diagnostic_Category": severity,
            }
        )

    return (
        pd.DataFrame(
            records
        )
        .sort_values(
            "VIF",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def extract_high_correlation_pairs(
    correlation_matrix: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Extract unique feature pairs exceeding an absolute threshold."""

    records: list[dict[str, object]] = []

    columns = correlation_matrix.columns.tolist()

    for row_index in range(
        len(
            columns
        )
    ):
        for column_index in range(
            row_index + 1,
            len(
                columns
            ),
        ):
            correlation = float(
                correlation_matrix.iloc[
                    row_index,
                    column_index,
                ]
            )

            if abs(
                correlation
            ) >= threshold:
                records.append(
                    {
                        "Feature_1": columns[
                            row_index
                        ],
                        "Feature_2": columns[
                            column_index
                        ],
                        "Spearman_Correlation": correlation,
                        "Absolute_Correlation": abs(
                            correlation
                        ),
                    }
                )

    if not records:
        return pd.DataFrame(
            columns=[
                "Feature_1",
                "Feature_2",
                "Spearman_Correlation",
                "Absolute_Correlation",
            ]
        )

    return (
        pd.DataFrame(
            records
        )
        .sort_values(
            "Absolute_Correlation",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

def main() -> None:
    print("=" * 74)
    print("SCRIPT 05B: FINAL ML PANEL DIAGNOSTICS")
    print("=" * 74)

    validate_files()

    AUDIT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 5.1 Load and validate panels
    # --------------------------------------------------------

    feature_panel = pd.read_csv(
        INPUT_FEATURE_PANEL
    )

    target_panel = pd.read_csv(
        INPUT_TARGET_PANEL
    )

    feature_panel["Date"] = pd.to_datetime(
        feature_panel["Date"],
        errors="raise",
    )

    target_panel["Date"] = pd.to_datetime(
        target_panel["Date"],
        errors="raise",
    )

    validate_panel(
        panel=feature_panel,
        required_columns={
            "Date",
            "Ticker",
            TARGET,
            "N_t",
            "Model_N_t",
            *FEATURES,
        },
        panel_name="Feature panel",
    )

    validate_panel(
        panel=target_panel,
        required_columns={
            "Date",
            "Ticker",
            TARGET,
            "N_t",
        },
        panel_name="Target panel",
    )

    feature_panel = (
        feature_panel
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

    target_panel = (
        target_panel
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

    # --------------------------------------------------------
    # 5.2 Missingness and finite-value checks
    # --------------------------------------------------------

    feature_values = feature_panel[
        FEATURES
    ].to_numpy(
        dtype=float
    )

    missing_feature_values = int(
        feature_panel[
            FEATURES
        ]
        .isna()
        .sum()
        .sum()
    )

    infinite_feature_values = int(
        np.isinf(
            feature_values
        ).sum()
    )

    if missing_feature_values > 0:
        raise ValueError(
            f"{missing_feature_values:,} missing feature values "
            "were detected."
        )

    if infinite_feature_values > 0:
        raise ValueError(
            f"{infinite_feature_values:,} infinite feature values "
            "were detected."
        )

    if feature_panel[
        TARGET
    ].isna().any():
        raise ValueError(
            "The model target contains missing values."
        )

    if not feature_panel[
        TARGET
    ].between(
        0,
        1,
        inclusive="both",
    ).all():
        raise ValueError(
            "Target values outside [0, 1] were detected."
        )

    if not feature_panel[
        "N_t"
    ].eq(
        40
    ).all():
        raise ValueError(
            "The stored target-universe size N_t is not always 40."
        )

    # --------------------------------------------------------
    # 5.3 Model eligibility and excluded observations
    # --------------------------------------------------------

    feature_keys = feature_panel[
        [
            "Date",
            "Ticker",
        ]
    ].copy()

    exclusions = target_panel.merge(
        feature_keys,
        on=[
            "Date",
            "Ticker",
        ],
        how="left",
        indicator=True,
        validate="one_to_one",
    )

    exclusions = (
        exclusions.loc[
            exclusions[
                "_merge"
            ].eq(
                "left_only"
            )
        ]
        .drop(
            columns="_merge"
        )
        .copy()
    )

    exclusions[
        "Exclusion_Reason"
    ] = (
        "Incomplete rolling feature history"
    )

    first_model_date = feature_panel[
        "Date"
    ].min()

    exclusions[
        "Exclusion_Category"
    ] = np.where(
        exclusions[
            "Date"
        ]
        < first_model_date,
        "Initial feature warm-up",
        "Security-specific insufficient history",
    )

    exclusions.to_csv(
        OUTPUT_EXCLUSIONS,
        index=False,
    )

    # --------------------------------------------------------
    # 5.4 Daily cross-sectional availability
    # --------------------------------------------------------

    target_daily_counts = (
        target_panel
        .groupby(
            "Date"
        )[
            "Ticker"
        ]
        .nunique()
        .rename(
            "Target_Universe_Size"
        )
    )

    model_daily_counts = (
        feature_panel
        .groupby(
            "Date"
        )[
            "Ticker"
        ]
        .nunique()
        .rename(
            "Model_Eligible_Size"
        )
    )

    daily_counts = (
        pd.concat(
            [
                target_daily_counts,
                model_daily_counts,
            ],
            axis=1,
        )
        .fillna(
            0
        )
        .reset_index()
        .sort_values(
            "Date"
        )
        .reset_index(
            drop=True
        )
    )

    daily_counts[
        "Target_Universe_Size"
    ] = daily_counts[
        "Target_Universe_Size"
    ].astype(
        int
    )

    daily_counts[
        "Model_Eligible_Size"
    ] = daily_counts[
        "Model_Eligible_Size"
    ].astype(
        int
    )

    daily_counts[
        "Excluded_Securities"
    ] = (
        daily_counts[
            "Target_Universe_Size"
        ]
        - daily_counts[
            "Model_Eligible_Size"
        ]
    )

    daily_counts[
        "Model_Eligibility_Rate"
    ] = np.where(
        daily_counts[
            "Target_Universe_Size"
        ].gt(
            0
        ),
        (
            daily_counts[
                "Model_Eligible_Size"
            ]
            / daily_counts[
                "Target_Universe_Size"
            ]
        ),
        np.nan,
    )

    daily_counts.to_csv(
        OUTPUT_DAILY_COUNTS,
        index=False,
    )

    model_period_counts = (
        daily_counts.loc[
            daily_counts[
                "Date"
            ]
            >= first_model_date
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if model_period_counts.empty:
        raise ValueError(
            "No daily model-period counts were constructed."
        )

    # Confirm the stored row-level Model_N_t equals the observed count.
    observed_model_size = (
        feature_panel
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

    if not feature_panel[
        "Model_N_t"
    ].astype(
        int
    ).eq(
        observed_model_size
    ).all():
        raise AssertionError(
            "Stored Model_N_t does not match the observed daily "
            "model cross-sectional size."
        )

    # --------------------------------------------------------
    # 5.5 Standardisation and extreme-value diagnostics
    # --------------------------------------------------------

    extreme_records: list[dict[str, object]] = []

    for feature in FEATURES:
        daily_statistics = (
            feature_panel
            .groupby(
                "Date"
            )[
                feature
            ]
            .agg(
                [
                    "mean",
                    "std",
                ]
            )
        )

        absolute_values = feature_panel[
            feature
        ].abs()

        extreme_records.append(
            {
                "Feature": feature,
                "Minimum": float(
                    feature_panel[
                        feature
                    ].min()
                ),
                "Maximum": float(
                    feature_panel[
                        feature
                    ].max()
                ),
                "Global_Mean": float(
                    feature_panel[
                        feature
                    ].mean()
                ),
                "Global_Standard_Deviation": float(
                    feature_panel[
                        feature
                    ].std()
                ),
                "Maximum_Absolute_Daily_Mean": float(
                    daily_statistics[
                        "mean"
                    ]
                    .abs()
                    .max()
                ),
                "Minimum_Daily_Standard_Deviation": float(
                    daily_statistics[
                        "std"
                    ].min()
                ),
                "Median_Daily_Standard_Deviation": float(
                    daily_statistics[
                        "std"
                    ].median()
                ),
                "Maximum_Daily_Standard_Deviation": float(
                    daily_statistics[
                        "std"
                    ].max()
                ),
                "Observations_Absolute_Z_Above_3": int(
                    absolute_values.gt(
                        3
                    ).sum()
                ),
                "Observations_Absolute_Z_Above_4": int(
                    absolute_values.gt(
                        4
                    ).sum()
                ),
            }
        )

    extreme_diagnostics = pd.DataFrame(
        extreme_records
    )

    extreme_diagnostics.to_csv(
        OUTPUT_EXTREME_VALUES,
        index=False,
    )

    # --------------------------------------------------------
    # 5.6 Correlation diagnostics
    # --------------------------------------------------------

    spearman_matrix = feature_panel[
        FEATURES
    ].corr(
        method="spearman"
    )

    pearson_matrix = feature_panel[
        FEATURES
    ].corr(
        method="pearson"
    )

    high_correlations = extract_high_correlation_pairs(
        correlation_matrix=spearman_matrix,
        threshold=HIGH_CORRELATION_THRESHOLD,
    )

    high_correlations.to_csv(
        OUTPUT_HIGH_CORRELATIONS,
        index=False,
    )

    target_correlation_records: list[
        dict[str, object]
    ] = []

    for feature in FEATURES:
        spearman_target_correlation = (
            feature_panel[
                [
                    feature,
                    TARGET,
                ]
            ]
            .corr(
                method="spearman"
            )
            .iloc[
                0,
                1,
            ]
        )

        pearson_target_correlation = (
            feature_panel[
                [
                    feature,
                    TARGET,
                ]
            ]
            .corr(
                method="pearson"
            )
            .iloc[
                0,
                1,
            ]
        )

        target_correlation_records.append(
            {
                "Feature": feature,
                "Spearman_Correlation_With_Target": (
                    spearman_target_correlation
                ),
                "Pearson_Correlation_With_Target": (
                    pearson_target_correlation
                ),
            }
        )

    target_correlations = pd.DataFrame(
        target_correlation_records
    )

    target_correlations.to_csv(
        OUTPUT_TARGET_CORRELATIONS,
        index=False,
    )

    # --------------------------------------------------------
    # 5.7 Variance Inflation Factors
    # --------------------------------------------------------

    vif_diagnostics = calculate_vif(
        feature_panel[
            FEATURES
        ]
    )

    vif_diagnostics.to_csv(
        OUTPUT_VIF,
        index=False,
    )

    # --------------------------------------------------------
    # 5.8 Condition diagnostics
    # --------------------------------------------------------

    correlation_array = pearson_matrix.to_numpy(
        dtype=float
    )

    eigenvalues = np.linalg.eigvalsh(
        correlation_array
    )

    largest_eigenvalue = float(
        eigenvalues.max()
    )

    smallest_eigenvalue = float(
        eigenvalues.min()
    )

    correlation_matrix_rank = int(
        np.linalg.matrix_rank(
            correlation_array
        )
    )

    correlation_condition_number = float(
        np.linalg.cond(
            correlation_array
        )
    )

    maximum_condition_index = (
        np.inf
        if smallest_eigenvalue <= 0
        else float(
            np.sqrt(
                largest_eigenvalue
                / smallest_eigenvalue
            )
        )
    )

    condition_diagnostics = pd.DataFrame(
        {
            "Statistic": [
                "Number of predictors",
                "Predictor correlation-matrix rank",
                "Largest correlation eigenvalue",
                "Smallest correlation eigenvalue",
                "Correlation-matrix condition number",
                "Maximum condition index",
            ],
            "Value": [
                len(
                    FEATURES
                ),
                correlation_matrix_rank,
                largest_eigenvalue,
                smallest_eigenvalue,
                correlation_condition_number,
                maximum_condition_index,
            ],
        }
    )

    condition_diagnostics.to_csv(
        OUTPUT_CONDITION_DIAGNOSTICS,
        index=False,
    )

    # --------------------------------------------------------
    # 5.9 Annual panel coverage
    # --------------------------------------------------------

    annual_source = feature_panel.copy()

    annual_source[
        "Year"
    ] = annual_source[
        "Date"
    ].dt.year

    annual_coverage = (
        annual_source
        .groupby(
            "Year",
            as_index=False,
        )
        .agg(
            Stock_Day_Observations=(
                "Ticker",
                "size",
            ),
            Trading_Dates=(
                "Date",
                "nunique",
            ),
            Unique_Securities=(
                "Ticker",
                "nunique",
            ),
        )
    )

    annual_daily_counts = (
        annual_source
        .groupby(
            [
                "Year",
                "Date",
            ]
        )[
            "Ticker"
        ]
        .nunique()
        .reset_index(
            name="Daily_Model_Size"
        )
    )

    annual_daily_summary = (
        annual_daily_counts
        .groupby(
            "Year",
            as_index=False,
        )
        .agg(
            Minimum_Daily_Model_Size=(
                "Daily_Model_Size",
                "min",
            ),
            Mean_Daily_Model_Size=(
                "Daily_Model_Size",
                "mean",
            ),
            Maximum_Daily_Model_Size=(
                "Daily_Model_Size",
                "max",
            ),
        )
    )

    annual_coverage = annual_coverage.merge(
        annual_daily_summary,
        on="Year",
        how="left",
        validate="one_to_one",
    )

    annual_coverage.to_csv(
        OUTPUT_ANNUAL_COVERAGE,
        index=False,
    )

    # --------------------------------------------------------
    # 5.10 Model cross-sectional-size figure
    # --------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(
            11,
            5,
        )
    )

    axis.plot(
        model_period_counts[
            "Date"
        ],
        model_period_counts[
            "Model_Eligible_Size"
        ],
        linewidth=1.2,
        label="Model-eligible securities",
    )

    axis.axhline(
        40,
        linestyle="--",
        linewidth=1.2,
        label="Target-universe size",
    )

    axis.set_title(
        "Daily Size of the Model-Eligible Cross-Section"
    )

    axis.set_xlabel(
        "Trading date"
    )

    axis.set_ylabel(
        "Number of securities"
    )

    axis.set_ylim(
        max(
            0,
            int(
                model_period_counts[
                    "Model_Eligible_Size"
                ].min()
            )
            - 1,
        ),
        41,
    )

    axis.grid(
        axis="both",
        linestyle="--",
        alpha=0.4,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        OUTPUT_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------
    # 5.11 Final summary
    # --------------------------------------------------------

    model_count_frequency = (
        model_period_counts[
            "Model_Eligible_Size"
        ]
        .value_counts()
        .sort_index()
    )

    initial_warmup_exclusions = int(
        exclusions[
            "Exclusion_Category"
        ]
        .eq(
            "Initial feature warm-up"
        )
        .sum()
    )

    later_history_exclusions = int(
        exclusions[
            "Exclusion_Category"
        ]
        .eq(
            "Security-specific insufficient history"
        )
        .sum()
    )

    minimum_model_cross_section = int(
        model_period_counts[
            "Model_Eligible_Size"
        ].min()
    )

    # Correct calculation: average one count per trading date.
    mean_model_cross_section = float(
        model_period_counts[
            "Model_Eligible_Size"
        ].mean()
    )

    maximum_model_cross_section = int(
        model_period_counts[
            "Model_Eligible_Size"
        ].max()
    )

    features_vif_at_least_5 = int(
        vif_diagnostics[
            "VIF"
        ]
        .ge(
            5
        )
        .sum()
    )

    features_vif_at_least_10 = int(
        vif_diagnostics[
            "VIF"
        ]
        .ge(
            10
        )
        .sum()
    )

    summary = pd.DataFrame(
        {
            "Statistic": [
                "Target observations",
                "Model observations",
                "Excluded observations",
                "Initial feature-warm-up exclusions",
                "Later security-specific exclusions",
                "Model-panel start date",
                "Model-panel end date",
                "Model-panel trading dates",
                "Minimum model cross-section",
                "Mean model cross-section",
                "Maximum model cross-section",
                "Duplicate stock-date records",
                "Missing feature values",
                "Infinite feature values",
                "Features with VIF at least 5",
                "Features with VIF at least 10",
                "High-correlation feature pairs",
                "Correlation-matrix condition number",
                "Maximum condition index",
            ],
            "Value": [
                len(
                    target_panel
                ),
                len(
                    feature_panel
                ),
                len(
                    exclusions
                ),
                initial_warmup_exclusions,
                later_history_exclusions,
                feature_panel[
                    "Date"
                ].min().date(),
                feature_panel[
                    "Date"
                ].max().date(),
                feature_panel[
                    "Date"
                ].nunique(),
                minimum_model_cross_section,
                mean_model_cross_section,
                maximum_model_cross_section,
                int(
                    feature_panel.duplicated(
                        [
                            "Date",
                            "Ticker",
                        ]
                    ).sum()
                ),
                missing_feature_values,
                infinite_feature_values,
                features_vif_at_least_5,
                features_vif_at_least_10,
                len(
                    high_correlations
                ),
                correlation_condition_number,
                maximum_condition_index,
            ],
        }
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # --------------------------------------------------------
    # 5.12 Terminal output
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 74
    )

    print(
        "SCRIPT 05B EMPIRICAL OUTPUTS"
    )

    print(
        "=" * 74
    )

    print(
        f"Target-panel observations: "
        f"{len(target_panel):,}"
    )

    print(
        f"Model-panel observations: "
        f"{len(feature_panel):,}"
    )

    print(
        f"Excluded observations: "
        f"{len(exclusions):,}"
    )

    print(
        "Initial warm-up exclusions: "
        f"{initial_warmup_exclusions:,}"
    )

    print(
        "Later security-specific history exclusions: "
        f"{later_history_exclusions:,}"
    )

    print(
        "Model-panel date range: "
        f"{feature_panel['Date'].min().date()} to "
        f"{feature_panel['Date'].max().date()}"
    )

    print(
        "Model cross-section "
        "(minimum/mean/maximum): "
        f"{minimum_model_cross_section:,}/"
        f"{mean_model_cross_section:.2f}/"
        f"{maximum_model_cross_section:,}"
    )

    print(
        "\nTrading dates by model cross-sectional size:"
    )

    for count, frequency in model_count_frequency.items():
        print(
            f"- {count} securities: "
            f"{frequency:,} dates"
        )

    print(
        "\nMaximum VIF: "
        f"{vif_diagnostics['VIF'].max():.3f}"
    )

    print(
        "Features with VIF >= 5: "
        f"{features_vif_at_least_5:,}"
    )

    print(
        "Features with VIF >= 10: "
        f"{features_vif_at_least_10:,}"
    )

    print(
        "Spearman feature pairs with "
        f"|correlation| >= "
        f"{HIGH_CORRELATION_THRESHOLD:.2f}: "
        f"{len(high_correlations):,}"
    )

    print(
        "Correlation-matrix condition number: "
        f"{correlation_condition_number:.3f}"
    )

    print(
        "Maximum condition index: "
        f"{maximum_condition_index:.3f}"
    )

    print(
        "\nIntegrity checks passed:"
    )

    print(
        "- No duplicate stock-date observations."
    )

    print(
        "- No missing or infinite model features."
    )

    print(
        "- The original target-universe size remains 40."
    )

    print(
        "- Stored Model_N_t matches the observed daily "
        "model cross-sectional size."
    )

    print(
        "- Model eligibility losses are explicitly identified "
        "and preserved in an audit file."
    )

    print(
        "- The reported mean model cross-section is the "
        "unweighted mean across trading dates."
    )

    print(
        "- VIF and condition diagnostics were calculated "
        "without modifying the feature panel."
    )

    print(
        "\nOutput files:"
    )

    for output_file in [
        OUTPUT_SUMMARY,
        OUTPUT_DAILY_COUNTS,
        OUTPUT_EXCLUSIONS,
        OUTPUT_VIF,
        OUTPUT_HIGH_CORRELATIONS,
        OUTPUT_TARGET_CORRELATIONS,
        OUTPUT_EXTREME_VALUES,
        OUTPUT_ANNUAL_COVERAGE,
        OUTPUT_CONDITION_DIAGNOSTICS,
        OUTPUT_FIGURE,
    ]:
        print(
            f"- {relative_path(output_file)}"
        )

    print(
        "=" * 74
    )


if __name__ == "__main__":
    main()