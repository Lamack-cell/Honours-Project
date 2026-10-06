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
1. JSE_Panel_Features_Engineered.csv
2. JSE_Panel_Target_Engineered.csv

Outputs
-------
outputs/audits/JSE_ML_Panel_Diagnostics_Summary.csv
outputs/audits/JSE_ML_Panel_Daily_Cross_Section.csv
outputs/audits/JSE_Model_Eligibility_Exclusions.csv
outputs/audits/JSE_Feature_VIF_Diagnostics.csv
outputs/audits/JSE_High_Correlation_Pairs.csv
outputs/audits/JSE_Feature_Target_Correlations.csv
outputs/audits/JSE_Feature_Extreme_Value_Diagnostics.csv
outputs/audits/JSE_Annual_Model_Panel_Coverage.csv
outputs/audits/JSE_Feature_Condition_Diagnostics.csv
outputs/figures/figure_4_5_model_cross_section_size.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FEATURE_PANEL = Path(
    "JSE_Panel_Features_Engineered.csv"
)

INPUT_TARGET_PANEL = Path(
    "JSE_Panel_Target_Engineered.csv"
)

AUDIT_DIRECTORY = Path("outputs") / "audits"
FIGURE_DIRECTORY = Path("outputs") / "figures"

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
# 2. HELPER FUNCTIONS
# ============================================================

def validate_files() -> None:
    """Confirm that all required files exist."""

    required_files = [
        INPUT_FEATURE_PANEL,
        INPUT_TARGET_PANEL,
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


def validate_panel(
    panel: pd.DataFrame,
    required_columns: set[str],
    panel_name: str,
) -> None:
    """Run basic structural validation."""

    missing_columns = required_columns.difference(
        panel.columns
    )

    if missing_columns:
        raise ValueError(
            f"{panel_name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if panel.duplicated(
        ["Date", "Ticker"]
    ).any():
        raise ValueError(
            f"{panel_name} contains duplicate Date-Ticker records."
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
        dependent = matrix[:, feature_index]

        explanatory = np.delete(
            matrix,
            feature_index,
            axis=1,
        )

        explanatory = np.column_stack(
            [
                np.ones(len(explanatory)),
                explanatory,
            ]
        )

        coefficients, _, _, _ = np.linalg.lstsq(
            explanatory,
            dependent,
            rcond=None,
        )

        fitted = explanatory @ coefficients

        residual_sum_squares = np.sum(
            (dependent - fitted) ** 2
        )

        total_sum_squares = np.sum(
            (
                dependent
                - dependent.mean()
            ) ** 2
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
                max(r_squared, 0.0),
                1.0,
            )

            vif = (
                np.inf
                if np.isclose(
                    1.0 - r_squared,
                    0.0,
                )
                else 1.0 / (1.0 - r_squared)
            )

        tolerance = (
            0.0
            if not np.isfinite(vif)
            else 1.0 / vif
        )

        if not np.isfinite(vif):
            severity = "Perfect collinearity"
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
        pd.DataFrame(records)
        .sort_values(
            "VIF",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def extract_high_correlation_pairs(
    correlation_matrix: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Extract unique feature pairs exceeding the threshold."""

    records: list[dict[str, object]] = []

    columns = correlation_matrix.columns.tolist()

    for row_index in range(len(columns)):
        for column_index in range(
            row_index + 1,
            len(columns),
        ):
            correlation = correlation_matrix.iloc[
                row_index,
                column_index,
            ]

            if abs(correlation) >= threshold:
                records.append(
                    {
                        "Feature_1": columns[row_index],
                        "Feature_2": columns[column_index],
                        "Spearman_Correlation": correlation,
                        "Absolute_Correlation": abs(correlation),
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
        pd.DataFrame(records)
        .sort_values(
            "Absolute_Correlation",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# 3. MAIN PIPELINE
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
    # 3.1 Load and validate panels
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
        feature_panel,
        {
            "Date",
            "Ticker",
            TARGET,
            "N_t",
            "Model_N_t",
            *FEATURES,
        },
        "Feature panel",
    )

    validate_panel(
        target_panel,
        {
            "Date",
            "Ticker",
            TARGET,
            "N_t",
        },
        "Target panel",
    )

    # --------------------------------------------------------
    # 3.2 Missingness and finite-value checks
    # --------------------------------------------------------

    feature_values = feature_panel[
        FEATURES
    ].to_numpy(dtype=float)

    missing_feature_values = int(
        feature_panel[FEATURES]
        .isna()
        .sum()
        .sum()
    )

    infinite_feature_values = int(
        np.isinf(feature_values).sum()
    )

    if missing_feature_values > 0:
        raise ValueError(
            f"{missing_feature_values:,} missing feature values detected."
        )

    if infinite_feature_values > 0:
        raise ValueError(
            f"{infinite_feature_values:,} infinite feature values detected."
        )

    if not feature_panel[TARGET].between(
        0,
        1,
        inclusive="both",
    ).all():
        raise ValueError(
            "Target values outside [0, 1] were detected."
        )

    if not feature_panel["N_t"].eq(40).all():
        raise ValueError(
            "The stored target-universe size N_t is not always 40."
        )

    # --------------------------------------------------------
    # 3.3 Model eligibility and excluded observations
    # --------------------------------------------------------

    feature_keys = feature_panel[
        ["Date", "Ticker"]
    ].copy()

    exclusions = target_panel.merge(
        feature_keys,
        on=["Date", "Ticker"],
        how="left",
        indicator=True,
    )

    exclusions = exclusions.loc[
        exclusions["_merge"].eq("left_only")
    ].drop(columns="_merge")

    exclusions["Exclusion_Reason"] = (
        "Incomplete rolling feature history"
    )

    first_model_date = feature_panel[
        "Date"
    ].min()

    exclusions["Exclusion_Category"] = np.where(
        exclusions["Date"] < first_model_date,
        "Initial feature warm-up",
        "Security-specific insufficient history",
    )

    exclusions.to_csv(
        OUTPUT_EXCLUSIONS,
        index=False,
    )

    # --------------------------------------------------------
    # 3.4 Daily cross-sectional availability
    # --------------------------------------------------------

    target_daily_counts = (
        target_panel.groupby("Date")["Ticker"]
        .nunique()
        .rename("Target_Universe_Size")
    )

    model_daily_counts = (
        feature_panel.groupby("Date")["Ticker"]
        .nunique()
        .rename("Model_Eligible_Size")
    )

    daily_counts = (
        pd.concat(
            [
                target_daily_counts,
                model_daily_counts,
            ],
            axis=1,
        )
        .fillna(0)
        .reset_index()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    daily_counts[
        "Model_Eligible_Size"
    ] = daily_counts[
        "Model_Eligible_Size"
    ].astype(int)

    daily_counts[
        "Excluded_Securities"
    ] = (
        daily_counts["Target_Universe_Size"]
        - daily_counts["Model_Eligible_Size"]
    )

    daily_counts[
        "Model_Eligibility_Rate"
    ] = (
        daily_counts["Model_Eligible_Size"]
        / daily_counts["Target_Universe_Size"]
    )

    daily_counts.to_csv(
        OUTPUT_DAILY_COUNTS,
        index=False,
    )

    model_period_counts = daily_counts.loc[
        daily_counts["Date"] >= first_model_date
    ].copy()

    # --------------------------------------------------------
    # 3.5 Standardisation and extreme-value diagnostics
    # --------------------------------------------------------

    extreme_records: list[dict[str, object]] = []

    for feature in FEATURES:
        daily_statistics = (
            feature_panel.groupby("Date")[feature]
            .agg(["mean", "std"])
        )

        absolute_values = feature_panel[
            feature
        ].abs()

        extreme_records.append(
            {
                "Feature": feature,
                "Minimum": feature_panel[feature].min(),
                "Maximum": feature_panel[feature].max(),
                "Global_Mean": feature_panel[feature].mean(),
                "Global_Standard_Deviation": feature_panel[feature].std(),
                "Maximum_Absolute_Daily_Mean": (
                    daily_statistics["mean"]
                    .abs()
                    .max()
                ),
                "Minimum_Daily_Standard_Deviation": (
                    daily_statistics["std"].min()
                ),
                "Median_Daily_Standard_Deviation": (
                    daily_statistics["std"].median()
                ),
                "Maximum_Daily_Standard_Deviation": (
                    daily_statistics["std"].max()
                ),
                "Observations_Absolute_Z_Above_3": int(
                    absolute_values.gt(3).sum()
                ),
                "Observations_Absolute_Z_Above_4": int(
                    absolute_values.gt(4).sum()
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
    # 3.6 Correlation and VIF diagnostics
    # --------------------------------------------------------

    spearman_matrix = feature_panel[
        FEATURES
    ].corr(method="spearman")

    pearson_matrix = feature_panel[
        FEATURES
    ].corr(method="pearson")

    high_correlations = (
        extract_high_correlation_pairs(
            spearman_matrix,
            threshold=HIGH_CORRELATION_THRESHOLD,
        )
    )

    high_correlations.to_csv(
        OUTPUT_HIGH_CORRELATIONS,
        index=False,
    )

    target_correlation_records = []

    for feature in FEATURES:
        target_correlation_records.append(
            {
                "Feature": feature,
                "Spearman_Correlation_With_Target": (
                    feature_panel[
                        [feature, TARGET]
                    ]
                    .corr(method="spearman")
                    .iloc[0, 1]
                ),
                "Pearson_Correlation_With_Target": (
                    feature_panel[
                        [feature, TARGET]
                    ]
                    .corr(method="pearson")
                    .iloc[0, 1]
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

    vif_diagnostics = calculate_vif(
        feature_panel[FEATURES]
    )

    vif_diagnostics.to_csv(
        OUTPUT_VIF,
        index=False,
    )

    # --------------------------------------------------------
    # 3.7 Condition diagnostics
    # --------------------------------------------------------

    eigenvalues = np.linalg.eigvalsh(
        pearson_matrix.to_numpy()
    )

    largest_eigenvalue = float(
        eigenvalues.max()
    )

    smallest_eigenvalue = float(
        eigenvalues.min()
    )

    correlation_condition_number = float(
        np.linalg.cond(
            pearson_matrix.to_numpy()
        )
    )

    condition_index = (
        np.inf
        if smallest_eigenvalue <= 0
        else np.sqrt(
            largest_eigenvalue
            / smallest_eigenvalue
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
                len(FEATURES),
                np.linalg.matrix_rank(
                    pearson_matrix.to_numpy()
                ),
                largest_eigenvalue,
                smallest_eigenvalue,
                correlation_condition_number,
                condition_index,
            ],
        }
    )

    condition_diagnostics.to_csv(
        OUTPUT_CONDITION_DIAGNOSTICS,
        index=False,
    )

    # --------------------------------------------------------
    # 3.8 Annual panel coverage
    # --------------------------------------------------------

    annual_source = feature_panel.copy()

    annual_source["Year"] = (
        annual_source["Date"].dt.year
    )

    annual_coverage = (
        annual_source.groupby(
            "Year",
            as_index=False,
        )
        .agg(
            Stock_Day_Observations=("Ticker", "size"),
            Trading_Dates=("Date", "nunique"),
            Unique_Securities=("Ticker", "nunique"),
        )
    )

    annual_daily_counts = (
        annual_source.groupby(
            ["Year", "Date"]
        )["Ticker"]
        .nunique()
        .reset_index(
            name="Daily_Model_Size"
        )
    )

    annual_daily_summary = (
        annual_daily_counts.groupby(
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
    # 3.9 Model cross-sectional-size figure
    # --------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(11, 5)
    )

    axis.plot(
        model_period_counts["Date"],
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

    axis.set_xlabel("Trading date")
    axis.set_ylabel("Number of securities")

    axis.set_ylim(
        max(
            0,
            int(
                model_period_counts[
                    "Model_Eligible_Size"
                ].min()
            ) - 1,
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

    plt.close(figure)

    # --------------------------------------------------------
    # 3.10 Final summary
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
        ].eq(
            "Initial feature warm-up"
        ).sum()
    )

    later_history_exclusions = int(
        exclusions[
            "Exclusion_Category"
        ].eq(
            "Security-specific insufficient history"
        ).sum()
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
                len(target_panel),
                len(feature_panel),
                len(exclusions),
                initial_warmup_exclusions,
                later_history_exclusions,
                feature_panel["Date"].min().date(),
                feature_panel["Date"].max().date(),
                feature_panel["Date"].nunique(),
                feature_panel["Model_N_t"].min(),
                feature_panel["Model_N_t"].mean(),
                feature_panel["Model_N_t"].max(),
                int(
                    feature_panel.duplicated(
                        ["Date", "Ticker"]
                    ).sum()
                ),
                missing_feature_values,
                infinite_feature_values,
                int(
                    vif_diagnostics[
                        "VIF"
                    ].ge(5).sum()
                ),
                int(
                    vif_diagnostics[
                        "VIF"
                    ].ge(10).sum()
                ),
                len(high_correlations),
                correlation_condition_number,
                condition_index,
            ],
        }
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # --------------------------------------------------------
    # 3.11 Terminal output
    # --------------------------------------------------------

    print("\n" + "=" * 74)
    print("SCRIPT 05B EMPIRICAL OUTPUTS")
    print("=" * 74)

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
        f"{feature_panel['Model_N_t'].min():,}/"
        f"{feature_panel['Model_N_t'].mean():.2f}/"
        f"{feature_panel['Model_N_t'].max():,}"
    )

    print("\nTrading dates by model cross-sectional size:")

    for count, frequency in (
        model_count_frequency.items()
    ):
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
        f"{vif_diagnostics['VIF'].ge(5).sum():,}"
    )

    print(
        "Features with VIF >= 10: "
        f"{vif_diagnostics['VIF'].ge(10).sum():,}"
    )

    print(
        "Spearman feature pairs with "
        f"|correlation| >= {HIGH_CORRELATION_THRESHOLD:.2f}: "
        f"{len(high_correlations):,}"
    )

    print(
        "Correlation-matrix condition number: "
        f"{correlation_condition_number:.3f}"
    )

    print(
        "Maximum condition index: "
        f"{condition_index:.3f}"
    )

    print("\nIntegrity checks passed:")
    print("- No duplicate stock-date observations.")
    print("- No missing or infinite model features.")
    print("- The original target-universe size remains 40.")
    print(
        "- Model eligibility losses are explicitly identified "
        "and preserved in an audit file."
    )
    print(
        "- VIF and condition diagnostics were calculated "
        "without modifying the feature panel."
    )

    print("\nOutput files:")
    print(f"- {OUTPUT_SUMMARY}")
    print(f"- {OUTPUT_DAILY_COUNTS}")
    print(f"- {OUTPUT_EXCLUSIONS}")
    print(f"- {OUTPUT_VIF}")
    print(f"- {OUTPUT_HIGH_CORRELATIONS}")
    print(f"- {OUTPUT_TARGET_CORRELATIONS}")
    print(f"- {OUTPUT_EXTREME_VALUES}")
    print(f"- {OUTPUT_ANNUAL_COVERAGE}")
    print(f"- {OUTPUT_CONDITION_DIAGNOSTICS}")
    print(f"- {OUTPUT_FIGURE}")

    print("=" * 74)


if __name__ == "__main__":
    main()