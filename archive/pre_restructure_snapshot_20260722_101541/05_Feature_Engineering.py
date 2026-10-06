"""
Script 05: Feature Engineering and Cross-Sectional Standardisation
------------------------------------------------------------------

Purpose
-------
Calculates predictive features on each security's complete trading history
before merging them onto the point-in-time active-universe target panel.

Feature families
----------------
1. Momentum
2. Volatility
3. Relative Strength Index
4. Moving-average position
5. Volume shock
6. Value-traded liquidity shock

All price-derived features use Adjusted Close.

No winsorisation is performed in this script. Any winsorisation thresholds
must be estimated independently within each chronological training window
during model estimation.

Inputs
------
1. JSE_Broad_Trading_Panel.csv
2. JSE_Panel_Target_Engineered.csv

Outputs
-------
1. JSE_Panel_Features_Engineered.csv
2. outputs/audits/JSE_Feature_Engineering_Summary.csv
3. outputs/audits/JSE_Feature_Descriptive_Statistics.csv
4. outputs/audits/JSE_Feature_Cross_Sectional_Checks.csv
5. outputs/figures/figure_4_4_feature_correlation_heatmap.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_BROAD_PANEL = Path("JSE_Broad_Trading_Panel.csv")
INPUT_TARGET_PANEL = Path("JSE_Panel_Target_Engineered.csv")

OUTPUT_FEATURE_PANEL = Path("JSE_Panel_Features_Engineered.csv")

AUDIT_DIRECTORY = Path("outputs") / "audits"
FIGURE_DIRECTORY = Path("outputs") / "figures"

OUTPUT_SUMMARY = (
    AUDIT_DIRECTORY
    / "JSE_Feature_Engineering_Summary.csv"
)

OUTPUT_DESCRIPTIVE_STATISTICS = (
    AUDIT_DIRECTORY
    / "JSE_Feature_Descriptive_Statistics.csv"
)

OUTPUT_CROSS_SECTIONAL_CHECKS = (
    AUDIT_DIRECTORY
    / "JSE_Feature_Cross_Sectional_Checks.csv"
)

OUTPUT_CORRELATION_FIGURE = (
    FIGURE_DIRECTORY
    / "figure_4_4_feature_correlation_heatmap.png"
)


# Feature horizons in market trading observations
MOMENTUM_1M_WINDOW = 21
MOMENTUM_3M_WINDOW = 63
MOMENTUM_6M_WINDOW = 126
VOLATILITY_WINDOW = 21
RSI_WINDOW = 14
SMA_50_WINDOW = 50
SMA_200_WINDOW = 200
VOLUME_SHOCK_WINDOW = 21
VALUE_TRADED_SHOCK_WINDOW = 63


RAW_FEATURE_COLUMNS = [
    "Mom_1M_Raw",
    "Mom_3M_Raw",
    "Mom_6M_Raw",
    "Vol_21D_Raw",
    "RSI_14_Raw",
    "SMA_50_Ratio_Raw",
    "SMA_200_Ratio_Raw",
    "Volume_Shock_21D_Raw",
    "Value_Traded_Shock_63D_Raw",
]

STANDARDISED_FEATURE_COLUMNS = [
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

FEATURE_MAPPING = dict(
    zip(
        RAW_FEATURE_COLUMNS,
        STANDARDISED_FEATURE_COLUMNS,
    )
)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def validate_input_files() -> None:
    """Confirm that the required pipeline files exist."""

    required_files = [
        INPUT_BROAD_PANEL,
        INPUT_TARGET_PANEL,
    ]

    missing_files = [
        str(filepath)
        for filepath in required_files
        if not filepath.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required input files are missing:\n- "
            + "\n- ".join(missing_files)
        )


def validate_broad_panel(panel: pd.DataFrame) -> None:
    """Validate the broad cleaned trading panel."""

    required_columns = {
        "Date",
        "Ticker",
        "Adj Close",
        "Close",
        "Volume",
    }

    missing_columns = required_columns.difference(
        panel.columns
    )

    if missing_columns:
        raise ValueError(
            "The broad trading panel is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if panel.duplicated(["Date", "Ticker"]).any():
        raise ValueError(
            "Duplicate Date-Ticker observations exist in the "
            "broad trading panel."
        )

    if panel[
        ["Date", "Ticker", "Adj Close", "Close", "Volume"]
    ].isna().any().any():
        raise ValueError(
            "Missing values exist in required broad-panel fields."
        )

    if panel["Adj Close"].le(0).any():
        raise ValueError(
            "Adjusted Close contains non-positive values."
        )

    if panel["Close"].le(0).any():
        raise ValueError(
            "Close contains non-positive values."
        )

    if panel["Volume"].lt(0).any():
        raise ValueError(
            "Volume contains negative values."
        )


def validate_target_panel(panel: pd.DataFrame) -> None:
    """Validate the target-engineered active panel."""

    required_columns = {
        "Date",
        "Ticker",
        "Forward_Return",
        "N_t",
        "Daily_Return_Rank",
        "Target_Y",
    }

    missing_columns = required_columns.difference(
        panel.columns
    )

    if missing_columns:
        raise ValueError(
            "The target panel is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if panel.duplicated(["Date", "Ticker"]).any():
        raise ValueError(
            "Duplicate Date-Ticker observations exist in the "
            "target panel."
        )

    if panel[list(required_columns)].isna().any().any():
        raise ValueError(
            "Missing values exist in required target-panel fields."
        )

    if not panel["Target_Y"].between(
        0,
        1,
        inclusive="both",
    ).all():
        raise ValueError(
            "Target_Y contains values outside [0, 1]."
        )


def calculate_wilder_rsi(
    adjusted_close: pd.Series,
    period: int = RSI_WINDOW,
) -> pd.Series:
    """
    Calculate Wilder-style Relative Strength Index.

    The first valid value requires at least `period` return observations.
    """

    price_change = adjusted_close.diff()

    gains = price_change.clip(lower=0.0)
    losses = -price_change.clip(upper=0.0)

    average_gain = gains.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    average_loss = losses.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    relative_strength = (
        average_gain / average_loss
    )

    rsi = 100.0 - (
        100.0 / (1.0 + relative_strength)
    )

    # No losses but at least one gain implies RSI = 100.
    rsi = rsi.mask(
        average_loss.eq(0)
        & average_gain.gt(0),
        100.0,
    )

    # No gains and no losses implies a flat market, assigned RSI = 50.
    rsi = rsi.mask(
        average_loss.eq(0)
        & average_gain.eq(0),
        50.0,
    )

    return rsi


def cross_sectional_standardise(
    panel: pd.DataFrame,
    raw_column: str,
    output_column: str,
) -> None:
    """
    Apply daily cross-sectional z-scoring.

    If every security has the same value on a date, the standardised
    value is set to zero because the feature contains no cross-sectional
    information on that date.
    """

    daily_mean = (
        panel.groupby("Date")[raw_column]
        .transform("mean")
    )

    daily_std = (
        panel.groupby("Date")[raw_column]
        .transform("std")
    )

    panel[output_column] = np.where(
        daily_std.gt(0),
        (
            panel[raw_column] - daily_mean
        ) / daily_std,
        0.0,
    )


def make_descriptive_statistics(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Create a combined raw and standardised descriptive table."""

    raw_statistics = (
        panel[RAW_FEATURE_COLUMNS]
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
            ]
        )
        .transpose()
        .reset_index()
        .rename(columns={"index": "Feature"})
    )

    raw_statistics.insert(
        1,
        "Feature_Scale",
        "Raw",
    )

    standardised_statistics = (
        panel[STANDARDISED_FEATURE_COLUMNS]
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
            ]
        )
        .transpose()
        .reset_index()
        .rename(columns={"index": "Feature"})
    )

    standardised_statistics.insert(
        1,
        "Feature_Scale",
        "Daily cross-sectional z-score",
    )

    return pd.concat(
        [
            raw_statistics,
            standardised_statistics,
        ],
        ignore_index=True,
    )


def make_cross_sectional_checks(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise daily standardisation diagnostics by feature."""

    records: list[dict[str, object]] = []

    for feature in STANDARDISED_FEATURE_COLUMNS:
        daily_statistics = (
            panel.groupby("Date")[feature]
            .agg(["mean", "std", "min", "max"])
        )

        records.append(
            {
                "Feature": feature,
                "Maximum_Absolute_Daily_Mean": float(
                    daily_statistics["mean"]
                    .abs()
                    .max()
                ),
                "Minimum_Daily_Standard_Deviation": float(
                    daily_statistics["std"].min()
                ),
                "Median_Daily_Standard_Deviation": float(
                    daily_statistics["std"].median()
                ),
                "Maximum_Daily_Standard_Deviation": float(
                    daily_statistics["std"].max()
                ),
                "Minimum_Standardised_Value": float(
                    panel[feature].min()
                ),
                "Maximum_Standardised_Value": float(
                    panel[feature].max()
                ),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# 3. MAIN PIPELINE
# ============================================================

def main() -> None:
    print("=" * 74)
    print("SCRIPT 05: FEATURE ENGINEERING AND STANDARDISATION")
    print("=" * 74)

    validate_input_files()

    AUDIT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 3.1 Load and validate complete broad histories
    # --------------------------------------------------------

    broad = pd.read_csv(
        INPUT_BROAD_PANEL
    )

    broad["Date"] = pd.to_datetime(
        broad["Date"],
        errors="raise",
    )

    for column in [
        "Adj Close",
        "Close",
        "Volume",
    ]:
        broad[column] = pd.to_numeric(
            broad[column],
            errors="raise",
        )

    broad = (
        broad.sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    validate_broad_panel(broad)

    broad_rows_loaded = len(broad)

    # --------------------------------------------------------
    # 3.2 Core adjusted-price transformations
    # --------------------------------------------------------

    broad["Log_Return_1D"] = (
        broad.groupby("Ticker", sort=False)["Adj Close"]
        .transform(
            lambda prices: np.log(
                prices / prices.shift(1)
            )
        )
    )

    broad["Mom_1M_Raw"] = (
        broad.groupby("Ticker", sort=False)["Adj Close"]
        .transform(
            lambda prices: (
                prices / prices.shift(MOMENTUM_1M_WINDOW)
                - 1.0
            )
        )
    )

    broad["Mom_3M_Raw"] = (
        broad.groupby("Ticker", sort=False)["Adj Close"]
        .transform(
            lambda prices: (
                prices / prices.shift(MOMENTUM_3M_WINDOW)
                - 1.0
            )
        )
    )

    broad["Mom_6M_Raw"] = (
        broad.groupby("Ticker", sort=False)["Adj Close"]
        .transform(
            lambda prices: (
                prices / prices.shift(MOMENTUM_6M_WINDOW)
                - 1.0
            )
        )
    )

    broad["Vol_21D_Raw"] = (
        broad.groupby("Ticker", sort=False)["Log_Return_1D"]
        .transform(
            lambda returns: returns.rolling(
                window=VOLATILITY_WINDOW,
                min_periods=VOLATILITY_WINDOW,
            ).std()
        )
    )

    # --------------------------------------------------------
    # 3.3 Relative Strength Index
    # --------------------------------------------------------

    broad["RSI_14_Raw"] = (
        broad.groupby("Ticker", sort=False)["Adj Close"]
        .transform(calculate_wilder_rsi)
    )

    # --------------------------------------------------------
    # 3.4 Moving-average position
    # --------------------------------------------------------

    broad["SMA_50"] = (
        broad.groupby("Ticker", sort=False)["Adj Close"]
        .transform(
            lambda prices: prices.rolling(
                window=SMA_50_WINDOW,
                min_periods=SMA_50_WINDOW,
            ).mean()
        )
    )

    broad["SMA_200"] = (
        broad.groupby("Ticker", sort=False)["Adj Close"]
        .transform(
            lambda prices: prices.rolling(
                window=SMA_200_WINDOW,
                min_periods=SMA_200_WINDOW,
            ).mean()
        )
    )

    broad["SMA_50_Ratio_Raw"] = (
        broad["Adj Close"] / broad["SMA_50"]
    )

    broad["SMA_200_Ratio_Raw"] = (
        broad["Adj Close"] / broad["SMA_200"]
    )

    # --------------------------------------------------------
    # 3.5 Volume and liquidity shocks
    # --------------------------------------------------------

    broad["Log_Volume"] = np.log1p(
        broad["Volume"]
    )

    broad["Prior_Log_Volume_Mean_21D"] = (
        broad.groupby("Ticker", sort=False)["Log_Volume"]
        .transform(
            lambda values: values.shift(1).rolling(
                window=VOLUME_SHOCK_WINDOW,
                min_periods=VOLUME_SHOCK_WINDOW,
            ).mean()
        )
    )

    broad["Volume_Shock_21D_Raw"] = (
        broad["Log_Volume"]
        - broad["Prior_Log_Volume_Mean_21D"]
    )

    broad["Value_Traded"] = (
        broad["Close"] * broad["Volume"]
    )

    broad["Log_Value_Traded"] = np.log1p(
        broad["Value_Traded"]
    )

    broad["Prior_Log_Value_Traded_Mean_63D"] = (
        broad.groupby(
            "Ticker",
            sort=False,
        )["Log_Value_Traded"]
        .transform(
            lambda values: values.shift(1).rolling(
                window=VALUE_TRADED_SHOCK_WINDOW,
                min_periods=VALUE_TRADED_SHOCK_WINDOW,
            ).mean()
        )
    )

    broad["Value_Traded_Shock_63D_Raw"] = (
        broad["Log_Value_Traded"]
        - broad["Prior_Log_Value_Traded_Mean_63D"]
    )

    # Convert any numerical infinities to missing values so they
    # cannot enter the final feature matrix silently.
    broad[RAW_FEATURE_COLUMNS] = (
        broad[RAW_FEATURE_COLUMNS]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    # --------------------------------------------------------
    # 3.6 Load and validate the active target panel
    # --------------------------------------------------------

    target_panel = pd.read_csv(
        INPUT_TARGET_PANEL
    )

    target_panel["Date"] = pd.to_datetime(
        target_panel["Date"],
        errors="raise",
    )

    target_panel = (
        target_panel.sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )

    validate_target_panel(target_panel)

    target_rows_loaded = len(target_panel)

    # --------------------------------------------------------
    # 3.7 Merge features onto the active target observations
    # --------------------------------------------------------

    feature_source = broad[
        [
            "Date",
            "Ticker",
            *RAW_FEATURE_COLUMNS,
        ]
    ].copy()

    model_panel = target_panel.merge(
        feature_source,
        on=["Date", "Ticker"],
        how="left",
        validate="one_to_one",
    )

    missing_before_complete_case = (
        model_panel[RAW_FEATURE_COLUMNS]
        .isna()
        .sum()
        .rename("Missing_Observations")
        .reset_index()
        .rename(columns={"index": "Feature"})
    )

    rows_before_feature_filter = len(
        model_panel
    )

    model_panel = model_panel.dropna(
        subset=RAW_FEATURE_COLUMNS
    ).copy()

    rows_removed_for_feature_warmup = (
        rows_before_feature_filter
        - len(model_panel)
    )

    if model_panel.empty:
        raise ValueError(
            "No complete feature observations remain."
        )

    # --------------------------------------------------------
    # 3.8 Daily cross-sectional standardisation
    # --------------------------------------------------------

    for raw_feature, output_feature in FEATURE_MAPPING.items():
        cross_sectional_standardise(
            panel=model_panel,
            raw_column=raw_feature,
            output_column=output_feature,
        )

    model_panel.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    if model_panel[
        STANDARDISED_FEATURE_COLUMNS
    ].isna().any().any():
        raise ValueError(
            "Missing values were created during cross-sectional "
            "standardisation."
        )

    model_panel["Model_N_t"] = (
        model_panel.groupby("Date")["Ticker"]
        .transform("nunique")
        .astype(int)
    )

    model_panel = (
        model_panel.sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )

    if model_panel.duplicated(
        ["Date", "Ticker"]
    ).any():
        raise AssertionError(
            "Duplicate stock-date observations exist in the "
            "final feature panel."
        )

    # --------------------------------------------------------
    # 3.9 Standardisation checks
    # --------------------------------------------------------

    cross_sectional_checks = (
        make_cross_sectional_checks(
            model_panel
        )
    )

    maximum_absolute_daily_mean = float(
        cross_sectional_checks[
            "Maximum_Absolute_Daily_Mean"
        ].max()
    )

    if maximum_absolute_daily_mean > 1e-10:
        raise AssertionError(
            "Daily cross-sectional means are not sufficiently "
            "close to zero."
        )

    # --------------------------------------------------------
    # 3.10 Save feature panel and audit tables
    # --------------------------------------------------------

    model_panel.to_csv(
        OUTPUT_FEATURE_PANEL,
        index=False,
    )

    descriptive_statistics = (
        make_descriptive_statistics(
            model_panel
        )
    )

    descriptive_statistics.to_csv(
        OUTPUT_DESCRIPTIVE_STATISTICS,
        index=False,
    )

    cross_sectional_checks.to_csv(
        OUTPUT_CROSS_SECTIONAL_CHECKS,
        index=False,
    )

    model_dates = model_panel["Date"].nunique()

    daily_model_counts = (
        model_panel.groupby("Date")["Ticker"]
        .nunique()
    )

    summary = pd.DataFrame(
        {
            "Statistic": [
                "Broad trading-panel observations loaded",
                "Target-panel observations loaded",
                "Rows before complete-feature filtering",
                "Rows removed for rolling-feature warm-up",
                "Final feature-engineered observations",
                "Final feature-engineered dates",
                "Feature-panel start date",
                "Feature-panel end date",
                "Minimum daily model cross-section",
                "Mean daily model cross-section",
                "Maximum daily model cross-section",
                "Number of raw features",
                "Number of standardised features",
                "Duplicate stock-date observations",
                "Missing standardised feature values",
                "Maximum absolute daily standardised mean",
            ],
            "Value": [
                broad_rows_loaded,
                target_rows_loaded,
                rows_before_feature_filter,
                rows_removed_for_feature_warmup,
                len(model_panel),
                model_dates,
                model_panel["Date"].min().date(),
                model_panel["Date"].max().date(),
                int(daily_model_counts.min()),
                float(daily_model_counts.mean()),
                int(daily_model_counts.max()),
                len(RAW_FEATURE_COLUMNS),
                len(STANDARDISED_FEATURE_COLUMNS),
                int(
                    model_panel.duplicated(
                        ["Date", "Ticker"]
                    ).sum()
                ),
                int(
                    model_panel[
                        STANDARDISED_FEATURE_COLUMNS
                    ].isna().sum().sum()
                ),
                maximum_absolute_daily_mean,
            ],
        }
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # Include per-feature pre-filter missingness below the summary
    # in a separate companion table inside the same audit directory.
    missing_before_complete_case.to_csv(
        AUDIT_DIRECTORY
        / "JSE_Feature_PreFilter_Missingness.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 3.11 Spearman correlation matrix and figure
    # --------------------------------------------------------

    correlation_columns = [
        *STANDARDISED_FEATURE_COLUMNS,
        "Target_Y",
    ]

    correlation_matrix = model_panel[
        correlation_columns
    ].corr(
        method="spearman"
    )

    correlation_matrix.to_csv(
        AUDIT_DIRECTORY
        / "JSE_Feature_Spearman_Correlation_Matrix.csv"
    )

    figure_size = max(
        9,
        len(correlation_columns) * 0.9,
    )

    figure, axis = plt.subplots(
        figsize=(figure_size, figure_size)
    )

    image = axis.imshow(
        correlation_matrix.to_numpy(),
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        aspect="equal",
    )

    axis.set_xticks(
        np.arange(len(correlation_columns))
    )

    axis.set_yticks(
        np.arange(len(correlation_columns))
    )

    axis.set_xticklabels(
        correlation_columns,
        rotation=45,
        ha="right",
    )

    axis.set_yticklabels(
        correlation_columns,
    )

    for row_index in range(
        len(correlation_columns)
    ):
        for column_index in range(
            len(correlation_columns)
        ):
            correlation_value = (
                correlation_matrix.iloc[
                    row_index,
                    column_index,
                ]
            )

            axis.text(
                column_index,
                row_index,
                f"{correlation_value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    axis.set_title(
        "Spearman Correlations among Standardised "
        "Predictors and the Ranking Target"
    )

    colour_bar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )

    colour_bar.set_label(
        "Spearman correlation"
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_CORRELATION_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    # --------------------------------------------------------
    # 3.12 Terminal outputs
    # --------------------------------------------------------

    print("\n" + "=" * 74)
    print("SCRIPT 05 EMPIRICAL OUTPUTS")
    print("=" * 74)

    print(
        "Broad trading-panel observations loaded: "
        f"{broad_rows_loaded:,}"
    )

    print(
        "Target-panel observations loaded: "
        f"{target_rows_loaded:,}"
    )

    print(
        "Rows removed for rolling-feature warm-up or "
        "incomplete feature history: "
        f"{rows_removed_for_feature_warmup:,}"
    )

    print(
        "Final feature-engineered observations: "
        f"{len(model_panel):,}"
    )

    print(
        "Final feature-engineered dates: "
        f"{model_dates:,}"
    )

    print(
        "Feature-panel date range: "
        f"{model_panel['Date'].min().date()} to "
        f"{model_panel['Date'].max().date()}"
    )

    print(
        "Daily model cross-sectional size "
        "(minimum/mean/maximum): "
        f"{daily_model_counts.min():,}/"
        f"{daily_model_counts.mean():.2f}/"
        f"{daily_model_counts.max():,}"
    )

    print(
        "Raw predictive features constructed: "
        f"{len(RAW_FEATURE_COLUMNS):,}"
    )

    print(
        "Standardised predictive features constructed: "
        f"{len(STANDARDISED_FEATURE_COLUMNS):,}"
    )

    print(
        "Duplicate stock-date observations: "
        f"{model_panel.duplicated(['Date', 'Ticker']).sum():,}"
    )

    print(
        "Missing standardised feature values: "
        f"{model_panel[STANDARDISED_FEATURE_COLUMNS].isna().sum().sum():,}"
    )

    print(
        "Maximum absolute daily standardised-feature mean: "
        f"{maximum_absolute_daily_mean:.3e}"
    )

    print("\nStandardised model features:")

    for feature in STANDARDISED_FEATURE_COLUMNS:
        print(f"- {feature}")

    print("\nIntegrity checks passed:")
    print(
        "- Raw features were calculated on complete broad "
        "ticker histories before active-universe filtering."
    )
    print(
        "- All price-derived features use Adjusted Close."
    )
    print(
        "- Volume and value-traded shocks use only current "
        "and prior observations."
    )
    print(
        "- The feature panel was merged one-to-one with the "
        "point-in-time target panel."
    )
    print(
        "- Every final feature has been standardised within "
        "its contemporaneous daily cross-section."
    )
    print(
        "- No full-sample winsorisation was applied."
    )

    print("\nCore output:")
    print(f"- {OUTPUT_FEATURE_PANEL}")

    print("\nAudit outputs:")
    print(f"- {OUTPUT_SUMMARY}")
    print(f"- {OUTPUT_DESCRIPTIVE_STATISTICS}")
    print(f"- {OUTPUT_CROSS_SECTIONAL_CHECKS}")
    print(
        "- "
        + str(
            AUDIT_DIRECTORY
            / "JSE_Feature_PreFilter_Missingness.csv"
        )
    )
    print(
        "- "
        + str(
            AUDIT_DIRECTORY
            / "JSE_Feature_Spearman_Correlation_Matrix.csv"
        )
    )

    print("\nFigure output:")
    print(f"- {OUTPUT_CORRELATION_FIGURE}")

    print("=" * 74)


if __name__ == "__main__":
    main()