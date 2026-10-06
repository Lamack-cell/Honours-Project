"""
Script 03B: Target Diagnostics and Academic Visualisation
---------------------------------------------------------

Purpose
-------
Validates the cross-sectional percentile target and produces a
thesis-ready distribution figure.

The expected distribution is based on the discrete 40-security
rank grid rather than a continuous uniform distribution.

Input
-----
JSE_Panel_Target_Engineered.csv

Outputs
-------
outputs/figures/figure_4_2_target_distribution.png
outputs/audits/JSE_Target_Distribution_Summary.csv
outputs/audits/JSE_Target_Value_Frequencies.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE = Path("JSE_Panel_Target_Engineered.csv")

AUDIT_DIRECTORY = Path("outputs") / "audits"
FIGURE_DIRECTORY = Path("outputs") / "figures"

OUTPUT_FIGURE = (
    FIGURE_DIRECTORY
    / "figure_4_2_target_distribution.png"
)

OUTPUT_SUMMARY = (
    AUDIT_DIRECTORY
    / "JSE_Target_Distribution_Summary.csv"
)

OUTPUT_FREQUENCIES = (
    AUDIT_DIRECTORY
    / "JSE_Target_Value_Frequencies.csv"
)

NUMBER_OF_BINS = 20


# ============================================================
# 2. DIRECTORY AND INPUT CHECKS
# ============================================================

AUDIT_DIRECTORY.mkdir(parents=True, exist_ok=True)
FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Required input file not found: {INPUT_FILE.resolve()}"
    )

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="raise",
)

required_columns = {
    "Date",
    "Ticker",
    "Forward_Return",
    "Daily_Return_Rank",
    "N_t",
    "Target_Y",
}

missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        "Target panel is missing required columns: "
        f"{sorted(missing_columns)}"
    )

if df.duplicated(["Date", "Ticker"]).any():
    raise ValueError(
        "Duplicate Date-Ticker observations detected."
    )

if df[list(required_columns)].isna().any().any():
    raise ValueError(
        "Missing values detected in required target fields."
    )

if not df["Target_Y"].between(
    0,
    1,
    inclusive="both",
).all():
    raise ValueError(
        "Target_Y contains values outside [0, 1]."
    )


# ============================================================
# 3. DAILY INTEGRITY CHECKS
# ============================================================

daily = (
    df.groupby("Date", as_index=False)
    .agg(
        Active_Securities=("Ticker", "nunique"),
        Stored_N_t=("N_t", "first"),
        Mean_Target_Y=("Target_Y", "mean"),
        Minimum_Target_Y=("Target_Y", "min"),
        Maximum_Target_Y=("Target_Y", "max"),
        Standard_Deviation_Target_Y=("Target_Y", "std"),
        Unique_Forward_Returns=("Forward_Return", "nunique"),
        Zero_Return_Observations=(
            "Forward_Return",
            lambda values: int(
                np.isclose(
                    values.to_numpy(),
                    0.0,
                    atol=1e-15,
                ).sum()
            ),
        ),
    )
)

if not (
    daily["Active_Securities"]
    == daily["Stored_N_t"]
).all():
    raise AssertionError(
        "Stored N_t does not match the observed daily security count."
    )

if not np.allclose(
    daily["Mean_Target_Y"],
    0.5,
    atol=1e-12,
):
    raise AssertionError(
        "At least one daily cross-section does not have a "
        "target mean of 0.5."
    )

if not np.allclose(
    daily["Minimum_Target_Y"],
    0.0,
    atol=1e-12,
):
    raise AssertionError(
        "At least one daily cross-section does not contain "
        "target value 0."
    )

if not np.allclose(
    daily["Maximum_Target_Y"],
    1.0,
    atol=1e-12,
):
    raise AssertionError(
        "At least one daily cross-section does not contain "
        "target value 1."
    )


# ============================================================
# 4. SUMMARY STATISTICS
# ============================================================

total_observations = len(df)
total_dates = df["Date"].nunique()

minimum_cross_section = int(
    daily["Active_Securities"].min()
)

maximum_cross_section = int(
    daily["Active_Securities"].max()
)

days_with_ties = int(
    (
        daily["Unique_Forward_Returns"]
        < daily["Active_Securities"]
    ).sum()
)

days_with_zero_return_ties = int(
    (
        daily["Zero_Return_Observations"] > 1
    ).sum()
)

percentage_tied_days = (
    days_with_ties / total_dates * 100
)

summary = pd.DataFrame(
    {
        "Statistic": [
            "Total target observations",
            "Total target dates",
            "Target-panel start date",
            "Target-panel end date",
            "Minimum daily cross-sectional size",
            "Maximum daily cross-sectional size",
            "Target minimum",
            "Target maximum",
            "Target mean",
            "Target standard deviation",
            "Days with tied returns",
            "Percentage of days with tied returns",
            "Days with tied zero returns",
        ],
        "Value": [
            total_observations,
            total_dates,
            df["Date"].min().date(),
            df["Date"].max().date(),
            minimum_cross_section,
            maximum_cross_section,
            df["Target_Y"].min(),
            df["Target_Y"].max(),
            df["Target_Y"].mean(),
            df["Target_Y"].std(),
            days_with_ties,
            percentage_tied_days,
            days_with_zero_return_ties,
        ],
    }
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False,
)


# ============================================================
# 5. TARGET-VALUE FREQUENCIES
# ============================================================

frequencies = (
    df.assign(
        Target_Y_Rounded=df["Target_Y"].round(8)
    )
    .groupby(
        "Target_Y_Rounded",
        as_index=False,
    )
    .agg(
        Observation_Count=("Target_Y", "size")
    )
    .sort_values("Target_Y_Rounded")
    .reset_index(drop=True)
)

frequencies.to_csv(
    OUTPUT_FREQUENCIES,
    index=False,
)


# ============================================================
# 6. NO-TIE DISCRETE RANK-GRID EXPECTATION
# ============================================================

daily_sizes = (
    df.groupby("Date")["N_t"]
    .first()
    .astype(int)
)

expected_target_values = np.concatenate(
    [
        np.arange(n, dtype=float) / (n - 1)
        for n in daily_sizes
        if n > 1
    ]
)

bin_edges = np.linspace(
    0.0,
    1.0,
    NUMBER_OF_BINS + 1,
)

observed_counts, _ = np.histogram(
    df["Target_Y"],
    bins=bin_edges,
)

expected_counts, _ = np.histogram(
    expected_target_values,
    bins=bin_edges,
)

bin_centres = (
    bin_edges[:-1]
    + bin_edges[1:]
) / 2


# ============================================================
# 7. THESIS-READY FIGURE
# ============================================================

figure, axis = plt.subplots(
    figsize=(10, 6)
)

bar_width = (
    bin_edges[1] - bin_edges[0]
) * 0.88

axis.bar(
    bin_centres,
    observed_counts,
    width=bar_width,
    edgecolor="black",
    alpha=0.80,
    label="Empirical target frequency",
)

axis.plot(
    bin_centres,
    expected_counts,
    linestyle="--",
    marker="o",
    linewidth=1.6,
    label="No-tie discrete rank-grid expectation",
)

axis.set_title(
    "Empirical Distribution of the Cross-Sectional "
    "Percentile Target"
)

axis.set_xlabel(
    "Realised cross-sectional percentile target"
)

axis.set_ylabel(
    "Number of stock-day observations"
)

axis.set_xlim(0, 1)

axis.grid(
    axis="y",
    linestyle="--",
    alpha=0.45,
)

axis.legend()

annotation_text = (
    f"{days_with_ties:,} of {total_dates:,} dates "
    f"contained tied returns\n"
    f"({percentage_tied_days:.2f}% of target dates)"
)

axis.text(
    0.02,
    0.96,
    annotation_text,
    transform=axis.transAxes,
    verticalalignment="top",
    bbox={
        "boxstyle": "round",
        "facecolor": "white",
        "alpha": 0.85,
    },
)

figure.tight_layout()

figure.savefig(
    OUTPUT_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close(figure)


# ============================================================
# 8. TERMINAL OUTPUT
# ============================================================

print("=" * 74)
print("SCRIPT 03B: TARGET DIAGNOSTICS AND VISUALISATION")
print("=" * 74)

print(
    f"Target observations validated: "
    f"{total_observations:,}"
)

print(
    f"Target dates validated: "
    f"{total_dates:,}"
)

print(
    "Target date range: "
    f"{df['Date'].min().date()} to "
    f"{df['Date'].max().date()}"
)

print(
    "Daily cross-sectional size: "
    f"{minimum_cross_section} to "
    f"{maximum_cross_section}"
)

print(
    f"Target minimum: "
    f"{df['Target_Y'].min():.4f}"
)

print(
    f"Target maximum: "
    f"{df['Target_Y'].max():.4f}"
)

print(
    f"Target mean: "
    f"{df['Target_Y'].mean():.4f}"
)

print(
    "Target standard deviation: "
    f"{df['Target_Y'].std():.6f}"
)

print(
    f"Days containing tied returns: "
    f"{days_with_ties:,}"
)

print(
    "Percentage of dates containing tied returns: "
    f"{percentage_tied_days:.2f}%"
)

print(
    "Days containing tied zero returns: "
    f"{days_with_zero_return_ties:,}"
)

print("\nIntegrity checks passed:")
print("- No duplicate stock-date observations.")
print("- No missing required target fields.")
print("- All target values lie within [0, 1].")
print("- Every daily target mean equals 0.5.")
print("- Every daily cross-section contains target values 0 and 1.")
print(
    "- The empirical distribution was compared with the "
    "appropriate discrete rank-grid expectation."
)

print("\nOutput files:")
print(f"- {OUTPUT_FIGURE}")
print(f"- {OUTPUT_SUMMARY}")
print(f"- {OUTPUT_FREQUENCIES}")

print("=" * 74)