"""
Script 03B: Target Diagnostics and Academic Visualisation
---------------------------------------------------------

Purpose
-------
Validates the cross-sectional percentile target and produces a
thesis-ready distribution figure.

The expected distribution is based on the discrete daily rank grid
rather than a continuous uniform distribution.

Input
-----
data/interim/03_target/JSE_Panel_Target_Engineered.csv

Outputs
-------
outputs/figures/chapter_4/figure_4_2_target_distribution.png
outputs/audits/03_target/JSE_Target_Distribution_Summary.csv
outputs/audits/03_target/JSE_Target_Value_Frequencies.csv

Run from the project root
-------------------------
python .\\scripts\\03B_Advanced_Target_Visual.py
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

INPUT_FILE = (
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
    / "03_target"
)

FIGURE_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "chapter_4"
)

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


# ============================================================
# 2. CONFIGURATION
# ============================================================

NUMBER_OF_BINS = 20

REQUIRED_COLUMNS = {
    "Date",
    "Ticker",
    "Forward_Return",
    "Daily_Return_Rank",
    "N_t",
    "Target_Y",
}


# ============================================================
# 3. HELPERS
# ============================================================

def relative_path(path: Path) -> str:
    """Return a readable path relative to the project root."""

    return str(
        path.relative_to(
            PROJECT_ROOT
        )
    )


# ============================================================
# 4. MAIN PIPELINE
# ============================================================

def main() -> None:
    print("=" * 74)
    print("SCRIPT 03B: TARGET DIAGNOSTICS AND VISUALISATION")
    print("=" * 74)

    AUDIT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Required input file not found:\n"
            f"{INPUT_FILE.resolve()}"
        )

    # --------------------------------------------------------
    # 4.1 Load and validate target panel
    # --------------------------------------------------------

    target_panel = pd.read_csv(
        INPUT_FILE
    )

    target_panel["Date"] = pd.to_datetime(
        target_panel["Date"],
        errors="raise",
    )

    missing_columns = REQUIRED_COLUMNS.difference(
        target_panel.columns
    )

    if missing_columns:
        raise ValueError(
            "Target panel is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if target_panel.empty:
        raise ValueError(
            "The target panel contains no observations."
        )

    if target_panel.duplicated(
        ["Date", "Ticker"]
    ).any():
        raise ValueError(
            "Duplicate Date-Ticker observations detected."
        )

    if target_panel[
        list(REQUIRED_COLUMNS)
    ].isna().any().any():
        raise ValueError(
            "Missing values detected in required target fields."
        )

    if not target_panel[
        "Target_Y"
    ].between(
        0,
        1,
        inclusive="both",
    ).all():
        raise ValueError(
            "Target_Y contains values outside [0, 1]."
        )

    if not np.isfinite(
        target_panel[
            [
                "Forward_Return",
                "Daily_Return_Rank",
                "N_t",
                "Target_Y",
            ]
        ].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "The target panel contains non-finite numerical values."
        )

    target_panel = (
        target_panel.sort_values(
            ["Date", "Ticker"]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # 4.2 Daily integrity diagnostics
    # --------------------------------------------------------

    daily = (
        target_panel
        .groupby(
            "Date",
            as_index=False,
        )
        .agg(
            Active_Securities=(
                "Ticker",
                "nunique",
            ),
            Stored_N_t=(
                "N_t",
                "first",
            ),
            Mean_Target_Y=(
                "Target_Y",
                "mean",
            ),
            Minimum_Target_Y=(
                "Target_Y",
                "min",
            ),
            Maximum_Target_Y=(
                "Target_Y",
                "max",
            ),
            Standard_Deviation_Target_Y=(
                "Target_Y",
                "std",
            ),
            Unique_Forward_Returns=(
                "Forward_Return",
                "nunique",
            ),
            Zero_Return_Observations=(
                "Forward_Return",
                lambda values: int(
                    np.isclose(
                        values.to_numpy(
                            dtype=float
                        ),
                        0.0,
                        rtol=0.0,
                        atol=1e-15,
                    ).sum()
                ),
            ),
        )
        .sort_values(
            "Date"
        )
        .reset_index(
            drop=True
        )
    )

    if not (
        daily["Active_Securities"]
        == daily["Stored_N_t"]
    ).all():
        raise AssertionError(
            "Stored N_t does not match the observed "
            "daily security count."
        )

    if not np.allclose(
        daily["Mean_Target_Y"],
        0.5,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError(
            "At least one daily cross-section does not "
            "have a target mean of 0.5."
        )

    if not np.allclose(
        daily["Minimum_Target_Y"],
        0.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError(
            "At least one daily cross-section does not "
            "contain target value 0."
        )

    if not np.allclose(
        daily["Maximum_Target_Y"],
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError(
            "At least one daily cross-section does not "
            "contain target value 1."
        )

    # --------------------------------------------------------
    # 4.3 Summary statistics
    # --------------------------------------------------------

    total_observations = len(
        target_panel
    )

    total_dates = target_panel[
        "Date"
    ].nunique()

    minimum_cross_section = int(
        daily[
            "Active_Securities"
        ].min()
    )

    maximum_cross_section = int(
        daily[
            "Active_Securities"
        ].max()
    )

    days_with_ties = int(
        (
            daily[
                "Unique_Forward_Returns"
            ]
            < daily[
                "Active_Securities"
            ]
        ).sum()
    )

    days_with_zero_return_ties = int(
        (
            daily[
                "Zero_Return_Observations"
            ]
            > 1
        ).sum()
    )

    percentage_tied_days = (
        days_with_ties
        / total_dates
        * 100
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
                "Minimum forward return",
                "Maximum forward return",
                "Days with tied returns",
                "Percentage of days with tied returns",
                "Days with tied zero returns",
            ],
            "Value": [
                total_observations,
                total_dates,
                target_panel[
                    "Date"
                ].min().date(),
                target_panel[
                    "Date"
                ].max().date(),
                minimum_cross_section,
                maximum_cross_section,
                target_panel[
                    "Target_Y"
                ].min(),
                target_panel[
                    "Target_Y"
                ].max(),
                target_panel[
                    "Target_Y"
                ].mean(),
                target_panel[
                    "Target_Y"
                ].std(),
                target_panel[
                    "Forward_Return"
                ].min(),
                target_panel[
                    "Forward_Return"
                ].max(),
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

    # --------------------------------------------------------
    # 4.4 Target-value frequencies
    # --------------------------------------------------------

    frequencies = (
        target_panel.assign(
            Target_Y_Rounded=target_panel[
                "Target_Y"
            ].round(
                8
            )
        )
        .groupby(
            "Target_Y_Rounded",
            as_index=False,
        )
        .agg(
            Observation_Count=(
                "Target_Y",
                "size",
            )
        )
        .sort_values(
            "Target_Y_Rounded"
        )
        .reset_index(
            drop=True
        )
    )

    frequencies[
        "Observation_Share"
    ] = (
        frequencies[
            "Observation_Count"
        ]
        / total_observations
    )

    frequencies.to_csv(
        OUTPUT_FREQUENCIES,
        index=False,
    )

    # --------------------------------------------------------
    # 4.5 Construct no-tie discrete rank-grid expectation
    # --------------------------------------------------------

    daily_sizes = (
        target_panel
        .groupby(
            "Date"
        )[
            "N_t"
        ]
        .first()
        .astype(
            int
        )
    )

    expected_target_arrays = [
        (
            np.arange(
                daily_size,
                dtype=float,
            )
            / (
                daily_size
                - 1
            )
        )
        for daily_size in daily_sizes
        if daily_size > 1
    ]

    if not expected_target_arrays:
        raise ValueError(
            "No valid daily cross-sections were available "
            "for the discrete rank-grid expectation."
        )

    expected_target_values = np.concatenate(
        expected_target_arrays
    )

    bin_edges = np.linspace(
        0.0,
        1.0,
        NUMBER_OF_BINS + 1,
    )

    observed_counts, _ = np.histogram(
        target_panel[
            "Target_Y"
        ],
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

    # --------------------------------------------------------
    # 4.6 Thesis-ready target-distribution figure
    # --------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    bar_width = (
        bin_edges[1]
        - bin_edges[0]
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
        label=(
            "No-tie discrete "
            "rank-grid expectation"
        ),
    )

    axis.set_title(
        "Empirical Distribution of the "
        "Cross-Sectional Percentile Target"
    )

    axis.set_xlabel(
        "Realised cross-sectional percentile target"
    )

    axis.set_ylabel(
        "Number of stock-day observations"
    )

    axis.set_xlim(
        0,
        1,
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.45,
    )

    axis.legend()

    annotation_text = (
        f"{days_with_ties:,} of {total_dates:,} dates "
        "contained tied returns\n"
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

    plt.close(
        figure
    )

    # --------------------------------------------------------
    # 4.7 Terminal output
    # --------------------------------------------------------

    print(
        f"Input file: "
        f"{relative_path(INPUT_FILE)}"
    )

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
        f"{target_panel['Date'].min().date()} to "
        f"{target_panel['Date'].max().date()}"
    )

    print(
        "Daily cross-sectional size: "
        f"{minimum_cross_section} to "
        f"{maximum_cross_section}"
    )

    print(
        f"Target minimum: "
        f"{target_panel['Target_Y'].min():.4f}"
    )

    print(
        f"Target maximum: "
        f"{target_panel['Target_Y'].max():.4f}"
    )

    print(
        f"Target mean: "
        f"{target_panel['Target_Y'].mean():.4f}"
    )

    print(
        "Target standard deviation: "
        f"{target_panel['Target_Y'].std():.6f}"
    )

    print(
        "Minimum forward return: "
        f"{target_panel['Forward_Return'].min():.6f}"
    )

    print(
        "Maximum forward return: "
        f"{target_panel['Forward_Return'].max():.6f}"
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

    print(
        "\nIntegrity checks passed:"
    )

    print(
        "- No duplicate stock-date observations."
    )

    print(
        "- No missing or non-finite required target fields."
    )

    print(
        "- All target values lie within [0, 1]."
    )

    print(
        "- Every daily stored N_t matches the observed "
        "cross-sectional size."
    )

    print(
        "- Every daily target mean equals 0.5."
    )

    print(
        "- Every daily cross-section contains target values 0 and 1."
    )

    print(
        "- The empirical distribution was compared with the "
        "appropriate discrete daily rank-grid expectation."
    )

    print(
        "\nOutput files:"
    )

    print(
        f"- {relative_path(OUTPUT_FIGURE)}"
    )

    print(
        f"- {relative_path(OUTPUT_SUMMARY)}"
    )

    print(
        f"- {relative_path(OUTPUT_FREQUENCIES)}"
    )

    print("=" * 74)


if __name__ == "__main__":
    main()