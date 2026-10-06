"""
Script 02: Point-in-Time Liquidity Proxy and Active-Universe Construction
-------------------------------------------------------------------------

Purpose
-------
Constructs a dynamic liquidity-screened proxy for the investable JSE
large-cap universe without using information from the quarter in which the
selected securities are traded.

The script:

1. removes market-wide non-trading dates;
2. calculates nominal daily value traded using unadjusted Close × Volume;
3. calculates a strict 63-trading-day rolling average for each security;
4. ranks eligible securities at each calendar-quarter end;
5. applies the quarter-end selection only to the subsequent quarter; and
6. creates auditable membership, ranking, replacement and daily-panel outputs.

Important
---------
The resulting universe is a researcher-constructed liquidity proxy. It is
not an exact historical reconstruction of the official FTSE/JSE Top 40 index.

Input
-----
data/raw/JSE_Broad_Master_Panel.csv

Data outputs
------------
data/interim/02_universe/JSE_Broad_Trading_Panel.csv
data/interim/02_universe/JSE_Active_Universe_Membership.csv
data/interim/02_universe/JSE_Active_Universe_Panel.csv

Audit outputs
-------------
outputs/audits/02_universe/JSE_Market_Date_Audit.csv
outputs/audits/02_universe/JSE_Removed_Non_Trading_Dates.csv
outputs/audits/02_universe/JSE_Ticker_Trading_Coverage_Audit.csv
outputs/audits/02_universe/JSE_Quarterly_Liquidity_Rankings.csv
outputs/audits/02_universe/JSE_Quarterly_Universe_Turnover.csv
outputs/audits/02_universe/JSE_Active_Universe_Daily_Counts.csv

Figure output
-------------
outputs/figures/chapter_4/figure_4_1_quarterly_universe_turnover.png

Run from the project root
-------------------------
python .\\scripts\\02_Dynamic_Proxy_Construction.py
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
    / "raw"
    / "JSE_Broad_Master_Panel.csv"
)

DATA_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "02_universe"
)

AUDIT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "audits"
    / "02_universe"
)

FIGURE_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "chapter_4"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

TOP_N = 40

ROLLING_WINDOW = 63

MIN_ROLLING_OBSERVATIONS = 63

# A date is treated as a market-wide non-trading date when fewer
# than 20% of observed securities have positive reported volume.
#
# In the current extraction, all dates removed by this rule have
# zero positive-volume securities.
MIN_POSITIVE_VOLUME_SHARE = 0.20

# A security is excluded from a quarter-end ranking when its
# latest available record is more than five retained market
# trading dates before the overall market quarter-end date.
MAX_STALENESS_TRADING_DAYS = 5

REQUIRED_COLUMNS = {
    "Date",
    "Ticker",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
}


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def validate_input(panel: pd.DataFrame) -> None:
    """Validate the broad panel before liquidity construction."""

    missing_columns = REQUIRED_COLUMNS.difference(
        panel.columns
    )

    if missing_columns:
        raise ValueError(
            "The broad panel is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if panel.empty:
        raise ValueError(
            "The broad panel contains no observations."
        )

    duplicate_count = int(
        panel.duplicated(
            ["Date", "Ticker"]
        ).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"The broad panel contains {duplicate_count:,} duplicate "
            "Date-Ticker observations. Resolve them before running "
            "Script 02."
        )

    if (
        panel["Date"].isna().any()
        or panel["Ticker"].isna().any()
    ):
        raise ValueError(
            "Date and Ticker must not contain missing values."
        )

    if (
        panel["Close"].isna().any()
        or panel["Volume"].isna().any()
    ):
        raise ValueError(
            "Close and Volume contain missing values. Resolve them "
            "before constructing the liquidity proxy."
        )

    if panel["Close"].le(0).any():
        raise ValueError(
            "Close must be strictly positive for the "
            "value-traded calculation."
        )

    if panel["Volume"].lt(0).any():
        raise ValueError(
            "Volume must not be negative."
        )


def build_market_date_audit(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Determine which dates represent genuine market trading sessions.

    Isolated zero-volume security observations are retained. Only dates
    with extremely low market-wide positive-volume participation are
    removed.
    """

    audit = (
        panel.assign(
            Positive_Volume=panel["Volume"].gt(0)
        )
        .groupby(
            "Date",
            as_index=False,
        )
        .agg(
            Observed_Securities=(
                "Ticker",
                "nunique",
            ),
            Positive_Volume_Securities=(
                "Positive_Volume",
                "sum",
            ),
            Total_Reported_Volume=(
                "Volume",
                "sum",
            ),
        )
        .sort_values(
            "Date"
        )
        .reset_index(
            drop=True
        )
    )

    audit["Positive_Volume_Share"] = (
        audit["Positive_Volume_Securities"]
        / audit["Observed_Securities"]
    )

    audit["Is_Market_Trading_Date"] = (
        audit["Positive_Volume_Share"].ge(
            MIN_POSITIVE_VOLUME_SHARE
        )
        & audit["Total_Reported_Volume"].gt(0)
    )

    audit["Removal_Reason"] = np.where(
        audit["Is_Market_Trading_Date"],
        "Retained market trading date",
        (
            "Market-wide non-trading date or insufficient "
            "positive-volume participation"
        ),
    )

    return audit


def build_ticker_coverage_audit(
    trading_panel: pd.DataFrame,
    market_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Confirm that each ticker has no missing retained market dates
    between its first and last observed trading dates.
    """

    records: list[dict[str, object]] = []

    for ticker, group in trading_panel.groupby(
        "Ticker",
        sort=True,
    ):
        first_date = group["Date"].min()
        last_date = group["Date"].max()

        expected_dates = market_dates[
            (market_dates >= first_date)
            & (market_dates <= last_date)
        ]

        observed_dates = pd.DatetimeIndex(
            group["Date"].unique()
        )

        missing_dates = expected_dates.difference(
            observed_dates
        )

        records.append(
            {
                "Ticker": ticker,
                "First_Trading_Date": first_date,
                "Last_Trading_Date": last_date,
                "Observed_Trading_Dates": len(
                    observed_dates
                ),
                "Expected_Trading_Dates_Within_Coverage": len(
                    expected_dates
                ),
                "Missing_Trading_Dates_Within_Coverage": len(
                    missing_dates
                ),
                "First_Missing_Date": (
                    missing_dates.min()
                    if len(missing_dates) > 0
                    else pd.NaT
                ),
            }
        )

    return (
        pd.DataFrame(
            records
        )
        .sort_values(
            "Ticker"
        )
        .reset_index(
            drop=True
        )
    )


def period_columns_to_string(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Convert Period columns to strings before writing CSV files."""

    output = dataframe.copy()

    for column in columns:
        if column in output.columns:
            output[column] = output[column].astype(
                str
            )

    return output


def relative_path(path: Path) -> str:
    """Return a project-relative path for readable terminal output."""

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
    print("SCRIPT 02: POINT-IN-TIME LIQUIDITY PROXY CONSTRUCTION")
    print("=" * 74)

    DATA_OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Input file not found:\n"
            f"{INPUT_FILE.resolve()}"
        )

    # --------------------------------------------------------
    # 4.1 Load and validate the broad panel
    # --------------------------------------------------------

    panel = pd.read_csv(
        INPUT_FILE
    )

    panel["Date"] = pd.to_datetime(
        panel["Date"],
        errors="raise",
    )

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]

    for column in numeric_columns:
        panel[column] = pd.to_numeric(
            panel[column],
            errors="raise",
        )

    panel = (
        panel.sort_values(
            ["Date", "Ticker"]
        )
        .reset_index(
            drop=True
        )
    )

    validate_input(
        panel
    )

    raw_row_count = len(
        panel
    )

    raw_date_count = panel[
        "Date"
    ].nunique()

    # --------------------------------------------------------
    # 4.2 Remove only market-wide non-trading dates
    # --------------------------------------------------------

    market_date_audit = build_market_date_audit(
        panel
    )

    retained_market_dates = pd.DatetimeIndex(
        market_date_audit.loc[
            market_date_audit[
                "Is_Market_Trading_Date"
            ],
            "Date",
        ].sort_values()
    )

    removed_dates = market_date_audit.loc[
        ~market_date_audit[
            "Is_Market_Trading_Date"
        ]
    ].copy()

    trading_panel = panel.loc[
        panel["Date"].isin(
            retained_market_dates
        )
    ].copy()

    trading_panel = (
        trading_panel.sort_values(
            ["Ticker", "Date"]
        )
        .reset_index(
            drop=True
        )
    )

    coverage_audit = build_ticker_coverage_audit(
        trading_panel=trading_panel,
        market_dates=retained_market_dates,
    )

    missing_internal_dates = int(
        coverage_audit[
            "Missing_Trading_Dates_Within_Coverage"
        ].sum()
    )

    if missing_internal_dates > 0:
        raise ValueError(
            "At least one ticker has missing retained market dates "
            "inside its observed coverage period. Review "
            "outputs/audits/02_universe/"
            "JSE_Ticker_Trading_Coverage_Audit.csv."
        )

    # --------------------------------------------------------
    # 4.3 Construct daily and rolling liquidity measures
    # --------------------------------------------------------

    # Unadjusted Close is used intentionally for nominal value traded.
    # Adjusted Close is used in later scripts for returns and features.
    trading_panel["Value_Traded"] = (
        trading_panel["Close"]
        * trading_panel["Volume"]
    )

    trading_panel["Rolling_VT_63D"] = (
        trading_panel
        .groupby(
            "Ticker",
            sort=False,
        )["Value_Traded"]
        .transform(
            lambda series: series.rolling(
                window=ROLLING_WINDOW,
                min_periods=MIN_ROLLING_OBSERVATIONS,
            ).mean()
        )
    )

    trading_panel["Rolling_Observation_Count"] = (
        trading_panel
        .groupby(
            "Ticker",
            sort=False,
        )["Value_Traded"]
        .transform(
            lambda series: series.rolling(
                window=ROLLING_WINDOW,
                min_periods=1,
            ).count()
        )
    )

    trading_panel["Selection_Quarter"] = (
        trading_panel[
            "Date"
        ].dt.to_period(
            "Q"
        )
    )

    # --------------------------------------------------------
    # 4.4 Create quarter-end candidate snapshots
    # --------------------------------------------------------

    market_quarter_ends = (
        market_date_audit.loc[
            market_date_audit[
                "Is_Market_Trading_Date"
            ],
            ["Date"],
        ]
        .assign(
            Selection_Quarter=lambda data: (
                data["Date"].dt.to_period(
                    "Q"
                )
            )
        )
        .groupby(
            "Selection_Quarter",
            as_index=False,
        )
        .agg(
            Market_Quarter_End=(
                "Date",
                "max",
            )
        )
    )

    quarter_end_candidates = (
        trading_panel
        .sort_values(
            ["Ticker", "Date"]
        )
        .groupby(
            [
                "Ticker",
                "Selection_Quarter",
            ],
            as_index=False,
        )
        .tail(
            1
        )
        .copy()
    )

    quarter_end_candidates = (
        quarter_end_candidates.merge(
            market_quarter_ends,
            on="Selection_Quarter",
            how="left",
            validate="many_to_one",
        )
    )

    market_date_position = {
        date: position
        for position, date in enumerate(
            retained_market_dates
        )
    }

    quarter_end_candidates[
        "Staleness_Trading_Days"
    ] = [
        (
            market_date_position[
                market_end
            ]
            - market_date_position[
                ticker_end
            ]
        )
        for ticker_end, market_end in zip(
            quarter_end_candidates[
                "Date"
            ],
            quarter_end_candidates[
                "Market_Quarter_End"
            ],
        )
    ]

    quarter_end_candidates[
        "Eligible_For_Ranking"
    ] = (
        quarter_end_candidates[
            "Rolling_VT_63D"
        ].notna()
        & quarter_end_candidates[
            "Rolling_VT_63D"
        ].gt(0)
        & quarter_end_candidates[
            "Rolling_Observation_Count"
        ].ge(
            MIN_ROLLING_OBSERVATIONS
        )
        & quarter_end_candidates[
            "Staleness_Trading_Days"
        ].le(
            MAX_STALENESS_TRADING_DAYS
        )
    )

    # --------------------------------------------------------
    # 4.5 Rank securities and select the subsequent quarter
    # --------------------------------------------------------

    eligible_candidates = quarter_end_candidates.loc[
        quarter_end_candidates[
            "Eligible_For_Ranking"
        ]
    ].copy()

    # Deterministic ranking:
    # 1. highest rolling value traded receives the best rank;
    # 2. ticker alphabetically breaks exact numerical ties.
    eligible_candidates = (
        eligible_candidates.sort_values(
            [
                "Selection_Quarter",
                "Rolling_VT_63D",
                "Ticker",
            ],
            ascending=[
                True,
                False,
                True,
            ],
            kind="mergesort",
        )
    )

    eligible_candidates[
        "Liquidity_Rank"
    ] = (
        eligible_candidates
        .groupby(
            "Selection_Quarter"
        )
        .cumcount()
        .add(
            1
        )
    )

    eligible_candidates[
        "Selected_For_Next_Quarter"
    ] = eligible_candidates[
        "Liquidity_Rank"
    ].le(
        TOP_N
    )

    ranking_columns = [
        "Ticker",
        "Selection_Quarter",
        "Liquidity_Rank",
        "Selected_For_Next_Quarter",
    ]

    quarterly_rankings = (
        quarter_end_candidates.merge(
            eligible_candidates[
                ranking_columns
            ],
            on=[
                "Ticker",
                "Selection_Quarter",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    quarterly_rankings[
        "Selected_For_Next_Quarter"
    ] = quarterly_rankings[
        "Selected_For_Next_Quarter"
    ].eq(
        True
    )

    selected = eligible_candidates.loc[
        eligible_candidates[
            "Selected_For_Next_Quarter"
        ]
    ].copy()

    # Information measured at the end of quarter q determines
    # active membership only during quarter q + 1.
    selected["Active_Quarter"] = (
        selected["Selection_Quarter"]
        + 1
    )

    membership = selected[
        [
            "Selection_Quarter",
            "Date",
            "Market_Quarter_End",
            "Active_Quarter",
            "Ticker",
            "Rolling_VT_63D",
            "Liquidity_Rank",
            "Staleness_Trading_Days",
        ]
    ].copy()

    membership = membership.rename(
        columns={
            "Date": "Ticker_Selection_Date",
            "Rolling_VT_63D": (
                "Selection_Rolling_VT_63D"
            ),
        }
    )

    membership[
        "In_Active_Universe"
    ] = True

    membership = (
        membership.sort_values(
            [
                "Active_Quarter",
                "Liquidity_Rank",
                "Ticker",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # 4.6 Identify entries and retained members
    # --------------------------------------------------------

    membership_status: list[str] = []

    previous_members: set[str] = set()

    for _, quarter_group in membership.groupby(
        "Active_Quarter",
        sort=True,
    ):
        current_members = set(
            quarter_group[
                "Ticker"
            ]
        )

        for ticker in quarter_group[
            "Ticker"
        ]:
            if not previous_members:
                membership_status.append(
                    "Initial member"
                )
            elif ticker in previous_members:
                membership_status.append(
                    "Retained"
                )
            else:
                membership_status.append(
                    "Entry"
                )

        previous_members = current_members

    membership[
        "Membership_Status"
    ] = membership_status

    # --------------------------------------------------------
    # 4.7 Merge lagged membership onto daily observations
    # --------------------------------------------------------

    trading_panel[
        "Active_Quarter"
    ] = trading_panel[
        "Date"
    ].dt.to_period(
        "Q"
    )

    active_panel = trading_panel.merge(
        membership[
            [
                "Active_Quarter",
                "Ticker",
                "Selection_Quarter",
                "Market_Quarter_End",
                "Selection_Rolling_VT_63D",
                "Liquidity_Rank",
                "Membership_Status",
                "In_Active_Universe",
            ]
        ],
        on=[
            "Active_Quarter",
            "Ticker",
        ],
        how="inner",
        validate="many_to_one",
    )

    active_panel = (
        active_panel.sort_values(
            ["Date", "Ticker"]
        )
        .reset_index(
            drop=True
        )
    )

    daily_counts = (
        active_panel
        .groupby(
            "Date",
            as_index=False,
        )
        .agg(
            Active_Securities=(
                "Ticker",
                "nunique",
            ),
            Active_Quarter=(
                "Active_Quarter",
                "first",
            ),
        )
        .sort_values(
            "Date"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # 4.8 Construct quarterly membership-change diagnostics
    # --------------------------------------------------------

    turnover_records: list[dict[str, object]] = []

    previous_members = set()

    observed_active_quarters = set(
        active_panel[
            "Active_Quarter"
        ].unique()
    )

    for active_quarter, quarter_group in membership.groupby(
        "Active_Quarter",
        sort=True,
    ):
        current_members = set(
            quarter_group[
                "Ticker"
            ]
        )

        if previous_members:
            entries = current_members.difference(
                previous_members
            )

            exits = previous_members.difference(
                current_members
            )

            replacement_count = min(
                len(entries),
                len(exits),
            )
        else:
            entries = current_members
            exits = set()
            replacement_count = 0

        turnover_records.append(
            {
                "Active_Quarter": active_quarter,
                "Observed_In_Sample": (
                    active_quarter
                    in observed_active_quarters
                ),
                "Constituent_Count": len(
                    current_members
                ),
                "Entry_Count": len(
                    entries
                ),
                "Exit_Count": len(
                    exits
                ),
                "Replacement_Count": (
                    replacement_count
                ),
                "Entries": "; ".join(
                    sorted(
                        entries
                    )
                ),
                "Exits": "; ".join(
                    sorted(
                        exits
                    )
                ),
            }
        )

        previous_members = current_members

    quarterly_turnover = pd.DataFrame(
        turnover_records
    )

    # --------------------------------------------------------
    # 4.9 Final validity checks
    # --------------------------------------------------------

    leakage_check = (
        membership[
            "Active_Quarter"
        ]
        == membership[
            "Selection_Quarter"
        ]
        + 1
    )

    if not leakage_check.all():
        raise AssertionError(
            "At least one membership record was applied "
            "to the wrong quarter."
        )

    if active_panel.duplicated(
        ["Date", "Ticker"]
    ).any():
        raise AssertionError(
            "Duplicate Date-Ticker observations were created "
            "in the active panel."
        )

    membership_counts = (
        membership
        .groupby(
            "Active_Quarter"
        )["Ticker"]
        .nunique()
    )

    quarters_below_top_n = membership_counts.loc[
        membership_counts.lt(
            TOP_N
        )
    ]

    if not quarters_below_top_n.empty:
        print(
            "\nWARNING: Some scheduled active quarters contain "
            f"fewer than {TOP_N} securities because fewer eligible "
            "candidates were available at the preceding quarter-end."
        )

        print(
            quarters_below_top_n.to_string()
        )

    # --------------------------------------------------------
    # 4.10 Define output paths
    # --------------------------------------------------------

    output_paths = {
        "broad_trading_panel": (
            DATA_OUTPUT_DIRECTORY
            / "JSE_Broad_Trading_Panel.csv"
        ),
        "membership": (
            DATA_OUTPUT_DIRECTORY
            / "JSE_Active_Universe_Membership.csv"
        ),
        "active_panel": (
            DATA_OUTPUT_DIRECTORY
            / "JSE_Active_Universe_Panel.csv"
        ),
        "market_date_audit": (
            AUDIT_OUTPUT_DIRECTORY
            / "JSE_Market_Date_Audit.csv"
        ),
        "removed_dates": (
            AUDIT_OUTPUT_DIRECTORY
            / "JSE_Removed_Non_Trading_Dates.csv"
        ),
        "coverage_audit": (
            AUDIT_OUTPUT_DIRECTORY
            / "JSE_Ticker_Trading_Coverage_Audit.csv"
        ),
        "quarterly_rankings": (
            AUDIT_OUTPUT_DIRECTORY
            / "JSE_Quarterly_Liquidity_Rankings.csv"
        ),
        "turnover": (
            AUDIT_OUTPUT_DIRECTORY
            / "JSE_Quarterly_Universe_Turnover.csv"
        ),
        "daily_counts": (
            AUDIT_OUTPUT_DIRECTORY
            / "JSE_Active_Universe_Daily_Counts.csv"
        ),
        "turnover_figure": (
            FIGURE_OUTPUT_DIRECTORY
            / "figure_4_1_quarterly_universe_turnover.png"
        ),
    }

    period_columns = [
        "Selection_Quarter",
        "Active_Quarter",
    ]

    # --------------------------------------------------------
    # 4.11 Save outputs
    # --------------------------------------------------------

    period_columns_to_string(
        trading_panel,
        period_columns,
    ).to_csv(
        output_paths[
            "broad_trading_panel"
        ],
        index=False,
    )

    period_columns_to_string(
        membership,
        period_columns,
    ).to_csv(
        output_paths[
            "membership"
        ],
        index=False,
    )

    period_columns_to_string(
        active_panel,
        period_columns,
    ).to_csv(
        output_paths[
            "active_panel"
        ],
        index=False,
    )

    market_date_audit.to_csv(
        output_paths[
            "market_date_audit"
        ],
        index=False,
    )

    removed_dates.to_csv(
        output_paths[
            "removed_dates"
        ],
        index=False,
    )

    coverage_audit.to_csv(
        output_paths[
            "coverage_audit"
        ],
        index=False,
    )

    period_columns_to_string(
        quarterly_rankings,
        period_columns,
    ).to_csv(
        output_paths[
            "quarterly_rankings"
        ],
        index=False,
    )

    period_columns_to_string(
        quarterly_turnover,
        ["Active_Quarter"],
    ).to_csv(
        output_paths[
            "turnover"
        ],
        index=False,
    )

    period_columns_to_string(
        daily_counts,
        ["Active_Quarter"],
    ).to_csv(
        output_paths[
            "daily_counts"
        ],
        index=False,
    )

    # --------------------------------------------------------
    # 4.12 Generate quarterly membership-replacement figure
    # --------------------------------------------------------

    plot_data = quarterly_turnover.loc[
        quarterly_turnover[
            "Observed_In_Sample"
        ]
    ].copy()

    # Exclude initial formation because it is not a replacement event.
    plot_data = plot_data.iloc[
        1:
    ].copy()

    plot_data = period_columns_to_string(
        plot_data,
        ["Active_Quarter"],
    )

    figure, axis = plt.subplots(
        figsize=(
            12,
            5,
        )
    )

    axis.plot(
        plot_data[
            "Active_Quarter"
        ],
        plot_data[
            "Replacement_Count"
        ],
        marker="o",
        label="Securities replaced",
    )

    axis.set_title(
        "Quarterly Membership Replacements in the "
        "Liquidity-Screened Active Universe"
    )

    axis.set_xlabel(
        "Observed Active Quarter"
    )

    axis.set_ylabel(
        "Number of Securities Replaced"
    )

    axis.tick_params(
        axis="x",
        rotation=90,
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_paths[
            "turnover_figure"
        ],
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------
    # 4.13 Thesis-ready diagnostics
    # --------------------------------------------------------

    observed_quarter_count = active_panel[
        "Active_Quarter"
    ].nunique()

    scheduled_quarter_count = membership[
        "Active_Quarter"
    ].nunique()

    observed_turnover = quarterly_turnover.loc[
        quarterly_turnover[
            "Observed_In_Sample"
        ]
    ].copy()

    average_replacements = (
        observed_turnover[
            "Replacement_Count"
        ]
        .iloc[
            1:
        ]
        .mean()
    )

    print(
        "\n"
        + "=" * 74
    )

    print(
        "SCRIPT 02 EMPIRICAL OUTPUTS"
    )

    print(
        "=" * 74
    )

    print(
        f"Input file: "
        f"{relative_path(INPUT_FILE)}"
    )

    print(
        f"Raw broad-panel rows: "
        f"{raw_row_count:,}"
    )

    print(
        f"Raw distinct dates: "
        f"{raw_date_count:,}"
    )

    print(
        "Removed market-wide non-trading dates: "
        f"{len(removed_dates):,}"
    )

    print(
        "Rows removed with those dates: "
        f"{raw_row_count - len(trading_panel):,}"
    )

    print(
        "Retained broad trading-panel rows: "
        f"{len(trading_panel):,}"
    )

    print(
        "Retained market trading dates: "
        f"{trading_panel['Date'].nunique():,}"
    )

    print(
        "Ticker-internal missing trading dates after cleaning: "
        f"{missing_internal_dates:,}"
    )

    print(
        "Quarter-end selections with at least one eligible security: "
        f"{selected['Selection_Quarter'].nunique():,}"
    )

    print(
        "Scheduled active-universe quarters: "
        f"{scheduled_quarter_count:,}"
    )

    print(
        "Observed active-universe quarters in the available sample: "
        f"{observed_quarter_count:,}"
    )

    print(
        "Active-universe panel rows: "
        f"{len(active_panel):,}"
    )

    print(
        "Active-universe trading dates: "
        f"{active_panel['Date'].nunique():,}"
    )

    if not active_panel.empty:
        print(
            "Active-panel start date: "
            f"{active_panel['Date'].min().date()}"
        )

        print(
            "Active-panel end date: "
            f"{active_panel['Date'].max().date()}"
        )

    print(
        "Average selected securities per scheduled active quarter: "
        f"{membership_counts.mean():.2f}"
    )

    print(
        "Minimum/maximum selected securities per scheduled quarter: "
        f"{membership_counts.min():,}/"
        f"{membership_counts.max():,}"
    )

    print(
        "Average active securities per observed trading day: "
        f"{daily_counts['Active_Securities'].mean():.2f}"
    )

    print(
        "Minimum/maximum active securities per observed trading day: "
        f"{daily_counts['Active_Securities'].min():,}/"
        f"{daily_counts['Active_Securities'].max():,}"
    )

    print(
        "Average quarterly membership replacements after "
        "initial formation: "
        f"{average_replacements:.2f}"
    )

    print(
        "Look-ahead check passed: every selection quarter "
        "is applied only to the subsequent active quarter."
    )

    print(
        "\nOutput files:"
    )

    for output_path in output_paths.values():
        print(
            f"- {relative_path(output_path)}"
        )

    print(
        "=" * 74
    )


if __name__ == "__main__":
    main()