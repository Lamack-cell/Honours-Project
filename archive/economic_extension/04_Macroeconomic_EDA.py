"""
Script 04: SARB Risk-Free-Rate Processing and Macroeconomic EDA
---------------------------------------------------------------

Purpose
-------
Cleans the South African Reserve Bank 91-Day Treasury Bill Tender Rate,
aligns it to the equity market's actual trading dates using only historically
available observations, converts the annual percentage rate into a daily
decimal rate, and produces a thesis-ready time-series figure.

The daily alignment uses a backward-looking merge:

    market date t receives the most recent Treasury Bill observation
    dated on or before t.

This avoids assigning future macroeconomic observations to earlier dates.

Inputs
------
1. 91-TB_data.xlsx
2. JSE_Broad_Trading_Panel.csv

Outputs
-------
1. JSE_91Day_TB_Weekly_Clean.csv
2. JSE_Risk_Free_Daily.csv
3. outputs/audits/JSE_Risk_Free_Processing_Summary.csv
4. outputs/audits/JSE_Risk_Free_Date_Alignment_Audit.csv
5. outputs/figures/figure_4_3_risk_free_rate.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_MACRO_FILE = Path("91-TB_data.xlsx")
INPUT_MARKET_PANEL = Path("JSE_Broad_Trading_Panel.csv")

OUTPUT_WEEKLY_CLEAN = Path("JSE_91Day_TB_Weekly_Clean.csv")
OUTPUT_DAILY_RISK_FREE = Path("JSE_Risk_Free_Daily.csv")

AUDIT_DIRECTORY = Path("outputs") / "audits"
FIGURE_DIRECTORY = Path("outputs") / "figures"

OUTPUT_PROCESSING_SUMMARY = (
    AUDIT_DIRECTORY
    / "JSE_Risk_Free_Processing_Summary.csv"
)

OUTPUT_ALIGNMENT_AUDIT = (
    AUDIT_DIRECTORY
    / "JSE_Risk_Free_Date_Alignment_Audit.csv"
)

OUTPUT_FIGURE = (
    FIGURE_DIRECTORY
    / "figure_4_3_risk_free_rate.png"
)

TRADING_DAYS_PER_YEAR = 252


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def identify_date_column(columns: list[str]) -> str:
    """
    Identify the observation-date column using common naming conventions.
    """

    normalised = {
        str(column).strip().lower(): column
        for column in columns
    }

    preferred_names = [
        "date",
        "observation date",
        "observation_date",
        "period",
    ]

    for candidate in preferred_names:
        if candidate in normalised:
            return normalised[candidate]

    date_candidates = [
        original
        for cleaned, original in normalised.items()
        if "date" in cleaned
    ]

    if len(date_candidates) == 1:
        return date_candidates[0]

    raise ValueError(
        "Unable to identify the macroeconomic date column. "
        f"Available columns: {columns}"
    )


def identify_rate_column(
    columns: list[str],
    date_column: str,
) -> str:
    """
    Identify the 91-Day Treasury Bill rate column.
    """

    non_date_columns = [
        column
        for column in columns
        if column != date_column
    ]

    normalised = {
        str(column)
        .strip()
        .upper()
        .replace("_", " ")
        .replace("-", " "): column
        for column in non_date_columns
    }

    preferred_names = [
        "91 DAY TB",
        "91 DAY T B",
        "91 DAY TREASURY BILL",
        "91 DAY TREASURY BILL TENDER RATE",
        "KBP1574",
    ]

    for candidate in preferred_names:
        if candidate in normalised:
            return normalised[candidate]

    candidates = []

    for cleaned, original in normalised.items():
        contains_91 = "91" in cleaned
        contains_treasury_reference = (
            "TB" in cleaned
            or "TREASURY" in cleaned
            or "BILL" in cleaned
        )

        if contains_91 and contains_treasury_reference:
            candidates.append(original)

    if len(candidates) == 1:
        return candidates[0]

    if len(non_date_columns) == 1:
        return non_date_columns[0]

    raise ValueError(
        "Unable to identify the 91-Day Treasury Bill rate column. "
        f"Available columns: {columns}"
    )


def validate_market_dates(
    market_panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract and validate the unique equity-market trading calendar.
    """

    if "Date" not in market_panel.columns:
        raise ValueError(
            "JSE_Broad_Trading_Panel.csv does not contain a Date column."
        )

    market_panel["Date"] = pd.to_datetime(
        market_panel["Date"],
        errors="raise",
    )

    market_dates = (
        market_panel[["Date"]]
        .drop_duplicates()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if market_dates.empty:
        raise ValueError(
            "No market trading dates were found."
        )

    if market_dates["Date"].duplicated().any():
        raise AssertionError(
            "Duplicate dates remain in the market calendar."
        )

    return market_dates


# ============================================================
# 3. MAIN PIPELINE
# ============================================================

def main() -> None:
    print("=" * 74)
    print("SCRIPT 04: SARB RISK-FREE-RATE PROCESSING")
    print("=" * 74)

    AUDIT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_files = [
        INPUT_MACRO_FILE,
        INPUT_MARKET_PANEL,
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

    # --------------------------------------------------------
    # 3.1 Load the raw weekly Treasury Bill series
    # --------------------------------------------------------

    raw_macro = pd.read_excel(
        INPUT_MACRO_FILE,
        sheet_name=0,
        engine="openpyxl",
    )

    raw_row_count = len(raw_macro)

    if raw_macro.empty:
        raise ValueError(
            "The SARB workbook contains no observations."
        )

    date_column = identify_date_column(
        raw_macro.columns.tolist()
    )

    rate_column = identify_rate_column(
        raw_macro.columns.tolist(),
        date_column=date_column,
    )

    macro = raw_macro[
        [date_column, rate_column]
    ].copy()

    macro = macro.rename(
        columns={
            date_column: "Observation_Date",
            rate_column: "Risk_Free_Rate_Annual_Percent",
        }
    )

    macro["Observation_Date"] = pd.to_datetime(
        macro["Observation_Date"],
        errors="coerce",
    )

    macro[
        "Risk_Free_Rate_Annual_Percent"
    ] = pd.to_numeric(
        macro["Risk_Free_Rate_Annual_Percent"],
        errors="coerce",
    )

    missing_date_rows = int(
        macro["Observation_Date"].isna().sum()
    )

    missing_rate_rows = int(
        macro[
            "Risk_Free_Rate_Annual_Percent"
        ].isna().sum()
    )

    macro = macro.dropna(
        subset=[
            "Observation_Date",
            "Risk_Free_Rate_Annual_Percent",
        ]
    ).copy()

    if macro.empty:
        raise ValueError(
            "No valid SARB observations remain after parsing."
        )

    # --------------------------------------------------------
    # 3.2 Validate rate levels and duplicate dates
    # --------------------------------------------------------

    if (
        macro["Risk_Free_Rate_Annual_Percent"] < 0
    ).any():
        negative_rows = macro.loc[
            macro[
                "Risk_Free_Rate_Annual_Percent"
            ] < 0
        ]

        raise ValueError(
            "Negative Treasury Bill rates were detected. "
            f"Review {len(negative_rows):,} observations."
        )

    if (
        macro["Risk_Free_Rate_Annual_Percent"] > 100
    ).any():
        raise ValueError(
            "Treasury Bill rates above 100% were detected. "
            "The rate column may have been identified incorrectly."
        )

    duplicate_date_rows = int(
        macro.duplicated(
            subset=["Observation_Date"],
            keep=False,
        ).sum()
    )

    if duplicate_date_rows > 0:
        duplicate_conflicts = (
            macro.groupby("Observation_Date")[
                "Risk_Free_Rate_Annual_Percent"
            ]
            .nunique()
        )

        conflicting_dates = duplicate_conflicts.loc[
            duplicate_conflicts.gt(1)
        ]

        if not conflicting_dates.empty:
            raise ValueError(
                "Conflicting Treasury Bill rates exist for the same "
                "observation date:\n"
                f"{conflicting_dates.to_string()}"
            )

        # Identical duplicate observations are harmless and can be reduced.
        macro = macro.drop_duplicates(
            subset=["Observation_Date"],
            keep="last",
        )

    macro = (
        macro.sort_values("Observation_Date")
        .reset_index(drop=True)
    )

    if macro["Observation_Date"].duplicated().any():
        raise AssertionError(
            "Duplicate Treasury Bill dates remain after cleaning."
        )

    # Convert annual percentage to decimal form.
    macro["Risk_Free_Rate_Annual_Decimal"] = (
        macro["Risk_Free_Rate_Annual_Percent"]
        / 100.0
    )

    # Daily effective rate, assuming 252 market trading days per year.
    macro["Risk_Free_Rate_Daily"] = (
        1.0
        + macro["Risk_Free_Rate_Annual_Decimal"]
    ) ** (
        1.0 / TRADING_DAYS_PER_YEAR
    ) - 1.0

    macro.to_csv(
        OUTPUT_WEEKLY_CLEAN,
        index=False,
    )

    # --------------------------------------------------------
    # 3.3 Construct the equity-market trading calendar
    # --------------------------------------------------------

    market_panel = pd.read_csv(
        INPUT_MARKET_PANEL,
        usecols=["Date"],
    )

    market_dates = validate_market_dates(
        market_panel
    )

    # --------------------------------------------------------
    # 3.4 Backward-align rates to market trading dates
    # --------------------------------------------------------

    daily_risk_free = pd.merge_asof(
        market_dates.sort_values("Date"),
        macro.sort_values("Observation_Date"),
        left_on="Date",
        right_on="Observation_Date",
        direction="backward",
        allow_exact_matches=True,
    )

    daily_risk_free[
        "Rate_Observation_Age_Calendar_Days"
    ] = (
        daily_risk_free["Date"]
        - daily_risk_free["Observation_Date"]
    ).dt.days

    unmatched_market_dates = daily_risk_free.loc[
        daily_risk_free[
            "Risk_Free_Rate_Annual_Percent"
        ].isna()
    ].copy()

    if not unmatched_market_dates.empty:
        unmatched_market_dates.to_csv(
            AUDIT_DIRECTORY
            / "JSE_Unmatched_Initial_Risk_Free_Dates.csv",
            index=False,
        )

        raise ValueError(
            f"{len(unmatched_market_dates):,} market dates occur before "
            "the first available Treasury Bill observation. Review "
            "outputs/audits/JSE_Unmatched_Initial_Risk_Free_Dates.csv."
        )

    if (
        daily_risk_free["Observation_Date"]
        > daily_risk_free["Date"]
    ).any():
        raise AssertionError(
            "A future Treasury Bill observation was assigned to an "
            "earlier market date."
        )

    if (
        daily_risk_free[
            "Rate_Observation_Age_Calendar_Days"
        ] < 0
    ).any():
        raise AssertionError(
            "Negative macroeconomic observation age detected."
        )

    daily_risk_free.to_csv(
        OUTPUT_DAILY_RISK_FREE,
        index=False,
    )

    # --------------------------------------------------------
    # 3.5 Build alignment audit
    # --------------------------------------------------------

    alignment_audit = (
        daily_risk_free.groupby(
            "Observation_Date",
            as_index=False,
        )
        .agg(
            First_Assigned_Market_Date=("Date", "min"),
            Last_Assigned_Market_Date=("Date", "max"),
            Assigned_Market_Dates=("Date", "size"),
            Risk_Free_Rate_Annual_Percent=(
                "Risk_Free_Rate_Annual_Percent",
                "first",
            ),
            Maximum_Observation_Age_Days=(
                "Rate_Observation_Age_Calendar_Days",
                "max",
            ),
        )
        .sort_values("Observation_Date")
        .reset_index(drop=True)
    )

    alignment_audit.to_csv(
        OUTPUT_ALIGNMENT_AUDIT,
        index=False,
    )

    # --------------------------------------------------------
    # 3.6 Summary statistics
    # --------------------------------------------------------

    minimum_rate_row = macro.loc[
        macro[
            "Risk_Free_Rate_Annual_Percent"
        ].idxmin()
    ]

    maximum_rate_row = macro.loc[
        macro[
            "Risk_Free_Rate_Annual_Percent"
        ].idxmax()
    ]

    summary = pd.DataFrame(
        {
            "Statistic": [
                "Raw workbook observations",
                "Rows with unparseable dates",
                "Rows with unparseable rates",
                "Duplicate-date rows before cleaning",
                "Valid unique weekly observations",
                "Weekly-series start date",
                "Weekly-series end date",
                "Minimum annual rate (%)",
                "Minimum-rate observation date",
                "Maximum annual rate (%)",
                "Maximum-rate observation date",
                "Market trading dates aligned",
                "Daily-series start date",
                "Daily-series end date",
                "Maximum assigned observation age (days)",
                "Future observations assigned",
            ],
            "Value": [
                raw_row_count,
                missing_date_rows,
                missing_rate_rows,
                duplicate_date_rows,
                len(macro),
                macro["Observation_Date"].min().date(),
                macro["Observation_Date"].max().date(),
                minimum_rate_row[
                    "Risk_Free_Rate_Annual_Percent"
                ],
                minimum_rate_row[
                    "Observation_Date"
                ].date(),
                maximum_rate_row[
                    "Risk_Free_Rate_Annual_Percent"
                ],
                maximum_rate_row[
                    "Observation_Date"
                ].date(),
                len(daily_risk_free),
                daily_risk_free["Date"].min().date(),
                daily_risk_free["Date"].max().date(),
                int(
                    daily_risk_free[
                        "Rate_Observation_Age_Calendar_Days"
                    ].max()
                ),
                int(
                    (
                        daily_risk_free["Observation_Date"]
                        > daily_risk_free["Date"]
                    ).sum()
                ),
            ],
        }
    )

    summary.to_csv(
        OUTPUT_PROCESSING_SUMMARY,
        index=False,
    )

    # --------------------------------------------------------
    # 3.7 Thesis-ready macroeconomic figure
    # --------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(11, 5.5)
    )

    axis.plot(
        macro["Observation_Date"],
        macro["Risk_Free_Rate_Annual_Percent"],
        linewidth=1.8,
        label="91-Day Treasury Bill Tender Rate",
    )

    axis.scatter(
        minimum_rate_row["Observation_Date"],
        minimum_rate_row[
            "Risk_Free_Rate_Annual_Percent"
        ],
        s=45,
        zorder=3,
    )

    axis.scatter(
        maximum_rate_row["Observation_Date"],
        maximum_rate_row[
            "Risk_Free_Rate_Annual_Percent"
        ],
        s=45,
        zorder=3,
    )

    axis.annotate(
        (
            f"Sample minimum: "
            f"{minimum_rate_row['Risk_Free_Rate_Annual_Percent']:.2f}%"
        ),
        xy=(
            minimum_rate_row["Observation_Date"],
            minimum_rate_row[
                "Risk_Free_Rate_Annual_Percent"
            ],
        ),
        xytext=(18, -30),
        textcoords="offset points",
        arrowprops={
            "arrowstyle": "->",
        },
    )

    axis.annotate(
        (
            f"Sample maximum: "
            f"{maximum_rate_row['Risk_Free_Rate_Annual_Percent']:.2f}%"
        ),
        xy=(
            maximum_rate_row["Observation_Date"],
            maximum_rate_row[
                "Risk_Free_Rate_Annual_Percent"
            ],
        ),
        xytext=(-145, 28),
        textcoords="offset points",
        arrowprops={
            "arrowstyle": "->",
        },
    )

    axis.set_title(
        "South African 91-Day Treasury Bill Tender Rate"
    )

    axis.set_xlabel("Observation date")
    axis.set_ylabel("Annualised rate (%)")

    axis.xaxis.set_major_locator(
        mdates.YearLocator()
    )

    axis.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y")
    )

    axis.grid(
        axis="both",
        linestyle="--",
        alpha=0.40,
    )

    axis.legend(
        loc="best"
    )

    figure.autofmt_xdate(
        rotation=45,
        ha="right",
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    # --------------------------------------------------------
    # 3.8 Terminal results
    # --------------------------------------------------------

    print("\n" + "=" * 74)
    print("SCRIPT 04 EMPIRICAL OUTPUTS")
    print("=" * 74)

    print(
        f"Raw SARB workbook observations: "
        f"{raw_row_count:,}"
    )

    print(
        f"Valid unique Treasury Bill observations: "
        f"{len(macro):,}"
    )

    print(
        "Weekly series date range: "
        f"{macro['Observation_Date'].min().date()} to "
        f"{macro['Observation_Date'].max().date()}"
    )

    print(
        "Minimum annual Treasury Bill rate: "
        f"{minimum_rate_row['Risk_Free_Rate_Annual_Percent']:.4f}% "
        f"on {minimum_rate_row['Observation_Date'].date()}"
    )

    print(
        "Maximum annual Treasury Bill rate: "
        f"{maximum_rate_row['Risk_Free_Rate_Annual_Percent']:.4f}% "
        f"on {maximum_rate_row['Observation_Date'].date()}"
    )

    print(
        f"Market trading dates aligned: "
        f"{len(daily_risk_free):,}"
    )

    print(
        "Daily risk-free date range: "
        f"{daily_risk_free['Date'].min().date()} to "
        f"{daily_risk_free['Date'].max().date()}"
    )

    print(
        "Maximum age of an assigned macro observation: "
        f"{daily_risk_free['Rate_Observation_Age_Calendar_Days'].max():,} "
        "calendar days"
    )

    print(
        "Market dates assigned a future Treasury Bill observation: "
        "0"
    )

    print(
        "Missing daily risk-free observations after alignment: "
        f"{daily_risk_free['Risk_Free_Rate_Daily'].isna().sum():,}"
    )

    print("\nIntegrity checks passed:")
    print(
        "- The weekly series contains one rate per observation date."
    )
    print(
        "- Every market date received the latest rate available "
        "on or before that date."
    )
    print(
        "- No future Treasury Bill observation was assigned "
        "retrospectively."
    )
    print(
        "- Annual percentage rates were converted to daily decimal rates "
        f"using {TRADING_DAYS_PER_YEAR} trading days per year."
    )

    print("\nCore output files:")
    print(f"- {OUTPUT_WEEKLY_CLEAN}")
    print(f"- {OUTPUT_DAILY_RISK_FREE}")

    print("\nAudit and figure outputs:")
    print(f"- {OUTPUT_PROCESSING_SUMMARY}")
    print(f"- {OUTPUT_ALIGNMENT_AUDIT}")
    print(f"- {OUTPUT_FIGURE}")

    print("=" * 74)


if __name__ == "__main__":
    main()