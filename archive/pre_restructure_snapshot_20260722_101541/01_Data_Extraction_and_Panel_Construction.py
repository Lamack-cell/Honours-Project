"""
Script 01: JSE Data Extraction and Broad Panel Construction
------------------------------------------------------------

Purpose
-------
Downloads daily OHLCV data for a broad, researcher-curated set of current
and historical large- and mid-cap JSE securities.

The output is a broad candidate-security panel. It is NOT presented as an
official historical FTSE/JSE Top 40 constituent dataset. Point-in-time
liquidity screening is performed separately in Script 02.

Outputs
-------
1. JSE_Broad_Master_Panel.csv
2. JSE_Extraction_Ticker_Summary.csv
3. JSE_Failed_Tickers.csv
4. JSE_Data_Quality_Issues.csv

Required packages
-----------------
pip install yfinance pandas numpy
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# 1. CONFIGURATION
# ============================================================

START_DATE = "2016-01-01"

# yfinance treats the end date as exclusive.
# This therefore downloads observations through 19 March 2026.
END_DATE_EXCLUSIVE = "2026-03-20"

OUTPUT_DIRECTORY = Path(".")
MAX_DOWNLOAD_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 3

BROAD_JSE_TICKERS = [
    # Current and recent large-cap securities
    "ABG.JO", "AGL.JO", "ANG.JO", "APN.JO", "ARI.JO", "AVI.JO",
    "BTI.JO", "BVT.JO", "CLS.JO", "CPI.JO", "DSY.JO", "EXX.JO",
    "FSR.JO", "GFI.JO", "GLN.JO", "IMP.JO", "INP.JO", "KIO.JO",
    "MRP.JO", "MTN.JO", "NED.JO", "NPN.JO", "OUT.JO", "REM.JO",
    "SBK.JO", "SHP.JO", "SLM.JO", "SPP.JO", "TBS.JO", "TRU.JO",
    "VOD.JO", "WHL.JO", "BID.JO", "RNI.JO", "PRX.JO", "SSW.JO",
    "TGA.JO", "NPH.JO", "CFR.JO",

    # Historical large- and mid-cap candidate securities
    "SOL.JO", "OMU.JO", "AMS.JO", "HAR.JO", "SAP.JO", "GRT.JO",
    "RDF.JO", "TFG.JO", "PIK.JO", "MCG.JO", "NTC.JO", "DRD.JO",
    "BAW.JO", "JSE.JO", "KAP.JO", "NEP.JO", "RES.JO", "HYP.JO",
    "ATT.JO", "VKE.JO", "AFT.JO", "BRL.JO", "DTC.JO", "EOH.JO",
    "IPL.JO", "LBH.JO", "MEI.JO", "MSP.JO", "N91.JO", "PPC.JO",
    "RLF.JO", "SUR.JO", "TCP.JO", "WBO.JO",
]

REQUIRED_COLUMNS = [
    "Date",
    "Ticker",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
]


# ============================================================
# 2. VALIDATION HELPERS
# ============================================================

def validate_configuration() -> list[str]:
    """
    Validate ticker-list and date configuration before downloading.
    """

    errors: list[str] = []

    duplicate_tickers = pd.Series(BROAD_JSE_TICKERS).duplicated(keep=False)
    if duplicate_tickers.any():
        duplicates = (
            pd.Series(BROAD_JSE_TICKERS)[duplicate_tickers]
            .drop_duplicates()
            .tolist()
        )
        errors.append(f"Duplicate ticker symbols found: {duplicates}")

    if pd.Timestamp(START_DATE) >= pd.Timestamp(END_DATE_EXCLUSIVE):
        errors.append("START_DATE must be earlier than END_DATE_EXCLUSIVE.")

    return errors


def standardise_downloaded_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert yfinance output into a predictable flat-column dataframe.
    """

    if isinstance(data.columns, pd.MultiIndex):
        # A single-ticker request can still occasionally return a MultiIndex.
        data.columns = [
            column[0] if isinstance(column, tuple) else column
            for column in data.columns
        ]

    data = data.reset_index()

    # Different yfinance versions may label the date index differently.
    if "Datetime" in data.columns and "Date" not in data.columns:
        data = data.rename(columns={"Datetime": "Date"})

    return data


def download_single_ticker(ticker: str) -> tuple[pd.DataFrame | None, str | None]:
    """
    Download one ticker with retries.

    Returns
    -------
    dataframe or None
        Clean ticker-level OHLCV observations.
    error_message or None
        Error explanation where no usable data were returned.
    """

    last_error: str | None = None

    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            data = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE_EXCLUSIVE,
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
            )

            if data is None or data.empty:
                last_error = "No observations returned."
            else:
                data = standardise_downloaded_columns(data)

                missing_columns = [
                    column
                    for column in REQUIRED_COLUMNS
                    if column not in data.columns and column != "Ticker"
                ]

                if missing_columns:
                    last_error = (
                        "Required columns missing from download: "
                        f"{missing_columns}"
                    )
                else:
                    data["Ticker"] = ticker
                    data = data[REQUIRED_COLUMNS].copy()

                    data["Date"] = pd.to_datetime(
                        data["Date"],
                        errors="coerce"
                    ).dt.tz_localize(None)

                    numeric_columns = [
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Adj Close",
                        "Volume",
                    ]

                    for column in numeric_columns:
                        data[column] = pd.to_numeric(
                            data[column],
                            errors="coerce"
                        )

                    # Rows containing no price information are unusable.
                    data = data.dropna(
                        subset=["Date", "Close", "Adj Close"],
                        how="any"
                    )

                    data = (
                        data.sort_values("Date")
                        .drop_duplicates(
                            subset=["Date", "Ticker"],
                            keep="last"
                        )
                        .reset_index(drop=True)
                    )

                    if data.empty:
                        last_error = "No valid observations remained after cleaning."
                    else:
                        return data, None

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < MAX_DOWNLOAD_ATTEMPTS:
            time.sleep(RETRY_WAIT_SECONDS)

    return None, last_error


def identify_quality_issues(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Identify row-level data problems without silently modifying valid history.
    """

    issue_frames: list[pd.DataFrame] = []

    checks = {
        "Missing_Open": panel["Open"].isna(),
        "Missing_High": panel["High"].isna(),
        "Missing_Low": panel["Low"].isna(),
        "Missing_Close": panel["Close"].isna(),
        "Missing_Adjusted_Close": panel["Adj Close"].isna(),
        "Missing_Volume": panel["Volume"].isna(),
        "Nonpositive_Close": panel["Close"].le(0),
        "Nonpositive_Adjusted_Close": panel["Adj Close"].le(0),
        "Negative_Volume": panel["Volume"].lt(0),
        "High_Below_Low": panel["High"].lt(panel["Low"]),
        "High_Below_Open": panel["High"].lt(panel["Open"]),
        "High_Below_Close": panel["High"].lt(panel["Close"]),
        "Low_Above_Open": panel["Low"].gt(panel["Open"]),
        "Low_Above_Close": panel["Low"].gt(panel["Close"]),
    }

    for issue_name, mask in checks.items():
        if mask.any():
            issue_data = panel.loc[
                mask,
                REQUIRED_COLUMNS
            ].copy()
            issue_data.insert(0, "Issue", issue_name)
            issue_frames.append(issue_data)

    duplicate_mask = panel.duplicated(
        subset=["Date", "Ticker"],
        keep=False
    )

    if duplicate_mask.any():
        duplicate_data = panel.loc[
            duplicate_mask,
            REQUIRED_COLUMNS
        ].copy()
        duplicate_data.insert(0, "Issue", "Duplicate_Ticker_Date")
        issue_frames.append(duplicate_data)

    if not issue_frames:
        return pd.DataFrame(columns=["Issue", *REQUIRED_COLUMNS])

    return pd.concat(issue_frames, ignore_index=True)


# ============================================================
# 3. DOWNLOAD AND PANEL CONSTRUCTION
# ============================================================

def main() -> None:
    print("=" * 70)
    print("SCRIPT 01: JSE BROAD PANEL EXTRACTION")
    print("=" * 70)

    configuration_errors = validate_configuration()
    if configuration_errors:
        raise ValueError(
            "Configuration errors:\n- "
            + "\n- ".join(configuration_errors)
        )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    successful_frames: list[pd.DataFrame] = []
    failed_records: list[dict[str, str]] = []

    print(
        f"Downloading {len(BROAD_JSE_TICKERS)} candidate securities "
        f"from {START_DATE} to {END_DATE_EXCLUSIVE} "
        "(exclusive end date)."
    )

    for position, ticker in enumerate(BROAD_JSE_TICKERS, start=1):
        print(
            f"[{position:02d}/{len(BROAD_JSE_TICKERS):02d}] "
            f"Downloading {ticker}..."
        )

        ticker_data, error = download_single_ticker(ticker)

        if ticker_data is not None:
            successful_frames.append(ticker_data)
            print(
                f"    Success: {len(ticker_data):,} rows, "
                f"{ticker_data['Date'].min().date()} to "
                f"{ticker_data['Date'].max().date()}."
            )
        else:
            failed_records.append(
                {
                    "Ticker": ticker,
                    "Reason": error or "Unknown error",
                }
            )
            print(f"    Failed: {error}")

    if not successful_frames:
        raise RuntimeError(
            "No ticker returned usable data. "
            "Check the internet connection and ticker symbols."
        )

    panel = pd.concat(successful_frames, ignore_index=True)

    panel = (
        panel[REQUIRED_COLUMNS]
        .sort_values(["Date", "Ticker"])
        .drop_duplicates(subset=["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )

    # Use nullable integer storage for volume where possible.
    panel["Volume"] = panel["Volume"].round().astype("Int64")

    quality_issues = identify_quality_issues(panel)

    ticker_summary = (
        panel.groupby("Ticker", as_index=False)
        .agg(
            First_Date=("Date", "min"),
            Last_Date=("Date", "max"),
            Observation_Count=("Date", "size"),
            Missing_Adjusted_Close=("Adj Close", lambda x: int(x.isna().sum())),
            Missing_Volume=("Volume", lambda x: int(x.isna().sum())),
            Zero_Volume_Days=("Volume", lambda x: int(x.fillna(0).eq(0).sum())),
        )
        .sort_values("Ticker")
        .reset_index(drop=True)
    )

    failed_tickers = pd.DataFrame(
        failed_records,
        columns=["Ticker", "Reason"]
    )

    panel_path = OUTPUT_DIRECTORY / "JSE_Broad_Master_Panel.csv"
    summary_path = OUTPUT_DIRECTORY / "JSE_Extraction_Ticker_Summary.csv"
    failed_path = OUTPUT_DIRECTORY / "JSE_Failed_Tickers.csv"
    issues_path = OUTPUT_DIRECTORY / "JSE_Data_Quality_Issues.csv"

    panel.to_csv(panel_path, index=False)
    ticker_summary.to_csv(summary_path, index=False)
    failed_tickers.to_csv(failed_path, index=False)
    quality_issues.to_csv(issues_path, index=False)

    # ========================================================
    # 4. THESIS-READY DIAGNOSTICS
    # ========================================================

    print("\n" + "=" * 70)
    print("SCRIPT 01 EMPIRICAL OUTPUTS")
    print("=" * 70)

    print(f"Requested ticker count: {len(BROAD_JSE_TICKERS):,}")
    print(f"Successful ticker count: {panel['Ticker'].nunique():,}")
    print(f"Failed ticker count: {len(failed_tickers):,}")
    print(f"Total stock-day observations: {len(panel):,}")
    print(f"Panel start date: {panel['Date'].min().date()}")
    print(f"Panel end date: {panel['Date'].max().date()}")
    print(
        "Duplicate ticker-date observations after cleaning: "
        f"{panel.duplicated(['Date', 'Ticker']).sum():,}"
    )
    print(f"Recorded quality-issue rows: {len(quality_issues):,}")

    print("\nMissing values in the final broad panel:")
    print(panel.isna().sum().to_string())

    print("\nObservations per ticker:")
    print(
        ticker_summary["Observation_Count"]
        .describe()
        .round(2)
        .to_string()
    )

    print("\nOutput files:")
    print(f"- {panel_path}")
    print(f"- {summary_path}")
    print(f"- {failed_path}")
    print(f"- {issues_path}")

    if not failed_tickers.empty:
        print(
            "\nImportant: failed historical tickers must be disclosed "
            "as a data-source limitation rather than silently omitted."
        )

    if not quality_issues.empty:
        print(
            "\nReview JSE_Data_Quality_Issues.csv before proceeding "
            "to Script 02."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()