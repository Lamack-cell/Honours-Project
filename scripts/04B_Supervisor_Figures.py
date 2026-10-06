"""
Script 04B: Supervisor-Requested Diagnostic Figures for Chapter 4
------------------------------------------------------------------

Generates five publication-quality figures for the thesis data chapter:

  Fig 4.5  - Price Index time-series for a Top Quartile, Middle Pack,
              and Low Quartile performer (single-panel overlay, indexed to 100).

  Fig 4.6  - Cross-sectional percentile rank (Target_Y) over time for
              the same three securities — daily texture + 63-day trend
              (3-panel subplot for readability).

  Fig 4.7a,b,c - Nine-panel (split across three files) standardised-feature
              time-series for NPN.JO, showing all cross-sectional Z-scored
              predictors. Each subplot has clear X and Y axis definitions.

  Fig 4.8  - Cross-sectional ranking snapshot table for one prediction
              date, sorted by Target_Y with quartile colour-coding.

Inputs
------
  data/interim/02_universe/JSE_Broad_Trading_Panel.csv
  data/interim/03_target/JSE_Panel_Target_Engineered.csv
  data/processed/05_features/JSE_Panel_Features_Engineered.csv

Outputs
-------
  outputs/figures/chapter_4/figure_4_5_price_time_series.png
  outputs/figures/chapter_4/figure_4_6_target_y_time_series.png
  outputs/figures/chapter_4/figure_4_7a_feature_time_series.png
  outputs/figures/chapter_4/figure_4_7b_feature_time_series.png
  outputs/figures/chapter_4/figure_4_7c_feature_time_series.png
  outputs/figures/chapter_4/figure_4_8_ranking_snapshot.png

Run from project root
---------------------
  python .\\scripts\\04B_Supervisor_Figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# ============================================================
# GLOBAL STYLE
# ============================================================

matplotlib.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["DejaVu Sans", "Arial"],
    "font.size":            9,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.spines.left":     True,
    "axes.spines.bottom":   True,
    "axes.linewidth":       0.7,
    "axes.edgecolor":       "#AAAAAA",
    "axes.facecolor":       "white",
    "axes.grid":            True,
    "grid.color":           "#E8E8E8",
    "grid.linewidth":       0.5,
    "grid.linestyle":       "-",
    "axes.labelsize":       9,
    "axes.titlesize":       9.5,
    "axes.titleweight":     "bold",
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "xtick.major.size":     3,
    "ytick.major.size":     3,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "legend.fontsize":      8,
    "legend.frameon":       False,
    "legend.handlelength":  1.5,
    "figure.facecolor":     "white",
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.facecolor":    "white",
    "savefig.pad_inches":   0.08,
})

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BROAD_PANEL_PATH    = PROJECT_ROOT / "data" / "interim"   / "02_universe" / "JSE_Broad_Trading_Panel.csv"
TARGET_PANEL_PATH   = PROJECT_ROOT / "data" / "interim"   / "03_target"   / "JSE_Panel_Target_Engineered.csv"
FEATURES_PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "05_features" / "JSE_Panel_Features_Engineered.csv"

FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "chapter_4"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIGURATION
# ============================================================

SNAPSHOT_DATE = "2024-06-28"

STANDARDISED_FEATURES = [
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

FEATURE_LABELS = {
    "Mom_1M":                 "1-Month Momentum",
    "Mom_3M":                 "3-Month Momentum",
    "Mom_6M":                 "6-Month Momentum",
    "Vol_21D":                "21-Day Volatility",
    "RSI_14":                 "RSI (14-Day)",
    "SMA_50_Ratio":           "50-Day SMA Ratio",
    "SMA_200_Ratio":          "200-Day SMA Ratio",
    "Volume_Shock_21D":       "Volume Shock (21D)",
    "Value_Traded_Shock_63D": "Value-Traded Shock (63D)",
}

# ============================================================
# HELPERS
# ============================================================

def _validate_schema(df: pd.DataFrame, required: list[str], label: str) -> None:
    df.columns = df.columns.str.strip()
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"\n[SCHEMA ERROR] {label} is missing required columns:\n"
            f"  Missing : {missing}\n"
            f"  Found   : {df.columns.tolist()}\n"
            f"  Shape   : {df.shape}"
        )

def _year_locator(ax: plt.Axes, fmt: str = "%Y") -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    ax.tick_params(axis="x", rotation=0)

def _annotate_end(ax: plt.Axes, x, y: float, label: str, color: str) -> None:
    """Place a compact right-margin label at the end of a line."""
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=7.5,
        color=color,
        fontweight="bold",
        va="center",
        clip_on=False,
    )

# ============================================================
# DATA LOADING & SELECTION
# ============================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (BROAD_PANEL_PATH, TARGET_PANEL_PATH, FEATURES_PANEL_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"\n[PATH ERROR] CSV not found:\n  {path}\n"
                f"  PROJECT_ROOT = {PROJECT_ROOT}\n"
                f"  Ensure pipeline scripts 03 and 05 have been run first."
            )

    print("Loading broad trading panel ...")
    broad = pd.read_csv(BROAD_PANEL_PATH)
    broad.columns = broad.columns.str.strip()
    _validate_schema(broad, ["Date", "Ticker", "Adj Close"], "Broad Trading Panel")
    broad["Date"] = pd.to_datetime(broad["Date"])

    print("Loading target-engineered panel ...")
    target = pd.read_csv(TARGET_PANEL_PATH)
    target.columns = target.columns.str.strip()
    _validate_schema(target, ["Date", "Ticker", "Target_Y", "Daily_Return_Rank", "Forward_Return"], "Target Panel")
    target["Date"] = pd.to_datetime(target["Date"])

    print("Loading feature-engineered panel ...")
    features = pd.read_csv(FEATURES_PANEL_PATH)
    features.columns = features.columns.str.strip()
    _validate_schema(
        features,
        ["Date", "Ticker", "Target_Y", "Daily_Return_Rank", "Forward_Return"] + STANDARDISED_FEATURES,
        "Features Panel",
    )
    features["Date"] = pd.to_datetime(features["Date"])

    print(f"  Broad   : {len(broad):,} rows  |  {broad['Ticker'].nunique()} tickers  |  "
          f"{broad['Date'].min().date()} – {broad['Date'].max().date()}")
    print(f"  Target  : {len(target):,} rows  |  {target['Ticker'].nunique()} tickers  |  "
          f"{target['Date'].min().date()} – {target['Date'].max().date()}")
    print(f"  Features: {len(features):,} rows  |  {features['Ticker'].nunique()} tickers  |  "
          f"{features['Date'].min().date()} – {features['Date'].max().date()}")

    return broad, target, features

def get_showcase_tickers(target: pd.DataFrame) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """
    Dynamically select 3 showcase tickers:
    - Top quartile performer (consistently high Target_Y)
    - Middle of the pack (Target_Y around 0.5)
    - Low quartile performer (consistently low Target_Y)
    Must have sufficient history to be a good plot.
    """
    counts = target.groupby("Ticker")["Target_Y"].count()
    valid_tickers = counts[counts > 2400].index  # Requires nearly full history from 2016 to 2026
    
    mean_rank = target[target["Ticker"].isin(valid_tickers)].groupby("Ticker")["Target_Y"].mean().sort_values(ascending=False)
    
    top_ticker = mean_rank.index[0]
    bottom_ticker = mean_rank.index[-1]
    
    # Find the one closest to 0.5
    middle_ticker = (mean_rank - 0.5).abs().idxmin()

    tickers = [top_ticker, middle_ticker, bottom_ticker]
    
    labels = {
        top_ticker: f"{top_ticker.replace('.JO', '')} (Avg Rank: {mean_rank[top_ticker]:.2f})",
        middle_ticker: f"{middle_ticker.replace('.JO', '')} (Avg Rank: {mean_rank[middle_ticker]:.2f})",
        bottom_ticker: f"{bottom_ticker.replace('.JO', '')} (Avg Rank: {mean_rank[bottom_ticker]:.2f})",
    }
    
    palette = {
        top_ticker: "#0072B2",     # blue
        middle_ticker: "#D55E00",  # vermillion
        bottom_ticker: "#009E73",  # teal
    }
    
    return tickers, labels, palette

# ============================================================
# FIGURE 4.5 – PRICE INDEX TIME SERIES
# ============================================================

def plot_price_time_series(broad: pd.DataFrame, showcase_tickers: list[str], ticker_labels: dict, palette: dict) -> None:

    print("\nGenerating Figure 4.5 – Price Index time series ...")

    present = set(broad["Ticker"].unique())
    for t in showcase_tickers:
        if t not in present:
            print(f"  WARNING: {t} not found in broad panel.")

    subset = (
        broad[broad["Ticker"].isin(showcase_tickers)]
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
        .copy()
    )

    if subset.empty:
        print("  ERROR: No showcase tickers in broad panel. Skipping.")
        return

    # Index each security to 100 at its first available observation
    subset["Index"] = subset.groupby("Ticker")["Adj Close"].transform(
        lambda s: s / s.iloc[0] * 100.0
    )

    # ── Figure ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    for ticker in showcase_tickers:
        data  = subset[subset["Ticker"] == ticker].reset_index(drop=True)
        color = palette[ticker]
        label = ticker_labels[ticker]
        if data.empty:
            continue

        dates = data["Date"].values
        idx   = data["Index"].values

        ax.plot(dates, idx, color=color, linewidth=1.6, zorder=3)

        # End-of-line annotation (right margin)
        _annotate_end(ax, dates[-1], idx[-1], label, color)

    # Base-100 reference
    ax.axhline(100, color="#AAAAAA", linewidth=0.7, linestyle=":", zorder=1)

    ax.set_ylabel("Price Index", fontsize=9)
    ax.set_xlabel("Date", fontsize=9)
    _year_locator(ax)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    # Compact subtitle as figure annotation below the x-axis
    fig.text(
        0.5, -0.04,
        "Each series indexed to 100 at its first available observation.",
        ha="center", fontsize=7, color="#555555", style="italic",
    )

    fig.suptitle(
        "Figure 4.5 — Price Index: Top Quartile, Middle Pack, and Low Quartile Performer",
        fontsize=10, fontweight="bold", x=0.5, ha="center",
    )
    fig.tight_layout()

    out_path = FIGURE_DIR / "figure_4_5_price_time_series.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ============================================================
# FIGURE 4.6 – TARGET_Y TIME SERIES
# ============================================================

def plot_target_y_time_series(target: pd.DataFrame, showcase_tickers: list[str], ticker_labels: dict, palette: dict) -> None:

    print("\nGenerating Figure 4.6 – Target_Y time series ...")

    subset = (
        target[target["Ticker"].isin(showcase_tickers)]
        .sort_values(["Ticker", "Date"])
        .copy()
    )

    if subset.empty:
        print("  ERROR: No showcase tickers in target panel. Skipping.")
        return

    # 3x1 subplots to give each ticker its own readable space
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.0), sharex=True, gridspec_kw={"hspace": 0.25})

    for ax, ticker in zip(axes, showcase_tickers):
        data  = subset[subset["Ticker"] == ticker].copy()
        color = palette[ticker]
        if data.empty:
            print(f"  WARNING: {ticker} not in target panel.")
            ax.set_visible(False)
            continue

        # Dashed reference lines
        for level in [0.25, 0.50, 0.75]:
            ax.axhline(level, color="#BBBBBB", linewidth=0.7, linestyle="--", zorder=1)

        # Labels for reference lines only on the top panel to avoid clutter
        if ax == axes[0]:
            for level, lbl in [(0.75, "Top quartile (0.75)"), (0.50, "Median (0.50)"), (0.25, "Bottom quartile (0.25)")]:
                ax.text(
                    subset["Date"].max(), level + 0.012,
                    lbl, fontsize=6.5, color="#888888", ha="right", va="bottom",
                )

        # Daily raw rank — texture layer
        ax.plot(data["Date"].values, data["Target_Y"].values,
                color=color, linewidth=0.4, alpha=0.15, zorder=2)

        # 63-day rolling mean — primary signal
        rolling = data.set_index("Date")["Target_Y"].rolling(63, min_periods=20).mean()
        ax.plot(rolling.index, rolling.values,
                color=color, linewidth=2.0, zorder=3,
                label=ticker_labels[ticker])

        # Text label for the ticker to identify the subplot
        ax.text(
            subset["Date"].min(), 0.95,
            ticker_labels[ticker], fontsize=8, color=color, fontweight="bold",
            va="top", ha="left"
        )

        ax.set_ylim(0.0, 1.05)
        ax.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # Apply common axis labels
    for ax_ in axes:
        ax_.set_ylabel("Cross-Sectional Rank (Target_Y)", fontsize=9)
    axes[-1].set_xlabel("Date", fontsize=9)
    axes[-1].set_xlim(subset["Date"].min(), subset["Date"].max())
    _year_locator(axes[-1])

    fig.text(
        0.5, 0.02,
        "Thin line = daily cross-sectional percentile rank  (0 = worst, 1 = best).  "
        "Thick line = 63-trading-day rolling mean.",
        ha="center", fontsize=7, color="#555555", style="italic",
    )

    fig.suptitle(
        "Figure 4.6 — Engineered Target Variable (Target_Y) Over Time",
        fontsize=10, fontweight="bold", y=0.96
    )
    fig.subplots_adjust(bottom=0.10)

    out_path = FIGURE_DIR / "figure_4_6_target_y_time_series.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ============================================================
# FIGURE 4.7 – FEATURE TIME SERIES  (split across three pages)
# ============================================================

def _draw_feature_page(
    data: pd.DataFrame,
    features_subset: list[str],
    fig_title: str,
    out_path,
) -> None:
    """Shared rendering logic for one feature page (3x1 vertical layout)."""
    dates = data["Date"].values
    n_rows = 3
    n_cols = 1

    fig = plt.figure(figsize=(7.5, 7.5))
    gs  = GridSpec(
        n_rows, n_cols, figure=fig,
        hspace=0.55, wspace=0.10,
        left=0.10, right=0.97,
        top=0.92, bottom=0.08,
    )

    for idx, feature in enumerate(features_subset):
        row, col = divmod(idx, n_cols)
        ax = fig.add_subplot(gs[row, col])

        vals = data[feature].values

        # Shade above 0 light blue, below 0 light red
        c_above = "#89CFF0"  # light blue
        c_below = "#FF7F7F"  # light red
        line_color = "#00008B" # dark blue

        ax.fill_between(dates, vals, 0,
                        where=(vals >= 0), color=c_above, alpha=0.50, linewidth=0)
        ax.fill_between(dates, vals, 0,
                        where=(vals <  0), color=c_below, alpha=0.50, linewidth=0)
        ax.plot(dates, vals, color=line_color, linewidth=0.7, zorder=3)
        ax.axhline(0, color="#888888", linewidth=0.8, linestyle="-", zorder=2)

        ax.set_title(FEATURE_LABELS[feature], fontsize=9.5, pad=6, fontweight="bold")

        ax.set_ylabel("Z-score", fontsize=8)
        ax.set_xlabel("Date", fontsize=8)
        
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("'%y"))
        ax.tick_params(axis="x", labelsize=8, rotation=0)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)

    fig.suptitle(fig_title, fontsize=11, fontweight="bold", y=0.98)
    fig.text(
        0.5, 0.02,
        "Daily cross-sectional Z-score for Naspers (NPN.JO).  "
        "Light blue fill = above active-universe mean; light red fill = below mean.",
        ha="center", fontsize=8, color="#555555", style="italic",
    )

    fig.savefig(out_path)
    plt.close(fig)


def plot_feature_time_series(features: pd.DataFrame) -> None:

    print("\nGenerating Figures 4.7a, 4.7b, and 4.7c – Feature time series ...")

    FEATURE_TICKER = "NPN.JO"

    data = (
        features[features["Ticker"] == FEATURE_TICKER]
        .sort_values("Date")
        .reset_index(drop=True)
        .copy()
    )

    if data.empty:
        print(f"  WARNING: {FEATURE_TICKER} not in features panel. Skipping.")
        return

    missing_feats = [f for f in STANDARDISED_FEATURES if f not in data.columns]
    if missing_feats:
        print(f"  ERROR: Feature columns missing: {missing_feats}. Skipping.")
        return

    # ── Group A: Momentum (first 3 features) ──
    group_a = STANDARDISED_FEATURES[:3]
    path_a  = FIGURE_DIR / "figure_4_7a_feature_time_series.png"
    _draw_feature_page(
        data, group_a,
        fig_title="Figure 4.7a — Momentum Features  | Naspers (NPN.JO)",
        out_path=path_a,
    )
    print(f"  Saved: {path_a.relative_to(PROJECT_ROOT)}")

    # ── Group B: Volatility & Oscillators (next 3 features) ──
    group_b = STANDARDISED_FEATURES[3:6]
    path_b  = FIGURE_DIR / "figure_4_7b_feature_time_series.png"
    _draw_feature_page(
        data, group_b,
        fig_title="Figure 4.7b — Volatility & Oscillator Features  | Naspers (NPN.JO)",
        out_path=path_b,
    )
    print(f"  Saved: {path_b.relative_to(PROJECT_ROOT)}")

    # ── Group C: Trend & Liquidity Shocks (last 3 features) ──
    group_c = STANDARDISED_FEATURES[6:]
    path_c  = FIGURE_DIR / "figure_4_7c_feature_time_series.png"
    _draw_feature_page(
        data, group_c,
        fig_title="Figure 4.7c — Trend & Liquidity Shock Features  | Naspers (NPN.JO)",
        out_path=path_c,
    )
    print(f"  Saved: {path_c.relative_to(PROJECT_ROOT)}")


# ============================================================
# FIGURE 4.8 – CROSS-SECTIONAL RANKING SNAPSHOT
# ============================================================

def plot_ranking_snapshot(features: pd.DataFrame) -> None:

    print(f"\nGenerating Figure 4.8 – Ranking snapshot ({SNAPSHOT_DATE}) ...")

    snap_ts   = pd.Timestamp(SNAPSHOT_DATE)
    available = sorted(features["Date"].unique())
    if not available:
        print("  ERROR: No dates in features panel. Skipping.")
        return

    closest  = min(available, key=lambda d: abs(d - snap_ts))
    snapshot = (
        features[features["Date"] == closest]
        .sort_values("Target_Y", ascending=False)
        .reset_index(drop=True)
    )

    if snapshot.empty:
        print(f"  WARNING: No data for {closest.date()}. Skipping.")
        return

    n = len(snapshot)
    print(f"  Date used : {closest.date()}  |  N = {n} securities")

    # All 9 standardised features shown alongside the key metadata columns
    display_cols = ["Ticker", "Target_Y", "Forward_Return"] + STANDARDISED_FEATURES
    missing_dc = [c for c in display_cols if c not in snapshot.columns]
    if missing_dc:
        print(f"  ERROR: Missing columns: {missing_dc}. Skipping.")
        return

    # ── Build display data ───────────────────────────────────
    disp = snapshot[display_cols].copy()
    disp["Ticker"]         = disp["Ticker"].str.replace(".JO", "", regex=False)
    disp["Forward_Return"] = (disp["Forward_Return"] * 100).round(2)
    disp["Target_Y"]       = disp["Target_Y"].round(3)
    for c in STANDARDISED_FEATURES:
        disp[c] = disp[c].round(2)
    disp.insert(0, "Rank", np.arange(1, n + 1))
    disp = disp[["Rank", "Ticker", "Target_Y", "Forward_Return"] + STANDARDISED_FEATURES]

    # Short column header labels to keep the table compact
    headers = [
        "#", "Ticker",
        "Rank\nScore", "Next-Day\nReturn (%)",
        "Mom\n1M (Z)", "Mom\n3M (Z)", "Mom\n6M (Z)",
        "Vol\n21D (Z)", "RSI\n14 (Z)",
        "SMA\n50 (Z)", "SMA\n200 (Z)",
        "Vol\nShk (Z)", "VT\nShk (Z)",
    ]

    # Column widths tuned for 13-column A4-landscape layout
    col_widths = [
        0.038, 0.062,          # Rank, Ticker
        0.075, 0.075,          # Rank Score, Fwd Return
        0.068, 0.068, 0.068,   # Mom 1M/3M/6M
        0.068, 0.068,          # Vol 21D, RSI 14
        0.068, 0.068,          # SMA 50, SMA 200
        0.068, 0.068,          # Vol Shock, VT Shock
    ]

    # ── Cell colours ─────────────────────────────────────────
    ty_vals  = snapshot["Target_Y"].values
    fwd_vals = snapshot["Forward_Return"].values * 100

    C_TOP_BG  = "#D6EAD9"   # soft sage green  — top quartile
    C_BOT_BG  = "#FAE3D0"   # soft apricot     — bottom quartile
    C_MID_BG  = "#F4F4F4"   # near-white       — middle
    C_ROW_ALT = "#EEF3F9"   # light blue-grey  — alternating rows
    C_POS_FWD = "#E2F0E6"   # light green      — positive return
    C_NEG_FWD = "#FAE8E8"   # light red        — negative return

    n_feat = len(STANDARDISED_FEATURES)   # = 9

    cell_colors = []
    for i in range(n):
        ty     = ty_vals[i]
        row_bg = "white" if i % 2 == 0 else C_ROW_ALT

        ty_bg  = C_TOP_BG if ty >= 0.75 else (C_BOT_BG if ty <= 0.25 else C_MID_BG)
        fwd    = fwd_vals[i]
        fwd_bg = C_POS_FWD if fwd > 0 else (C_NEG_FWD if fwd < 0 else row_bg)

        row_c = [
            row_bg,            # Rank
            ty_bg,             # Ticker (same shade as Rank Score)
            ty_bg,             # Rank Score (Target_Y)
            fwd_bg,            # Next-Day Return
        ] + [row_bg] * n_feat  # all 9 feature columns — plain alternating
        cell_colors.append(row_c)

    # ── Draw ─────────────────────────────────────────────────
    # Compact height: 0.26 in per row + header
    row_h   = 0.245
    fig_h   = max(7.5, n * row_h + 2.2)
    fig_w   = 11.7   # A4-landscape width in inches

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=disp.values.tolist(),
        colLabels=headers,
        cellColours=cell_colors,
        colWidths=col_widths,
        loc="upper center",
        cellLoc="center",
        bbox=[0, 0, 1, 1],   # fill entire axes
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.0)
    tbl.scale(1, row_h / 0.19)   # scale row height

    # Header styling
    HEADER_BG = "#1B3A6B"
    for col_idx in range(len(headers)):
        cell = tbl[(0, col_idx)]
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(color="white", fontweight="bold", fontsize=8.0)
        cell.set_edgecolor("white")
        cell.set_linewidth(0.8)

    # Data cell borders (thin, light grey)
    for (row_i, col_i), cell in tbl.get_celld().items():
        if row_i > 0:
            cell.set_edgecolor("#CCCCCC")
            cell.set_linewidth(0.35)

    # Ticker text colour reflects quartile
    for row_idx in range(1, n + 1):
        ty = ty_vals[row_idx - 1]
        tc = "#1A5C2A" if ty >= 0.75 else ("#A62828" if ty <= 0.25 else "#333333")
        tbl[(row_idx, 1)].set_text_props(fontweight="bold", color=tc, fontsize=8.0)
        tbl[(row_idx, 0)].set_text_props(color="#888888", fontsize=7.5)

    # ── Legend patches ────────────────────────────────────────
    patches = [
        mpatches.Patch(facecolor=C_TOP_BG, edgecolor="#AAAAAA", linewidth=0.5,
                       label="Top quartile  (Rank Score ≥ 0.75)"),
        mpatches.Patch(facecolor=C_BOT_BG, edgecolor="#AAAAAA", linewidth=0.5,
                       label="Bottom quartile  (Rank Score ≤ 0.25)"),
        mpatches.Patch(facecolor=C_POS_FWD, edgecolor="#AAAAAA", linewidth=0.5,
                       label="Positive next-day return"),
        mpatches.Patch(facecolor=C_NEG_FWD, edgecolor="#AAAAAA", linewidth=0.5,
                       label="Negative next-day return"),
    ]
    fig.legend(
        handles=patches, loc="lower center",
        ncol=4, fontsize=7.5, frameon=True,
        edgecolor="#CCCCCC", facecolor="white",
        bbox_to_anchor=(0.5, 0.0),
    )

    title_str = (
        f"Figure 4.8 — Cross-Sectional Ranking Snapshot: {closest.strftime('%d %B %Y')}\n"
        f"Active universe (N = {n} securities) ranked by Rank Score (Target_Y) — "
        f"highest to lowest.  "
        f"Z-scored features reflect each security's position relative to the cross-section on this date."
    )
    fig.suptitle(title_str, fontsize=9.5, fontweight="bold",
                 x=0.5, y=0.99, ha="center", va="top")

    out_path = FIGURE_DIR / "figure_4_8_ranking_snapshot.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 70)
    print("SCRIPT 04B: SUPERVISOR-REQUESTED DIAGNOSTIC FIGURES")
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print("=" * 70)

    broad, target, features = load_data()

    # Dynamically select tickers for demonstration based on their Target_Y
    showcase_tickers, ticker_labels, palette = get_showcase_tickers(target)

    plot_price_time_series(broad, showcase_tickers, ticker_labels, palette)
    plot_target_y_time_series(target, showcase_tickers, ticker_labels, palette)
    plot_feature_time_series(features)    # saves 4.7a, 4.7b, 4.7c
    plot_ranking_snapshot(features)

    print("\n" + "=" * 70)
    print("All figures saved to: outputs/figures/chapter_4/")
    print("=" * 70)


if __name__ == "__main__":
    main()
