"""
Script 06: Walk-Forward Validation Setup
----------------------------------------

Purpose
-------
Creates the common master modelling panel and the date-based expanding-window
validation schedule used by both Pooled OLS and XGBoost.

The study is a statistical comparison of cross-sectional ranking models.
No risk-free-rate or portfolio-performance variables are required in the
primary modelling pipeline.

Outer design
------------
- Initial out-of-sample quarter: 2024 Q1
- Test frequency: quarterly
- Training window: expanding
- Target-horizon purge: one trading date immediately before each test block

Inner XGBoost design
--------------------
For each outer training window, the final 378 eligible training dates are
split into three consecutive 126-date validation blocks.

Each inner fold:
1. uses all eligible earlier observations for training;
2. removes the trading date immediately preceding the validation block;
3. uses the later chronological block for validation; and
4. remains fully contained inside the corresponding outer training window.

The one-date purge is required because the target stored on feature date t
uses the adjusted closing price from the next market trading date.

Input
-----
1. JSE_Panel_Features_Engineered.csv

Outputs
-------
1. JSE_Master_Model_Panel.csv
2. JSE_Walk_Forward_Windows.csv
3. JSE_Inner_Validation_Windows.csv
4. outputs/audits/JSE_Walk_Forward_Setup_Summary.csv
5. outputs/audits/JSE_Model_Date_Index.csv
6. outputs/audits/JSE_Model_Feature_Configuration.csv
7. outputs/figures/figure_5_1_walk_forward_schedule.png

Required packages
-----------------
pip install pandas numpy matplotlib
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FEATURE_PANEL = Path(
    "JSE_Panel_Features_Engineered.csv"
)

OUTPUT_MASTER_PANEL = Path(
    "JSE_Master_Model_Panel.csv"
)

OUTPUT_OUTER_WINDOWS = Path(
    "JSE_Walk_Forward_Windows.csv"
)

OUTPUT_INNER_WINDOWS = Path(
    "JSE_Inner_Validation_Windows.csv"
)

AUDIT_DIRECTORY = Path("outputs") / "audits"
FIGURE_DIRECTORY = Path("outputs") / "figures"

OUTPUT_SUMMARY = (
    AUDIT_DIRECTORY
    / "JSE_Walk_Forward_Setup_Summary.csv"
)

OUTPUT_DATE_INDEX = (
    AUDIT_DIRECTORY
    / "JSE_Model_Date_Index.csv"
)

OUTPUT_FEATURE_CONFIGURATION = (
    AUDIT_DIRECTORY
    / "JSE_Model_Feature_Configuration.csv"
)

OUTPUT_FIGURE = (
    FIGURE_DIRECTORY
    / "figure_5_1_walk_forward_schedule.png"
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


# Reduced feature set retained as a possible robustness specification.
# It removes several of the most conceptually overlapping technical
# indicators while preserving the principal feature families.
REDUCED_ROBUSTNESS_FEATURES = [
    "Mom_1M",
    "Mom_3M",
    "Mom_6M",
    "Vol_21D",
    "Value_Traded_Shock_63D",
]


TARGET_COLUMN = "Target_Y"

INITIAL_TEST_QUARTER = pd.Period(
    "2024Q1",
    freq="Q",
)

# Target_Y on date t is based on the return from t to the next market date.
# Therefore, the date immediately preceding every test or validation block
# must be excluded from training.
TARGET_HORIZON_PURGE_DATES = 1

INNER_VALIDATION_FOLDS = 3
INNER_VALIDATION_DATES_PER_FOLD = 126
MINIMUM_INNER_TRAINING_DATES = 504

WINSOR_LOWER_QUANTILE = 0.01
WINSOR_UPPER_QUANTILE = 0.99


# ============================================================
# 2. VALIDATION AND UTILITY FUNCTIONS
# ============================================================

def validate_required_files() -> None:
    """Confirm that the required Script 05 output exists."""

    required_files = [
        INPUT_FEATURE_PANEL,
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


def validate_feature_panel(
    panel: pd.DataFrame,
) -> None:
    """
    Validate the final feature-engineered modelling panel.

    The panel must contain one observation per Date-Ticker pair,
    a valid target, a realised forward return and all nine primary
    predictors.
    """

    required_columns = {
        "Date",
        "Ticker",
        TARGET_COLUMN,
        "Forward_Return",
        "N_t",
        "Model_N_t",
        *PRIMARY_FEATURES,
    }

    missing_columns = required_columns.difference(
        panel.columns
    )

    if missing_columns:
        raise ValueError(
            "The feature panel is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if panel.empty:
        raise ValueError(
            "The feature panel contains no observations."
        )

    duplicate_count = int(
        panel.duplicated(
            ["Date", "Ticker"]
        ).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"The feature panel contains {duplicate_count:,} "
            "duplicate Date-Ticker observations."
        )

    required_nonmissing_columns = [
        "Date",
        "Ticker",
        TARGET_COLUMN,
        "Forward_Return",
        "N_t",
        "Model_N_t",
        *PRIMARY_FEATURES,
    ]

    missing_required_values = int(
        panel[
            required_nonmissing_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    if missing_required_values > 0:
        raise ValueError(
            f"The feature panel contains "
            f"{missing_required_values:,} missing required "
            "model values."
        )

    feature_array = panel[
        PRIMARY_FEATURES
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        feature_array
    ).all():
        raise ValueError(
            "Non-finite values exist in the primary "
            "feature matrix."
        )

    target_array = panel[
        TARGET_COLUMN
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        target_array
    ).all():
        raise ValueError(
            "Non-finite values exist in the target variable."
        )

    if not panel[
        TARGET_COLUMN
    ].between(
        0,
        1,
        inclusive="both",
    ).all():
        raise ValueError(
            "Target values outside [0, 1] were detected."
        )

    forward_return_array = panel[
        "Forward_Return"
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        forward_return_array
    ).all():
        raise ValueError(
            "Non-finite values exist in Forward_Return."
        )

    if (
        panel["Model_N_t"] < 2
    ).any():
        raise ValueError(
            "At least one modelling date contains fewer than "
            "two eligible securities."
        )

    observed_daily_counts = (
        panel.groupby(
            "Date"
        )["Ticker"]
        .nunique()
    )

    stored_daily_counts = (
        panel.groupby(
            "Date"
        )["Model_N_t"]
        .first()
    )

    if not observed_daily_counts.equals(
        stored_daily_counts.astype(int)
    ):
        raise AssertionError(
            "Stored Model_N_t values do not match the observed "
            "number of eligible securities on at least one date."
        )


def count_rows_between(
    panel: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> int:
    """Count panel rows between two dates, inclusive."""

    mask = panel[
        "Date"
    ].between(
        start_date,
        end_date,
        inclusive="both",
    )

    return int(
        mask.sum()
    )


def count_dates_between(
    dates: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> int:
    """Count unique model dates between two dates, inclusive."""

    mask = (
        (dates >= start_date)
        & (dates <= end_date)
    )

    return int(
        mask.sum()
    )


# ============================================================
# 3. OUTER WALK-FORWARD WINDOWS
# ============================================================

def construct_outer_windows(
    master_panel: pd.DataFrame,
    model_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Construct quarterly expanding-window outer test periods.

    Each test quarter is evaluated once. The training sample begins at
    the first eligible modelling date and expands through time.

    One market date is removed immediately before every test block to
    prevent a training target from using the first test date's closing
    price.
    """

    observed_periods = (
        model_dates
        .to_period("Q")
        .unique()
        .sort_values()
    )

    test_periods = [
        period
        for period in observed_periods
        if period >= INITIAL_TEST_QUARTER
    ]

    if not test_periods:
        raise ValueError(
            f"No model dates exist in or after "
            f"{INITIAL_TEST_QUARTER}."
        )

    maximum_observed_period = (
        observed_periods.max()
    )

    records: list[
        dict[str, object]
    ] = []

    for window_number, test_period in enumerate(
        test_periods,
        start=1,
    ):
        test_period_mask = (
            model_dates.to_period("Q")
            == test_period
        )

        test_dates = model_dates[
            test_period_mask
        ]

        if len(
            test_dates
        ) == 0:
            continue

        test_start = pd.Timestamp(
            test_dates.min()
        )

        test_end = pd.Timestamp(
            test_dates.max()
        )

        test_start_position = int(
            model_dates.get_loc(
                test_start
            )
        )

        minimum_required_position = (
            TARGET_HORIZON_PURGE_DATES
            + 1
        )

        if (
            test_start_position
            < minimum_required_position
        ):
            raise ValueError(
                f"Insufficient pre-test dates for "
                f"{test_period}."
            )

        # The date directly before the test block is purged.
        purge_position = (
            test_start_position
            - 1
        )

        # With a one-date purge, training ends one additional
        # model date before the purged date.
        train_end_position = (
            purge_position
            - TARGET_HORIZON_PURGE_DATES
        )

        train_start = pd.Timestamp(
            model_dates.min()
        )

        train_end = pd.Timestamp(
            model_dates[
                train_end_position
            ]
        )

        purge_date = pd.Timestamp(
            model_dates[
                purge_position
            ]
        )

        if not (
            train_end
            < purge_date
            < test_start
        ):
            raise AssertionError(
                "Outer date ordering is invalid."
            )

        train_date_count = (
            count_dates_between(
                model_dates,
                train_start,
                train_end,
            )
        )

        test_date_count = int(
            len(test_dates)
        )

        train_row_count = (
            count_rows_between(
                master_panel,
                train_start,
                train_end,
            )
        )

        test_row_count = (
            count_rows_between(
                master_panel,
                test_start,
                test_end,
            )
        )

        records.append(
            {
                "Outer_Window_ID": (
                    f"OW_{window_number:02d}"
                ),
                "Test_Quarter": str(
                    test_period
                ),
                "Train_Start": train_start,
                "Train_End": train_end,
                "Purged_Boundary_Date": (
                    purge_date
                ),
                "Test_Start": test_start,
                "Test_End": test_end,
                "Train_Dates": (
                    train_date_count
                ),
                "Train_Observations": (
                    train_row_count
                ),
                "Purged_Dates": (
                    TARGET_HORIZON_PURGE_DATES
                ),
                "Test_Dates": (
                    test_date_count
                ),
                "Test_Observations": (
                    test_row_count
                ),
                "Is_Complete_Calendar_Quarter": (
                    test_period
                    < maximum_observed_period
                ),
            }
        )

    outer_windows = pd.DataFrame(
        records
    )

    if outer_windows.empty:
        raise ValueError(
            "No outer walk-forward windows were constructed."
        )

    if not (
        outer_windows["Train_End"]
        < outer_windows[
            "Purged_Boundary_Date"
        ]
    ).all():
        raise AssertionError(
            "An outer training window reaches its purge date."
        )

    if not (
        outer_windows[
            "Purged_Boundary_Date"
        ]
        < outer_windows["Test_Start"]
    ).all():
        raise AssertionError(
            "An outer purge date is not earlier "
            "than its test period."
        )

    ordered_windows = (
        outer_windows
        .sort_values(
            "Test_Start"
        )
        .reset_index(
            drop=True
        )
    )

    for index in range(
        1,
        len(ordered_windows),
    ):
        previous_test_end = pd.Timestamp(
            ordered_windows.loc[
                index - 1,
                "Test_End",
            ]
        )

        current_test_start = pd.Timestamp(
            ordered_windows.loc[
                index,
                "Test_Start",
            ]
        )

        if (
            current_test_start
            <= previous_test_end
        ):
            raise AssertionError(
                "Outer test windows overlap."
            )

    return outer_windows


# ============================================================
# 4. INNER VALIDATION WINDOWS
# ============================================================

def construct_inner_windows(
    master_panel: pd.DataFrame,
    model_dates: pd.DatetimeIndex,
    outer_windows: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct three chronological validation folds within each
    outer training window.

    The final 378 eligible outer-training dates are divided into
    three consecutive 126-date validation blocks. Each fold uses
    an expanding inner training sample and a one-date purge.
    """

    total_validation_dates = (
        INNER_VALIDATION_FOLDS
        * INNER_VALIDATION_DATES_PER_FOLD
    )

    minimum_required_dates = (
        MINIMUM_INNER_TRAINING_DATES
        + total_validation_dates
        + TARGET_HORIZON_PURGE_DATES
    )

    records: list[
        dict[str, object]
    ] = []

    for outer_row in (
        outer_windows
        .itertuples(
            index=False
        )
    ):
        outer_train_start = pd.Timestamp(
            outer_row.Train_Start
        )

        outer_train_end = pd.Timestamp(
            outer_row.Train_End
        )

        outer_train_dates = model_dates[
            (
                model_dates
                >= outer_train_start
            )
            & (
                model_dates
                <= outer_train_end
            )
        ]

        if (
            len(outer_train_dates)
            < minimum_required_dates
        ):
            raise ValueError(
                f"{outer_row.Outer_Window_ID} contains only "
                f"{len(outer_train_dates):,} training dates; "
                f"{minimum_required_dates:,} are required."
            )

        first_validation_position = (
            len(outer_train_dates)
            - total_validation_dates
        )

        for fold_number in range(
            1,
            INNER_VALIDATION_FOLDS + 1,
        ):
            validation_start_position = (
                first_validation_position
                + (
                    fold_number - 1
                )
                * INNER_VALIDATION_DATES_PER_FOLD
            )

            validation_end_position = (
                validation_start_position
                + INNER_VALIDATION_DATES_PER_FOLD
                - 1
            )

            purge_position = (
                validation_start_position
                - 1
            )

            inner_train_end_position = (
                purge_position
                - TARGET_HORIZON_PURGE_DATES
            )

            if (
                inner_train_end_position
                < 0
            ):
                raise ValueError(
                    "The inner training window contains no dates."
                )

            inner_train_start = pd.Timestamp(
                outer_train_dates.min()
            )

            inner_train_end = pd.Timestamp(
                outer_train_dates[
                    inner_train_end_position
                ]
            )

            inner_purge_date = pd.Timestamp(
                outer_train_dates[
                    purge_position
                ]
            )

            validation_start = pd.Timestamp(
                outer_train_dates[
                    validation_start_position
                ]
            )

            validation_end = pd.Timestamp(
                outer_train_dates[
                    validation_end_position
                ]
            )

            if not (
                inner_train_end
                < inner_purge_date
                < validation_start
                <= validation_end
            ):
                raise AssertionError(
                    "Inner validation date ordering is invalid."
                )

            inner_train_date_count = (
                inner_train_end_position
                + 1
            )

            validation_date_count = (
                validation_end_position
                - validation_start_position
                + 1
            )

            inner_train_row_count = (
                count_rows_between(
                    master_panel,
                    inner_train_start,
                    inner_train_end,
                )
            )

            validation_row_count = (
                count_rows_between(
                    master_panel,
                    validation_start,
                    validation_end,
                )
            )

            records.append(
                {
                    "Outer_Window_ID": (
                        outer_row.Outer_Window_ID
                    ),
                    "Test_Quarter": (
                        outer_row.Test_Quarter
                    ),
                    "Inner_Fold_ID": (
                        f"{outer_row.Outer_Window_ID}"
                        f"_IF_{fold_number}"
                    ),
                    "Inner_Fold_Number": (
                        fold_number
                    ),
                    "Inner_Train_Start": (
                        inner_train_start
                    ),
                    "Inner_Train_End": (
                        inner_train_end
                    ),
                    "Inner_Purged_Boundary_Date": (
                        inner_purge_date
                    ),
                    "Validation_Start": (
                        validation_start
                    ),
                    "Validation_End": (
                        validation_end
                    ),
                    "Inner_Train_Dates": (
                        inner_train_date_count
                    ),
                    "Inner_Train_Observations": (
                        inner_train_row_count
                    ),
                    "Inner_Purged_Dates": (
                        TARGET_HORIZON_PURGE_DATES
                    ),
                    "Validation_Dates": (
                        validation_date_count
                    ),
                    "Validation_Observations": (
                        validation_row_count
                    ),
                }
            )

    inner_windows = pd.DataFrame(
        records
    )

    expected_inner_rows = (
        len(outer_windows)
        * INNER_VALIDATION_FOLDS
    )

    if (
        len(inner_windows)
        != expected_inner_rows
    ):
        raise AssertionError(
            "The number of inner folds is incorrect."
        )

    if not (
        inner_windows[
            "Inner_Train_End"
        ]
        < inner_windows[
            "Inner_Purged_Boundary_Date"
        ]
    ).all():
        raise AssertionError(
            "An inner training window reaches its purge date."
        )

    if not (
        inner_windows[
            "Inner_Purged_Boundary_Date"
        ]
        < inner_windows[
            "Validation_Start"
        ]
    ).all():
        raise AssertionError(
            "An inner purge date reaches its validation block."
        )

    return inner_windows


# ============================================================
# 5. TARGET-BOUNDARY LEAKAGE CHECKS
# ============================================================

def validate_target_boundaries(
    master_panel: pd.DataFrame,
    outer_windows: pd.DataFrame,
    inner_windows: pd.DataFrame,
) -> None:
    """
    Confirm that training targets do not use prices belonging to
    the subsequent test or validation block.

    This validation is performed when Script 05 retained the Next_Date
    field created during target engineering.
    """

    if "Next_Date" not in master_panel.columns:
        print(
            "Warning: Next_Date is not present in the feature panel. "
            "Schedule-based purge checks were completed, but direct "
            "target-boundary verification could not be performed."
        )
        return

    for outer_row in (
        outer_windows
        .itertuples(
            index=False
        )
    ):
        training_rows = master_panel.loc[
            master_panel["Date"].between(
                pd.Timestamp(
                    outer_row.Train_Start
                ),
                pd.Timestamp(
                    outer_row.Train_End
                ),
                inclusive="both",
            )
        ]

        leakage_count = int(
            (
                training_rows["Next_Date"]
                >= pd.Timestamp(
                    outer_row.Test_Start
                )
            ).sum()
        )

        if leakage_count > 0:
            raise AssertionError(
                f"{leakage_count:,} target-boundary leakage "
                f"rows detected in "
                f"{outer_row.Outer_Window_ID}."
            )

    for inner_row in (
        inner_windows
        .itertuples(
            index=False
        )
    ):
        inner_training_rows = master_panel.loc[
            master_panel["Date"].between(
                pd.Timestamp(
                    inner_row.Inner_Train_Start
                ),
                pd.Timestamp(
                    inner_row.Inner_Train_End
                ),
                inclusive="both",
            )
        ]

        inner_leakage_count = int(
            (
                inner_training_rows["Next_Date"]
                >= pd.Timestamp(
                    inner_row.Validation_Start
                )
            ).sum()
        )

        if inner_leakage_count > 0:
            raise AssertionError(
                f"{inner_leakage_count:,} inner "
                f"target-boundary leakage rows detected in "
                f"{inner_row.Inner_Fold_ID}."
            )


# ============================================================
# 6. FIGURE
# ============================================================

def create_walk_forward_figure(
    outer_windows: pd.DataFrame,
) -> None:
    """Create the quarterly expanding-window validation diagram."""

    plotting_windows = (
        outer_windows
        .sort_values(
            "Test_Start"
        )
        .reset_index(
            drop=True
        )
    )

    figure_height = max(
        5.5,
        len(plotting_windows)
        * 0.55,
    )

    figure, axis = plt.subplots(
        figsize=(
            12,
            figure_height,
        )
    )

    for y_position, row in (
        plotting_windows
        .iterrows()
    ):
        axis.hlines(
            y=y_position,
            xmin=pd.Timestamp(
                row["Train_Start"]
            ),
            xmax=pd.Timestamp(
                row["Train_End"]
            ),
            linewidth=6,
            linestyles="solid",
        )

        axis.hlines(
            y=y_position,
            xmin=pd.Timestamp(
                row["Test_Start"]
            ),
            xmax=pd.Timestamp(
                row["Test_End"]
            ),
            linewidth=6,
            linestyles="dashed",
        )

        axis.scatter(
            pd.Timestamp(
                row[
                    "Purged_Boundary_Date"
                ]
            ),
            y_position,
            marker="x",
            s=45,
            zorder=3,
        )

    axis.set_yticks(
        np.arange(
            len(plotting_windows)
        )
    )

    axis.set_yticklabels(
        plotting_windows[
            "Test_Quarter"
        ]
    )

    axis.set_title(
        "Quarterly Expanding-Window "
        "Out-of-Sample Design"
    )

    axis.set_xlabel(
        "Calendar date"
    )

    axis.set_ylabel(
        "Out-of-sample test quarter"
    )

    axis.xaxis.set_major_locator(
        mdates.YearLocator()
    )

    axis.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%Y"
        )
    )

    axis.grid(
        axis="x",
        linestyle="--",
        alpha=0.4,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="solid",
            linewidth=5,
            label=(
                "Expanding training period"
            ),
        ),
        Line2D(
            [0],
            [0],
            linestyle="dashed",
            linewidth=5,
            label=(
                "Quarterly test period"
            ),
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            markersize=8,
            label=(
                "Purged boundary date"
            ),
        ),
    ]

    axis.legend(
        handles=legend_handles,
        loc="lower right",
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


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def main() -> None:
    print(
        "=" * 78
    )

    print(
        "SCRIPT 06: WALK-FORWARD VALIDATION SETUP"
    )

    print(
        "=" * 78
    )

    validate_required_files()

    AUDIT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 7.1 Load and validate feature-engineered panel
    # --------------------------------------------------------

    feature_panel = pd.read_csv(
        INPUT_FEATURE_PANEL
    )

    feature_panel["Date"] = pd.to_datetime(
        feature_panel["Date"],
        errors="raise",
    )

    if (
        "Next_Date"
        in feature_panel.columns
    ):
        feature_panel[
            "Next_Date"
        ] = pd.to_datetime(
            feature_panel[
                "Next_Date"
            ],
            errors="coerce",
        )

    feature_panel = (
        feature_panel
        .sort_values(
            ["Date", "Ticker"]
        )
        .reset_index(
            drop=True
        )
    )

    validate_feature_panel(
        feature_panel
    )

    feature_rows_loaded = int(
        len(feature_panel)
    )

    # The feature-engineered panel is now the common master
    # modelling panel. No macroeconomic or risk-free-rate
    # variables are required for the statistical comparison.
    master_panel = (
        feature_panel
        .copy()
    )

    # --------------------------------------------------------
    # 7.2 Build unique model-date index
    # --------------------------------------------------------

    model_dates = pd.DatetimeIndex(
        master_panel[
            "Date"
        ]
        .drop_duplicates()
        .sort_values()
    )

    if (
        len(model_dates)
        == 0
    ):
        raise ValueError(
            "No eligible model dates were found."
        )

    date_index = pd.DataFrame(
        {
            "Date": model_dates,
            "Model_Date_ID": np.arange(
                len(model_dates)
            ),
        }
    )

    date_index[
        "Calendar_Quarter"
    ] = (
        date_index["Date"]
        .dt
        .to_period("Q")
        .astype(str)
    )

    date_index[
        "Calendar_Year"
    ] = (
        date_index["Date"]
        .dt
        .year
    )

    master_panel = master_panel.merge(
        date_index[
            [
                "Date",
                "Model_Date_ID",
                "Calendar_Quarter",
                "Calendar_Year",
            ]
        ],
        on="Date",
        how="left",
        validate="many_to_one",
    )

    if (
        len(master_panel)
        != feature_rows_loaded
    ):
        raise AssertionError(
            "Adding the model-date index changed the number "
            "of feature-panel observations."
        )

    if (
        master_panel[
            "Model_Date_ID"
        ]
        .isna()
        .any()
    ):
        raise AssertionError(
            "At least one observation did not receive "
            "a model-date index."
        )

    master_panel[
        "Model_Date_ID"
    ] = master_panel[
        "Model_Date_ID"
    ].astype(int)

    master_panel = (
        master_panel
        .sort_values(
            ["Date", "Ticker"]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # 7.3 Construct outer and inner validation schedules
    # --------------------------------------------------------

    outer_windows = (
        construct_outer_windows(
            master_panel,
            model_dates,
        )
    )

    inner_windows = (
        construct_inner_windows(
            master_panel,
            model_dates,
            outer_windows,
        )
    )

    # --------------------------------------------------------
    # 7.4 Confirm target-boundary integrity
    # --------------------------------------------------------

    validate_target_boundaries(
        master_panel,
        outer_windows,
        inner_windows,
    )

    # --------------------------------------------------------
    # 7.5 Confirm full out-of-sample coverage
    # --------------------------------------------------------

    first_window = (
        outer_windows
        .sort_values(
            "Test_Start"
        )
        .iloc[0]
    )

    last_window = (
        outer_windows
        .sort_values(
            "Test_Start"
        )
        .iloc[-1]
    )

    first_test_start_timestamp = pd.Timestamp(
        first_window[
            "Test_Start"
        ]
    )

    total_test_observations = int(
        outer_windows[
            "Test_Observations"
        ].sum()
    )

    total_test_dates = int(
        outer_windows[
            "Test_Dates"
        ].sum()
    )

    expected_out_of_sample_rows = int(
        (
            master_panel["Date"]
            >= first_test_start_timestamp
        ).sum()
    )

    expected_out_of_sample_dates = int(
        master_panel.loc[
            (
                master_panel["Date"]
                >= first_test_start_timestamp
            ),
            "Date",
        ].nunique()
    )

    if (
        total_test_observations
        != expected_out_of_sample_rows
    ):
        raise AssertionError(
            "The quarterly outer windows do not cover the "
            "complete out-of-sample observation set."
        )

    if (
        total_test_dates
        != expected_out_of_sample_dates
    ):
        raise AssertionError(
            "The quarterly outer windows do not cover the "
            "complete out-of-sample date set."
        )

    # Confirm that each out-of-sample date belongs to exactly
    # one test window.
    out_of_sample_date_membership = []

    for row in (
        outer_windows
        .itertuples(
            index=False
        )
    ):
        dates_in_window = model_dates[
            (
                model_dates
                >= pd.Timestamp(
                    row.Test_Start
                )
            )
            & (
                model_dates
                <= pd.Timestamp(
                    row.Test_End
                )
            )
        ]

        out_of_sample_date_membership.extend(
            dates_in_window.tolist()
        )

    out_of_sample_date_membership = (
        pd.Series(
            out_of_sample_date_membership
        )
    )

    if (
        out_of_sample_date_membership
        .duplicated()
        .any()
    ):
        raise AssertionError(
            "At least one out-of-sample date belongs to "
            "more than one outer test window."
        )

    # --------------------------------------------------------
    # 7.6 Save feature configuration
    # --------------------------------------------------------

    feature_configuration = pd.DataFrame(
        [
            {
                "Feature_Set": (
                    "Primary_Full_9"
                ),
                "Feature": feature,
                "Used_In_Primary_OLS": True,
                "Used_In_Primary_XGBoost": True,
                "Used_In_Reduced_Robustness_Model": (
                    feature
                    in REDUCED_ROBUSTNESS_FEATURES
                ),
                "Winsor_Lower_Quantile": (
                    WINSOR_LOWER_QUANTILE
                ),
                "Winsor_Upper_Quantile": (
                    WINSOR_UPPER_QUANTILE
                ),
                "Winsorisation_Timing": (
                    "Estimated separately within each "
                    "outer or inner training window"
                ),
            }
            for feature in PRIMARY_FEATURES
        ]
    )

    # --------------------------------------------------------
    # 7.7 Create walk-forward figure
    # --------------------------------------------------------

    create_walk_forward_figure(
        outer_windows
    )

    # --------------------------------------------------------
    # 7.8 Construct summary audit
    # --------------------------------------------------------

    initial_train_start = pd.Timestamp(
        first_window[
            "Train_Start"
        ]
    ).date()

    initial_train_end = pd.Timestamp(
        first_window[
            "Train_End"
        ]
    ).date()

    initial_purge_date = pd.Timestamp(
        first_window[
            "Purged_Boundary_Date"
        ]
    ).date()

    initial_test_start = pd.Timestamp(
        first_window[
            "Test_Start"
        ]
    ).date()

    final_test_end = pd.Timestamp(
        last_window[
            "Test_End"
        ]
    ).date()

    summary = pd.DataFrame(
        {
            "Statistic": [
                "Master model observations",
                "Master model dates",
                "Master-panel start date",
                "Master-panel end date",
                "Minimum daily model cross-section",
                "Mean daily model cross-section",
                "Maximum daily model cross-section",
                "Primary predictors",
                "Initial outer test quarter",
                "Outer walk-forward windows",
                "Inner validation folds per outer window",
                "Total inner validation folds",
                "Validation dates per inner fold",
                "Minimum inner training dates",
                "Target-horizon purge dates",
                "Initial training start date",
                "Initial training end date",
                "Initial purged boundary date",
                "Initial test start date",
                "Final test end date",
                "Total out-of-sample dates",
                "Total out-of-sample observations",
                "Winsorisation lower quantile",
                "Winsorisation upper quantile",
            ],
            "Value": [
                len(master_panel),
                len(model_dates),
                master_panel[
                    "Date"
                ].min().date(),
                master_panel[
                    "Date"
                ].max().date(),
                int(
                    master_panel
                    .groupby(
                        "Date"
                    )["Ticker"]
                    .nunique()
                    .min()
                ),
                float(
                    master_panel
                    .groupby(
                        "Date"
                    )["Ticker"]
                    .nunique()
                    .mean()
                ),
                int(
                    master_panel
                    .groupby(
                        "Date"
                    )["Ticker"]
                    .nunique()
                    .max()
                ),
                len(PRIMARY_FEATURES),
                str(
                    INITIAL_TEST_QUARTER
                ),
                len(outer_windows),
                INNER_VALIDATION_FOLDS,
                len(inner_windows),
                INNER_VALIDATION_DATES_PER_FOLD,
                MINIMUM_INNER_TRAINING_DATES,
                TARGET_HORIZON_PURGE_DATES,
                initial_train_start,
                initial_train_end,
                initial_purge_date,
                initial_test_start,
                final_test_end,
                total_test_dates,
                total_test_observations,
                WINSOR_LOWER_QUANTILE,
                WINSOR_UPPER_QUANTILE,
            ],
        }
    )

    # --------------------------------------------------------
    # 7.9 Save all outputs
    # --------------------------------------------------------

    master_panel.to_csv(
        OUTPUT_MASTER_PANEL,
        index=False,
    )

    outer_windows.to_csv(
        OUTPUT_OUTER_WINDOWS,
        index=False,
    )

    inner_windows.to_csv(
        OUTPUT_INNER_WINDOWS,
        index=False,
    )

    date_index.to_csv(
        OUTPUT_DATE_INDEX,
        index=False,
    )

    feature_configuration.to_csv(
        OUTPUT_FEATURE_CONFIGURATION,
        index=False,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # --------------------------------------------------------
    # 7.10 Terminal output
    # --------------------------------------------------------

    master_start_text = str(
        master_panel[
            "Date"
        ].min().date()
    )

    master_end_text = str(
        master_panel[
            "Date"
        ].max().date()
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "SCRIPT 06 EMPIRICAL OUTPUTS"
    )

    print(
        "=" * 78
    )

    print(
        "Feature-panel observations loaded: "
        f"{feature_rows_loaded:,}"
    )

    print(
        "Master model-panel observations: "
        f"{len(master_panel):,}"
    )

    print(
        "Master model-panel trading dates: "
        f"{len(model_dates):,}"
    )

    print(
        "Master-panel date range: "
        f"{master_start_text} to "
        f"{master_end_text}"
    )

    print(
        "Primary predictor count: "
        f"{len(PRIMARY_FEATURES):,}"
    )

    print(
        "Outer expanding-window test periods: "
        f"{len(outer_windows):,}"
    )

    print(
        "Inner validation folds created: "
        f"{len(inner_windows):,}"
    )

    print(
        "Initial training period: "
        f"{initial_train_start} to "
        f"{initial_train_end}"
    )

    print(
        "Initial purged boundary date: "
        f"{initial_purge_date}"
    )

    print(
        "Initial out-of-sample test date: "
        f"{initial_test_start}"
    )

    print(
        "Final out-of-sample test date: "
        f"{final_test_end}"
    )

    print(
        "Total out-of-sample trading dates: "
        f"{total_test_dates:,}"
    )

    print(
        "Total out-of-sample observations: "
        f"{total_test_observations:,}"
    )

    print(
        "\nOuter walk-forward windows:"
    )

    for row in (
        outer_windows
        .itertuples(
            index=False
        )
    ):
        completion_label = (
            "complete quarter"
            if row.Is_Complete_Calendar_Quarter
            else "partial final quarter"
        )

        train_end_text = str(
            pd.Timestamp(
                row.Train_End
            ).date()
        )

        purge_date_text = str(
            pd.Timestamp(
                row.Purged_Boundary_Date
            ).date()
        )

        test_start_text = str(
            pd.Timestamp(
                row.Test_Start
            ).date()
        )

        test_end_text = str(
            pd.Timestamp(
                row.Test_End
            ).date()
        )

        print(
            f"- {row.Outer_Window_ID} | "
            f"{row.Test_Quarter} | "
            f"train through {train_end_text} | "
            f"purge {purge_date_text} | "
            f"test {test_start_text} to "
            f"{test_end_text} | "
            f"{row.Test_Dates:,} dates | "
            f"{row.Test_Observations:,} rows | "
            f"{completion_label}"
        )

    print(
        "\nIntegrity checks passed:"
    )

    print(
        "- The master panel contains one observation per "
        "Date-Ticker pair."
    )

    print(
        "- The master-panel observation count is identical "
        "to the final feature-panel observation count."
    )

    print(
        "- Outer test windows were constructed from unique "
        "model trading dates."
    )

    print(
        "- Every outer boundary contains a one-date "
        "target-horizon purge."
    )

    print(
        "- Every XGBoost inner fold uses a later chronological "
        "validation block."
    )

    print(
        "- Every inner training-validation boundary contains "
        "a one-date target-horizon purge."
    )

    print(
        "- The quarterly test windows cover the complete "
        "2024-onward model sample without overlap."
    )

    print(
        "- Winsorisation parameters will be estimated inside "
        "each training window rather than globally."
    )

    print(
        "\nCore outputs:"
    )

    print(
        f"- {OUTPUT_MASTER_PANEL}"
    )

    print(
        f"- {OUTPUT_OUTER_WINDOWS}"
    )

    print(
        f"- {OUTPUT_INNER_WINDOWS}"
    )

    print(
        "\nAudit and figure outputs:"
    )

    print(
        f"- {OUTPUT_SUMMARY}"
    )

    print(
        f"- {OUTPUT_DATE_INDEX}"
    )

    print(
        f"- {OUTPUT_FEATURE_CONFIGURATION}"
    )

    print(
        f"- {OUTPUT_FIGURE}"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()