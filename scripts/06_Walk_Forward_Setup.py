"""
Script 06: Walk-Forward Validation Setup
----------------------------------------

Purpose
-------
Creates the date-based expanding-window validation schedule used by both
Pooled Cross-Sectional OLS and XGBoost.

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

1. uses all earlier dates for training;
2. removes the immediately preceding trading date as a purge; and
3. evaluates the candidate configuration on the later validation block.

Input
-----
data/processed/05_features/JSE_Panel_Features_Engineered.csv

Core outputs
------------
data/processed/06_validation/JSE_Master_Model_Panel.csv
data/processed/06_validation/JSE_Walk_Forward_Windows.csv
data/processed/06_validation/JSE_Inner_Validation_Windows.csv

Audit outputs
-------------
outputs/audits/06_validation/JSE_Walk_Forward_Setup_Summary.csv
outputs/audits/06_validation/JSE_Model_Date_Index.csv
outputs/audits/06_validation/JSE_Model_Feature_Configuration.csv

Figure output
-------------
outputs/figures/chapter_5/figure_5_1_walk_forward_schedule.png

Run from the project root
-------------------------
python .\\scripts\\06_Walk_Forward_Setup.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


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

DATA_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "06_validation"
)

AUDIT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "audits"
    / "06_validation"
)

FIGURE_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "chapter_5"
)

OUTPUT_MASTER_PANEL = (
    DATA_OUTPUT_DIRECTORY
    / "JSE_Master_Model_Panel.csv"
)

OUTPUT_OUTER_WINDOWS = (
    DATA_OUTPUT_DIRECTORY
    / "JSE_Walk_Forward_Windows.csv"
)

OUTPUT_INNER_WINDOWS = (
    DATA_OUTPUT_DIRECTORY
    / "JSE_Inner_Validation_Windows.csv"
)

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


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

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

TARGET_HORIZON_PURGE_DATES = 1

INNER_VALIDATION_FOLDS = 3

INNER_VALIDATION_DATES_PER_FOLD = 126

MINIMUM_INNER_TRAINING_DATES = 504

WINSOR_LOWER_QUANTILE = 0.01

WINSOR_UPPER_QUANTILE = 0.99


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


def validate_required_file() -> None:
    """Confirm that the final feature panel exists."""

    if not INPUT_FEATURE_PANEL.exists():
        raise FileNotFoundError(
            "Required input file is missing:\n"
            f"{INPUT_FEATURE_PANEL.resolve()}"
        )


def validate_feature_panel(
    panel: pd.DataFrame,
) -> None:
    """Validate the final modelling panel."""

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
            [
                "Date",
                "Ticker",
            ]
        ).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"The feature panel contains {duplicate_count:,} duplicate "
            "Date-Ticker observations."
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
            f"The feature panel contains {missing_required_values:,} "
            "missing required model values."
        )

    numeric_matrix = panel[
        [
            TARGET_COLUMN,
            "Forward_Return",
            *PRIMARY_FEATURES,
        ]
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        numeric_matrix
    ).all():
        raise ValueError(
            "Non-finite values exist in the model matrix."
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

    observed_model_sizes = (
        panel
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

    if not panel[
        "Model_N_t"
    ].astype(
        int
    ).eq(
        observed_model_sizes
    ).all():
        raise ValueError(
            "Stored Model_N_t does not match the observed daily "
            "model cross-sectional size."
        )


def count_rows_between(
    panel: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> int:
    """Count observations between two dates, inclusive."""

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
    """Count distinct model dates between two dates, inclusive."""

    mask = (
        dates
        >= start_date
    ) & (
        dates
        <= end_date
    )

    return int(
        mask.sum()
    )


# ============================================================
# 4. OUTER WALK-FORWARD WINDOWS
# ============================================================

def construct_outer_windows(
    master_panel: pd.DataFrame,
    model_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Construct quarterly expanding outer test windows."""

    observed_periods = pd.PeriodIndex(
        model_dates.to_period(
            "Q"
        ).unique()
    ).sort_values()

    test_periods = [
        period
        for period in observed_periods
        if period >= INITIAL_TEST_QUARTER
    ]

    if not test_periods:
        raise ValueError(
            "No model dates exist in or after "
            f"{INITIAL_TEST_QUARTER}."
        )

    maximum_observed_period = observed_periods.max()

    records: list[dict[str, object]] = []

    for window_number, test_period in enumerate(
        test_periods,
        start=1,
    ):
        test_period_mask = (
            model_dates.to_period(
                "Q"
            )
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

        if test_start_position < minimum_required_position:
            raise ValueError(
                f"Insufficient pre-test dates for {test_period}."
            )

        purge_position = (
            test_start_position
            - 1
        )

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

        train_date_count = count_dates_between(
            dates=model_dates,
            start_date=train_start,
            end_date=train_end,
        )

        test_date_count = int(
            len(
                test_dates
            )
        )

        train_row_count = count_rows_between(
            panel=master_panel,
            start_date=train_start,
            end_date=train_end,
        )

        test_row_count = count_rows_between(
            panel=master_panel,
            start_date=test_start,
            end_date=test_end,
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
                "Purged_Boundary_Date": purge_date,
                "Test_Start": test_start,
                "Test_End": test_end,
                "Train_Dates": train_date_count,
                "Train_Observations": train_row_count,
                "Purged_Dates": (
                    TARGET_HORIZON_PURGE_DATES
                ),
                "Test_Dates": test_date_count,
                "Test_Observations": test_row_count,
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
        outer_windows[
            "Train_End"
        ]
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
        < outer_windows[
            "Test_Start"
        ]
    ).all():
        raise AssertionError(
            "An outer purge date is not earlier than its test block."
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
        len(
            ordered_windows
        ),
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

        if current_test_start <= previous_test_end:
            raise AssertionError(
                "Outer test windows overlap."
            )

    return outer_windows


# ============================================================
# 5. INNER XGBOOST VALIDATION WINDOWS
# ============================================================

def construct_inner_windows(
    master_panel: pd.DataFrame,
    model_dates: pd.DatetimeIndex,
    outer_windows: pd.DataFrame,
) -> pd.DataFrame:
    """Construct nested chronological validation folds."""

    total_validation_dates = (
        INNER_VALIDATION_FOLDS
        * INNER_VALIDATION_DATES_PER_FOLD
    )

    minimum_required_dates = (
        MINIMUM_INNER_TRAINING_DATES
        + total_validation_dates
        + TARGET_HORIZON_PURGE_DATES
    )

    records: list[dict[str, object]] = []

    for outer_row in outer_windows.itertuples(
        index=False
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

        if len(
            outer_train_dates
        ) < minimum_required_dates:
            raise ValueError(
                f"{outer_row.Outer_Window_ID} contains only "
                f"{len(outer_train_dates):,} training dates; "
                f"{minimum_required_dates:,} are required."
            )

        first_validation_position = (
            len(
                outer_train_dates
            )
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

            if inner_train_end_position < 0:
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

            inner_train_row_count = count_rows_between(
                panel=master_panel,
                start_date=inner_train_start,
                end_date=inner_train_end,
            )

            validation_row_count = count_rows_between(
                panel=master_panel,
                start_date=validation_start,
                end_date=validation_end,
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
                    "Inner_Fold_Number": fold_number,
                    "Inner_Train_Start": inner_train_start,
                    "Inner_Train_End": inner_train_end,
                    "Inner_Purged_Boundary_Date": (
                        inner_purge_date
                    ),
                    "Validation_Start": validation_start,
                    "Validation_End": validation_end,
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
        len(
            outer_windows
        )
        * INNER_VALIDATION_FOLDS
    )

    if len(
        inner_windows
    ) != expected_inner_rows:
        raise AssertionError(
            "The number of inner validation folds is incorrect."
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
            "An inner training block reaches its purge date."
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
            "An inner purge date is not earlier than validation."
        )

    return inner_windows


# ============================================================
# 6. WALK-FORWARD FIGURE
# ============================================================

def create_walk_forward_figure(
    outer_windows: pd.DataFrame,
) -> None:
    """Create a thesis-ready walk-forward schedule figure."""

    plotting_windows = (
        outer_windows
        .sort_values(
            "Test_Start"
        )
        .reset_index(
            drop=True
        )
    )

    default_colours = (
        plt.rcParams[
            "axes.prop_cycle"
        ]
        .by_key()
        .get(
            "color",
            [
                "black",
                "grey",
                "silver",
            ],
        )
    )

    training_colour = default_colours[
        0
    ]

    test_colour = default_colours[
        1
        if len(
            default_colours
        ) > 1
        else 0
    ]

    purge_colour = default_colours[
        2
        if len(
            default_colours
        ) > 2
        else 0
    ]

    figure_height = max(
        5.5,
        len(
            plotting_windows
        )
        * 0.60,
    )

    figure, axis = plt.subplots(
        figsize=(
            12,
            figure_height,
        )
    )

    for y_position, row in plotting_windows.iterrows():
        axis.hlines(
            y=y_position,
            xmin=pd.Timestamp(
                row[
                    "Train_Start"
                ]
            ),
            xmax=pd.Timestamp(
                row[
                    "Train_End"
                ]
            ),
            linewidth=6,
            linestyle="solid",
            color=training_colour,
        )

        axis.hlines(
            y=y_position,
            xmin=pd.Timestamp(
                row[
                    "Test_Start"
                ]
            ),
            xmax=pd.Timestamp(
                row[
                    "Test_End"
                ]
            ),
            linewidth=6,
            linestyle="solid",
            color=test_colour,
        )

        axis.scatter(
            pd.Timestamp(
                row[
                    "Purged_Boundary_Date"
                ]
            ),
            y_position,
            marker="x",
            s=55,
            linewidths=1.8,
            color=purge_colour,
            zorder=3,
        )

    axis.set_yticks(
        np.arange(
            len(
                plotting_windows
            )
        )
    )

    axis.set_yticklabels(
        [
            (
                f"{window_id}  |  "
                f"{test_quarter}"
            )
            for window_id, test_quarter in zip(
                plotting_windows[
                    "Outer_Window_ID"
                ],
                plotting_windows[
                    "Test_Quarter"
                ],
            )
        ]
    )

    axis.set_title(
        "Quarterly Expanding-Window "
        "Out-of-Sample Validation Design"
    )

    axis.set_xlabel(
        "Calendar date"
    )

    axis.set_ylabel(
        "Outer test window"
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
            [
                0
            ],
            [
                0
            ],
            linestyle="solid",
            linewidth=6,
            color=training_colour,
            label="Expanding training period",
        ),
        Line2D(
            [
                0
            ],
            [
                0
            ],
            linestyle="solid",
            linewidth=6,
            color=test_colour,
            label="Quarterly test period",
        ),
        Line2D(
            [
                0
            ],
            [
                0
            ],
            marker="x",
            linestyle="None",
            markersize=8,
            markeredgewidth=1.8,
            color=purge_colour,
            label="Purged boundary date",
        ),
    ]

    axis.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(
            1.01,
            1.0,
        ),
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
    print("=" * 78)
    print("SCRIPT 06: WALK-FORWARD VALIDATION SETUP")
    print("=" * 78)

    validate_required_file()

    DATA_OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 7.1 Load the final feature panel
    # --------------------------------------------------------

    feature_panel = pd.read_csv(
        INPUT_FEATURE_PANEL
    )

    feature_panel[
        "Date"
    ] = pd.to_datetime(
        feature_panel[
            "Date"
        ],
        errors="raise",
    )

    if "Next_Date" in feature_panel.columns:
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
            [
                "Date",
                "Ticker",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    validate_feature_panel(
        feature_panel
    )

    feature_rows_loaded = int(
        len(
            feature_panel
        )
    )

    # No risk-free-rate or macroeconomic merge is required in
    # the reduced statistical-comparison study.
    master_panel = feature_panel.copy()

    if len(
        master_panel
    ) != feature_rows_loaded:
        raise AssertionError(
            "The master-panel construction changed the "
            "number of feature observations."
        )

    # --------------------------------------------------------
    # 7.2 Build the unique model-date index
    # --------------------------------------------------------

    model_dates = pd.DatetimeIndex(
        master_panel[
            "Date"
        ]
        .drop_duplicates()
        .sort_values()
    )

    date_index = pd.DataFrame(
        {
            "Date": model_dates,
            "Model_Date_ID": np.arange(
                len(
                    model_dates
                ),
                dtype=int,
            ),
        }
    )

    date_index[
        "Calendar_Quarter"
    ] = (
        date_index[
            "Date"
        ]
        .dt.to_period(
            "Q"
        )
        .astype(
            str
        )
    )

    date_index[
        "Calendar_Year"
    ] = date_index[
        "Date"
    ].dt.year

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

    if master_panel[
        "Model_Date_ID"
    ].isna().any():
        raise AssertionError(
            "At least one observation did not receive "
            "a model-date index."
        )

    master_panel[
        "Model_Date_ID"
    ] = master_panel[
        "Model_Date_ID"
    ].astype(
        int
    )

    master_panel = (
        master_panel
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

    if master_panel.duplicated(
        [
            "Date",
            "Ticker",
        ]
    ).any():
        raise AssertionError(
            "Duplicate Date-Ticker observations exist "
            "in the master model panel."
        )

    # --------------------------------------------------------
    # 7.3 Construct outer and inner schedules
    # --------------------------------------------------------

    outer_windows = construct_outer_windows(
        master_panel=master_panel,
        model_dates=model_dates,
    )

    inner_windows = construct_inner_windows(
        master_panel=master_panel,
        model_dates=model_dates,
        outer_windows=outer_windows,
    )

    # --------------------------------------------------------
    # 7.4 Explicit target-boundary leakage checks
    # --------------------------------------------------------

    if "Next_Date" in master_panel.columns:
        for outer_row in outer_windows.itertuples(
            index=False
        ):
            training_rows = master_panel.loc[
                master_panel[
                    "Date"
                ].between(
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
                    training_rows[
                        "Next_Date"
                    ]
                    >= pd.Timestamp(
                        outer_row.Test_Start
                    )
                ).sum()
            )

            if leakage_count > 0:
                raise AssertionError(
                    f"{leakage_count:,} target-boundary leakage rows "
                    f"were detected in {outer_row.Outer_Window_ID}."
                )

        for inner_row in inner_windows.itertuples(
            index=False
        ):
            inner_training_rows = master_panel.loc[
                master_panel[
                    "Date"
                ].between(
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
                    inner_training_rows[
                        "Next_Date"
                    ]
                    >= pd.Timestamp(
                        inner_row.Validation_Start
                    )
                ).sum()
            )

            if inner_leakage_count > 0:
                raise AssertionError(
                    f"{inner_leakage_count:,} inner target-boundary "
                    "leakage rows were detected in "
                    f"{inner_row.Inner_Fold_ID}."
                )

    # --------------------------------------------------------
    # 7.5 Confirm complete out-of-sample coverage
    # --------------------------------------------------------

    first_window = outer_windows.iloc[
        0
    ]

    last_window = outer_windows.iloc[
        -1
    ]

    first_test_start = pd.Timestamp(
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
            master_panel[
                "Date"
            ]
            >= first_test_start
        ).sum()
    )

    expected_out_of_sample_dates = int(
        master_panel.loc[
            master_panel[
                "Date"
            ]
            >= first_test_start,
            "Date",
        ].nunique()
    )

    if (
        total_test_observations
        != expected_out_of_sample_rows
    ):
        raise AssertionError(
            "The quarterly outer windows do not cover "
            "the complete out-of-sample observation set."
        )

    if (
        total_test_dates
        != expected_out_of_sample_dates
    ):
        raise AssertionError(
            "The quarterly outer windows do not cover "
            "the complete out-of-sample date set."
        )

    # --------------------------------------------------------
    # 7.6 Save core panel and validation schedules
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

    # --------------------------------------------------------
    # 7.7 Save feature configuration
    # --------------------------------------------------------

    feature_configuration = pd.DataFrame(
        [
            {
                "Feature_Set": "Primary_Full_9",
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

    feature_configuration.to_csv(
        OUTPUT_FEATURE_CONFIGURATION,
        index=False,
    )

    # --------------------------------------------------------
    # 7.8 Generate walk-forward figure
    # --------------------------------------------------------

    create_walk_forward_figure(
        outer_windows
    )

    # --------------------------------------------------------
    # 7.9 Save setup summary
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
                "Primary predictors",
                "Reduced robustness predictors",
                "Initial outer test quarter",
                "Outer walk-forward windows",
                "Inner validation folds per outer window",
                "Validation dates per inner fold",
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
                len(
                    master_panel
                ),
                len(
                    model_dates
                ),
                master_panel[
                    "Date"
                ].min().date(),
                master_panel[
                    "Date"
                ].max().date(),
                len(
                    PRIMARY_FEATURES
                ),
                len(
                    REDUCED_ROBUSTNESS_FEATURES
                ),
                str(
                    INITIAL_TEST_QUARTER
                ),
                len(
                    outer_windows
                ),
                INNER_VALIDATION_FOLDS,
                INNER_VALIDATION_DATES_PER_FOLD,
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

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    # --------------------------------------------------------
    # 7.10 Terminal output
    # --------------------------------------------------------

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
        f"{master_panel['Date'].min().date()} to "
        f"{master_panel['Date'].max().date()}"
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

    for row in outer_windows.itertuples(
        index=False
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
        "- The master panel contains one observation "
        "per Date-Ticker pair."
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
        "- Every XGBoost inner fold uses a later "
        "chronological validation block."
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

    for output_file in [
        OUTPUT_MASTER_PANEL,
        OUTPUT_OUTER_WINDOWS,
        OUTPUT_INNER_WINDOWS,
    ]:
        print(
            f"- {relative_path(output_file)}"
        )

    print(
        "\nAudit and figure outputs:"
    )

    for output_file in [
        OUTPUT_SUMMARY,
        OUTPUT_DATE_INDEX,
        OUTPUT_FEATURE_CONFIGURATION,
        OUTPUT_FIGURE,
    ]:
        print(
            f"- {relative_path(output_file)}"
        )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()