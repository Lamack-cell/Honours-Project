"""
Run the complete updated statistical-research pipeline in dependency order.

Current scope
-------------
The primary study is a statistical comparison of Pooled OLS and XGBoost.
The former risk-free-rate Script 04 is intentionally excluded because no
portfolio, Sharpe-ratio or transaction-cost analysis remains in the core study.

Pipeline
--------
Core modelling scripts:
1. Script 01: broad data extraction and panel construction
2. Script 02: dynamic liquidity-proxy construction
3. Script 03: target engineering
4. Script 05: feature engineering
5. Script 06: walk-forward validation setup
6. Script 07: Pooled OLS estimation
7. Script 08: XGBoost estimation

Optional diagnostic scripts, enabled by default:
- Script 03B: target diagnostics and figure
- Script 05B: final modelling-panel diagnostics

Usage
-----
Place this file in the same project folder as the scripts, then run:

    python Run_Updated_Statistical_Pipeline.py

On Windows, this also usually works:

    py -3 Run_Updated_Statistical_Pipeline.py
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIRECTORY = Path(__file__).resolve().parent
LOG_DIRECTORY = PROJECT_DIRECTORY / "outputs" / "logs"

# Keep both diagnostics in the complete rerun.
RUN_OPTIONAL_DIAGNOSTICS = True

# The runner does not delete prior outputs. Each script overwrites its own
# named outputs, preserving the existing project structure and audit files.


@dataclass(frozen=True)
class PipelineStep:
    label: str
    accepted_filenames: tuple[str, ...]
    optional: bool = False


PIPELINE_STEPS = [
    PipelineStep(
        label="01 Data extraction and broad panel construction",
        accepted_filenames=(
            "01_Data_Extraction_and_Panel_Construction.py",
            "01_Data_Extraction_and_Panel_Construction(1).py",
        ),
    ),
    PipelineStep(
        label="02 Dynamic liquidity-proxy construction",
        accepted_filenames=(
            "02_Dynamic_Proxy_Construction.py",
            "02_Dynamic_Proxy_Construction(1).py",
        ),
    ),
    PipelineStep(
        label="03 Target engineering",
        accepted_filenames=(
            "03_Target_Engineering.py",
            "03_Target_Engineering(1).py",
        ),
    ),
    PipelineStep(
        label="03B Target diagnostics and visualisation",
        accepted_filenames=(
            "03B_Advanced_Target_Visual.py",
            "03B_Advanced_Target_Visual(1).py",
        ),
        optional=True,
    ),
    PipelineStep(
        label="05 Feature engineering and standardisation",
        accepted_filenames=(
            "05_Feature_Engineering.py",
            "05_Feature_Engineering(1).py",
        ),
    ),
    PipelineStep(
        label="05B Final modelling-panel diagnostics",
        accepted_filenames=(
            "05B_ML_Panel_Diagnostics.py",
            "05B_ML_Panel_Diagnostics(1).py",
        ),
        optional=True,
    ),
    PipelineStep(
        label="06 Walk-forward validation setup",
        accepted_filenames=(
            "06_Walk_Forward_Setup.py",
        ),
    ),
    PipelineStep(
        label="07 Pooled OLS expanding-window estimation",
        accepted_filenames=(
            "07_Pooled_OLS_Baseline.py",
        ),
    ),
    PipelineStep(
        label="08 XGBoost nested expanding-window estimation",
        accepted_filenames=(
            "08_XGBoost_Estimation.py",
        ),
    ),
]


# ============================================================
# 2. HELPERS
# ============================================================


def resolve_script(step: PipelineStep) -> Path:
    """Return the first existing accepted filename for a pipeline step."""

    for filename in step.accepted_filenames:
        candidate = PROJECT_DIRECTORY / filename
        if candidate.exists():
            return candidate

    accepted = "\n  - ".join(step.accepted_filenames)
    raise FileNotFoundError(
        f"Missing script for: {step.label}\n"
        f"Accepted filename(s):\n  - {accepted}\n"
        f"Project directory: {PROJECT_DIRECTORY}"
    )


def run_script(script_path: Path, label: str, log_handle) -> float:
    """Run one Python script, stream output, log it and stop on failure."""

    started = time.perf_counter()
    command = [sys.executable, str(script_path)]

    header = (
        "\n"
        + "=" * 88
        + f"\nRUNNING: {label}\n"
        + f"FILE: {script_path.name}\n"
        + "=" * 88
        + "\n"
    )
    print(header, end="")
    log_handle.write(header)
    log_handle.flush()

    process = subprocess.Popen(
        command,
        cwd=PROJECT_DIRECTORY,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert process.stdout is not None

    for line in process.stdout:
        print(line, end="")
        log_handle.write(line)
        log_handle.flush()

    return_code = process.wait()
    elapsed = time.perf_counter() - started

    footer = (
        f"\nCompleted in {elapsed / 60:.2f} minutes "
        f"with exit code {return_code}.\n"
    )
    print(footer)
    log_handle.write(footer)
    log_handle.flush()

    if return_code != 0:
        raise subprocess.CalledProcessError(
            returncode=return_code,
            cmd=command,
        )

    return elapsed


# ============================================================
# 3. MAIN PIPELINE
# ============================================================


def main() -> None:
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIRECTORY / f"full_statistical_pipeline_{timestamp}.log"

    selected_steps = [
        step
        for step in PIPELINE_STEPS
        if RUN_OPTIONAL_DIAGNOSTICS or not step.optional
    ]

    resolved_steps: list[tuple[PipelineStep, Path]] = []
    for step in selected_steps:
        resolved_steps.append((step, resolve_script(step)))

    print("=" * 88)
    print("UPDATED JSE STATISTICAL PIPELINE")
    print("=" * 88)
    print(f"Project directory: {PROJECT_DIRECTORY}")
    print(f"Python executable: {sys.executable}")
    print(f"Optional diagnostics enabled: {RUN_OPTIONAL_DIAGNOSTICS}")
    print("Script 04 risk-free-rate processing: intentionally excluded")
    print(f"Pipeline steps to run: {len(resolved_steps)}")
    print(f"Log file: {log_file}")

    for number, (step, path) in enumerate(resolved_steps, start=1):
        print(f"{number:>2}. {step.label} -> {path.name}")

    overall_start = time.perf_counter()
    completed: list[tuple[str, float]] = []

    try:
        with log_file.open("w", encoding="utf-8") as log_handle:
            log_handle.write("UPDATED JSE STATISTICAL PIPELINE\n")
            log_handle.write(f"Started: {datetime.now().isoformat()}\n")
            log_handle.write(f"Project directory: {PROJECT_DIRECTORY}\n")
            log_handle.write(f"Python executable: {sys.executable}\n")
            log_handle.write(
                f"Optional diagnostics enabled: {RUN_OPTIONAL_DIAGNOSTICS}\n"
            )
            log_handle.write(
                "Script 04 risk-free-rate processing: intentionally excluded\n"
            )

            for step, path in resolved_steps:
                elapsed = run_script(
                    script_path=path,
                    label=step.label,
                    log_handle=log_handle,
                )
                completed.append((step.label, elapsed))

            total_elapsed = time.perf_counter() - overall_start

            completion_text = (
                "\n"
                + "=" * 88
                + "\nPIPELINE COMPLETED SUCCESSFULLY\n"
                + "=" * 88
                + f"\nCompleted steps: {len(completed)}\n"
                + f"Total runtime: {total_elapsed / 60:.2f} minutes\n"
                + f"Finished: {datetime.now().isoformat()}\n"
                + f"Log file: {log_file}\n"
            )

            print(completion_text)
            log_handle.write(completion_text)

            print("Step runtimes:")
            for label, elapsed in completed:
                print(f"- {label}: {elapsed / 60:.2f} minutes")

    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        total_elapsed = time.perf_counter() - overall_start
        print("\n" + "!" * 88)
        print("PIPELINE STOPPED")
        print("!" * 88)
        print(str(exc))
        print(f"Completed steps before failure: {len(completed)}")
        print(f"Elapsed time: {total_elapsed / 60:.2f} minutes")
        print(f"Review the log file: {log_file}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()