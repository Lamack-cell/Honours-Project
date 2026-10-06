# 00_Setup_Professional_Project_Structure.ps1
# ------------------------------------------------------------
# Reorganises the Honours project into a professional research structure.
# Creates a timestamped snapshot before moving files.
#
# Run from the project root:
# powershell -ExecutionPolicy Bypass -File .\00_Setup_Professional_Project_Structure.ps1
# ------------------------------------------------------------

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$SnapshotRoot = ".\archive\pre_restructure_snapshot_$Timestamp"


# ============================================================
# 1. HELPERS
# ============================================================

function Ensure-Directory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    New-Item `
        -ItemType Directory `
        -Force `
        -Path $Path |
        Out-Null
}


function Move-IfExists {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,

        [Parameter(Mandatory = $true)]
        [string]$DestinationDirectory
    )

    if (Test-Path $Source) {
        Ensure-Directory $DestinationDirectory

        $FileName = Split-Path $Source -Leaf

        $Destination = Join-Path `
            $DestinationDirectory `
            $FileName

        Move-Item `
            $Source `
            $Destination `
            -Force

        Write-Host "Moved: $Source -> $Destination"
    }
}


function Move-MatchingFiles {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourceDirectory,

        [Parameter(Mandatory = $true)]
        [string]$Pattern,

        [Parameter(Mandatory = $true)]
        [string]$DestinationDirectory
    )

    if (-not (Test-Path $SourceDirectory)) {
        return
    }

    Get-ChildItem `
        -Path $SourceDirectory `
        -File `
        -Filter $Pattern |
        ForEach-Object {
            Move-IfExists `
                $_.FullName `
                $DestinationDirectory
        }
}


# ============================================================
# 2. CREATE SAFE PRE-RESTRUCTURE SNAPSHOT
# ============================================================
# The archive folder itself is deliberately excluded because the
# snapshot is stored inside archive. Copying archive into itself
# would create an infinite recursive copy.

Ensure-Directory ".\archive"
Ensure-Directory $SnapshotRoot

Write-Host ""
Write-Host "Creating pre-restructure snapshot..."

$SnapshotDirectories = @(
    "outputs",
    "scripts",
    "data",
    "results",
    "docs"
)

foreach ($Directory in $SnapshotDirectories) {
    $SourcePath = ".\$Directory"

    if (Test-Path $SourcePath) {
        Copy-Item `
            $SourcePath `
            "$SnapshotRoot\$Directory" `
            -Recurse `
            -Force

        Write-Host "Snapshot copied: $Directory"
    }
}

Get-ChildItem `
    -Path "." `
    -File |
    Where-Object {
        $_.Extension -in @(
            ".py",
            ".ps1",
            ".csv",
            ".xlsx",
            ".png",
            ".md",
            ".txt",
            ".docx",
            ".pdf"
        )
    } |
    ForEach-Object {
        Copy-Item `
            $_.FullName `
            "$SnapshotRoot\$($_.Name)" `
            -Force
    }

Write-Host "Snapshot created at: $SnapshotRoot"


# ============================================================
# 3. CREATE PROFESSIONAL PROJECT STRUCTURE
# ============================================================

$Directories = @(
    ".\scripts",

    ".\data\raw",
    ".\data\interim\02_universe",
    ".\data\interim\03_target",
    ".\data\processed\05_features",
    ".\data\processed\06_validation",

    ".\results\07_ols",
    ".\results\08_xgboost",
    ".\results\09_comparison",

    ".\outputs\audits\01_data",
    ".\outputs\audits\02_universe",
    ".\outputs\audits\03_target",
    ".\outputs\audits\05_features",
    ".\outputs\audits\06_validation",
    ".\outputs\audits\07_ols",
    ".\outputs\audits\08_xgboost",
    ".\outputs\audits\09_comparison",

    ".\outputs\figures\chapter_4",
    ".\outputs\figures\chapter_5",
    ".\outputs\logs",

    ".\docs",

    ".\archive\economic_extension",
    ".\archive\legacy_visuals",
    ".\archive\legacy_pipeline",
    ".\archive\duplicates",
    ".\archive\unclassified_outputs"
)

foreach ($Directory in $Directories) {
    Ensure-Directory $Directory
}


# ============================================================
# 4. ARCHIVE ECONOMIC / RISK-FREE EXTENSION
# ============================================================
# These files are no longer part of the focused statistical study.

$EconomicFiles = @(
    "04_Macroeconomic_EDA.py",
    "91-TB_data.xlsx",
    "JSE_91Day_TB_Weekly_Clean.csv",
    "JSE_Risk_Free_Daily.csv"
)

foreach ($File in $EconomicFiles) {
    Move-IfExists `
        ".\$File" `
        ".\archive\economic_extension"
}

Move-MatchingFiles `
    ".\outputs\audits" `
    "JSE_Risk_Free_*.csv" `
    ".\archive\economic_extension"

Move-MatchingFiles `
    ".\outputs\figures" `
    "figure_4_3_risk_free_rate*.png" `
    ".\archive\economic_extension"


# ============================================================
# 5. MOVE ACTIVE RESEARCH SCRIPTS
# ============================================================

$ActiveScripts = @(
    "01_Data_Extraction_and_Panel_Construction.py",
    "01B_Price_Scale_Audit.py",
    "01C_Apply_Price_Scale_Corrections.py",
    "02_Dynamic_Proxy_Construction.py",
    "03_Target_Engineering.py",
    "03B_Advanced_Target_Visual.py",
    "05_Feature_Engineering.py",
    "05B_ML_Panel_Diagnostics.py",
    "06_Walk_Forward_Setup.py",
    "07_Pooled_OLS_Baseline.py",
    "08_XGBoost_Estimation.py"
)

foreach ($File in $ActiveScripts) {
    Move-IfExists `
        ".\$File" `
        ".\scripts"
}


# ============================================================
# 6. RAW DATA
# ============================================================

Move-IfExists `
    ".\JSE_Broad_Master_Panel.csv" `
    ".\data\raw"


# ============================================================
# 7. SCRIPT 01 AUDIT OUTPUTS
# ============================================================

$RootScript01Audits = @(
    "JSE_Extraction_Ticker_Summary.csv",
    "JSE_Failed_Tickers.csv",
    "JSE_Data_Quality_Issues.csv"
)

foreach ($File in $RootScript01Audits) {
    Move-IfExists `
        ".\$File" `
        ".\outputs\audits\01_data"
}

$Script01AuditPatterns = @(
    "JSE_Extraction_Ticker_Summary*.csv",
    "JSE_Failed_Tickers*.csv",
    "JSE_Data_Quality_Issues*.csv",
    "JSE_Price_Scale_Candidates*.csv",
    "JSE_Price_Scale_Corrections_Applied*.csv"
)

foreach ($Pattern in $Script01AuditPatterns) {
    Move-MatchingFiles `
        ".\outputs\audits" `
        $Pattern `
        ".\outputs\audits\01_data"
}


# ============================================================
# 8. SCRIPT 02 DATA OUTPUTS
# ============================================================

$Script02Data = @(
    "JSE_Broad_Trading_Panel.csv",
    "JSE_Active_Universe_Membership.csv",
    "JSE_Active_Universe_Panel.csv"
)

foreach ($File in $Script02Data) {
    Move-IfExists `
        ".\$File" `
        ".\data\interim\02_universe"
}


# ============================================================
# 9. SCRIPT 02 AUDIT OUTPUTS
# ============================================================

$RootScript02Audits = @(
    "JSE_Active_Universe_Daily_Counts.csv",
    "JSE_Quarterly_Liquidity_Rankings.csv",
    "JSE_Quarterly_Universe_Turnover.csv",
    "JSE_Market_Date_Audit.csv",
    "JSE_Removed_Non_Trading_Dates.csv",
    "JSE_Ticker_Trading_Coverage_Audit.csv"
)

foreach ($File in $RootScript02Audits) {
    Move-IfExists `
        ".\$File" `
        ".\outputs\audits\02_universe"
}

$Script02AuditPatterns = @(
    "JSE_Active_Universe_Daily_Counts*.csv",
    "JSE_Market_Date_Audit*.csv",
    "JSE_Removed_Non_Trading_Dates*.csv",
    "JSE_Quarterly_Liquidity_Rankings*.csv",
    "JSE_Quarterly_Universe_Turnover*.csv",
    "JSE_Ticker_Trading_Coverage_Audit*.csv"
)

foreach ($Pattern in $Script02AuditPatterns) {
    Move-MatchingFiles `
        ".\outputs\audits" `
        $Pattern `
        ".\outputs\audits\02_universe"
}


# ============================================================
# 10. SCRIPT 03 DATA OUTPUTS
# ============================================================

$Script03Data = @(
    "JSE_Broad_Trading_Panel_With_Returns.csv",
    "JSE_Active_Universe_Panel_With_Returns.csv",
    "JSE_Panel_Target_Engineered.csv"
)

foreach ($File in $Script03Data) {
    Move-IfExists `
        ".\$File" `
        ".\data\interim\03_target"
}


# ============================================================
# 11. SCRIPT 03 AUDIT OUTPUTS
# ============================================================

$RootScript03Audits = @(
    "JSE_Target_Daily_Diagnostics.csv",
    "JSE_Target_Tie_Diagnostics.csv"
)

foreach ($File in $RootScript03Audits) {
    Move-IfExists `
        ".\$File" `
        ".\outputs\audits\03_target"
}

$Script03AuditPatterns = @(
    "JSE_Target_Daily_Diagnostics*.csv",
    "JSE_Target_Tie_Diagnostics*.csv",
    "JSE_Target_Distribution_Summary*.csv",
    "JSE_Target_Value_Frequencies*.csv"
)

foreach ($Pattern in $Script03AuditPatterns) {
    Move-MatchingFiles `
        ".\outputs\audits" `
        $Pattern `
        ".\outputs\audits\03_target"
}


# ============================================================
# 12. SCRIPT 05 PROCESSED FEATURE PANEL
# ============================================================

Move-IfExists `
    ".\JSE_Panel_Features_Engineered.csv" `
    ".\data\processed\05_features"


# ============================================================
# 13. SCRIPT 05 AND 05B AUDITS
# ============================================================

$Script05AuditPatterns = @(
    "JSE_Feature_*.csv",
    "JSE_ML_Panel_*.csv",
    "JSE_Model_Eligibility_Exclusions*.csv",
    "JSE_Annual_Model_Panel_Coverage*.csv",
    "JSE_High_Correlation_Pairs*.csv"
)

foreach ($Pattern in $Script05AuditPatterns) {
    Move-MatchingFiles `
        ".\outputs\audits" `
        $Pattern `
        ".\outputs\audits\05_features"
}


# ============================================================
# 14. SCRIPT 06 PROCESSED MODEL DATA
# ============================================================

$Script06Data = @(
    "JSE_Master_Model_Panel.csv",
    "JSE_Walk_Forward_Windows.csv",
    "JSE_Inner_Validation_Windows.csv"
)

foreach ($File in $Script06Data) {
    Move-IfExists `
        ".\$File" `
        ".\data\processed\06_validation"
}


# ============================================================
# 15. SCRIPT 06 AUDITS
# ============================================================

Move-IfExists `
    ".\JSE_Model_Feature_Configuration.csv" `
    ".\outputs\audits\06_validation"

$Script06AuditPatterns = @(
    "JSE_Walk_Forward_Setup_Summary*.csv",
    "JSE_Model_Date_Index*.csv",
    "JSE_Model_Feature_Configuration*.csv"
)

foreach ($Pattern in $Script06AuditPatterns) {
    Move-MatchingFiles `
        ".\outputs\audits" `
        $Pattern `
        ".\outputs\audits\06_validation"
}


# ============================================================
# 16. SCRIPT 07 OLS RESULTS
# ============================================================

$Script07Results = @(
    "JSE_OLS_Walk_Forward_Predictions.csv",
    "JSE_OLS_Daily_Performance.csv",
    "JSE_OLS_Window_Performance.csv",
    "JSE_OLS_Overall_Performance.csv"
)

foreach ($File in $Script07Results) {
    Move-IfExists `
        ".\$File" `
        ".\results\07_ols"
}

Move-MatchingFiles `
    ".\outputs\audits" `
    "JSE_OLS_*.csv" `
    ".\outputs\audits\07_ols"


# ============================================================
# 17. SCRIPT 08 XGBOOST RESULTS
# ============================================================

$Script08Results = @(
    "JSE_XGB_Walk_Forward_Predictions.csv",
    "JSE_XGB_Daily_Performance.csv",
    "JSE_XGB_Window_Performance.csv",
    "JSE_XGB_Overall_Performance.csv"
)

foreach ($File in $Script08Results) {
    Move-IfExists `
        ".\$File" `
        ".\results\08_xgboost"
}

Move-MatchingFiles `
    ".\outputs\audits" `
    "JSE_XGB_*.csv" `
    ".\outputs\audits\08_xgboost"


# ============================================================
# 18. THESIS FIGURES
# ============================================================

Move-MatchingFiles `
    ".\outputs\figures" `
    "figure_4_*.png" `
    ".\outputs\figures\chapter_4"

Move-MatchingFiles `
    ".\outputs\figures" `
    "figure_5_*.png" `
    ".\outputs\figures\chapter_5"


# ============================================================
# 19. ARCHIVE LEGACY FIGURES AND TOOLS
# ============================================================

$LegacyVisuals = @(
    "Figure_4.1_Pipeline_Horizontal.png",
    "figure_4.1_quarterly_universe_turnover.png",
    "figure_4.2_target_distribution_Advanced.png",
    "figure_4.3_macro_regime.png",
    "Horizontal Flowchart.py"
)

foreach ($File in $LegacyVisuals) {
    Move-IfExists `
        ".\$File" `
        ".\archive\legacy_visuals"
}

Move-IfExists `
    ".\JSE_Broad_Trading_Panel - Copy.csv" `
    ".\archive\duplicates"

Move-IfExists `
    ".\Run_Updated_Statistical_Pipeline.py" `
    ".\archive\legacy_pipeline"

Move-IfExists `
    ".\00_Organise_Project_Files.ps1" `
    ".\archive\legacy_pipeline"

if (Test-Path ".\archive_old_pipeline") {
    Move-Item `
        ".\archive_old_pipeline" `
        ".\archive\legacy_pipeline\archive_old_pipeline" `
        -Force

    Write-Host `
        "Moved: archive_old_pipeline -> archive\legacy_pipeline"
}


# ============================================================
# 20. MOVE DOCUMENTS
# ============================================================

Get-ChildItem `
    -Path "." `
    -File |
    Where-Object {
        $_.Extension -in @(
            ".docx",
            ".pdf",
            ".ppt",
            ".pptx"
        )
    } |
    ForEach-Object {
        Move-IfExists `
            $_.FullName `
            ".\docs"
    }


# ============================================================
# 21. ARCHIVE ANY UNCLASSIFIED ROOT JSE CSV FILES
# ============================================================
# All active known CSV outputs should already have been moved.
# Anything remaining is preserved rather than deleted.

Get-ChildItem `
    -Path "." `
    -File `
    -Filter "JSE_*.csv" |
    ForEach-Object {
        Move-IfExists `
            $_.FullName `
            ".\archive\unclassified_outputs"
    }


# ============================================================
# 22. CREATE README
# ============================================================

$Readme = @"
# JSE Cross-Sectional Ranking Study

This project compares Pooled Cross-Sectional OLS and XGBoost for
one-day-ahead cross-sectional equity ranking on a dynamically constructed
liquid Johannesburg Stock Exchange universe.

## Project structure

- `scripts/` — numbered research scripts
- `data/raw/` — frozen source data
- `data/interim/` — active-universe and target datasets
- `data/processed/` — model-ready feature panel and validation schedules
- `results/` — OLS, XGBoost and statistical-comparison outputs
- `outputs/audits/` — script-specific audit files
- `outputs/figures/` — thesis-ready Chapter 4 and Chapter 5 figures
- `outputs/logs/` — execution logs
- `docs/` — thesis chapters and research documentation
- `archive/` — snapshots, superseded files and optional extensions

## Primary pipeline

1. Data extraction and correction
2. Dynamic liquidity-screened universe construction
3. Forward-return and percentile-target construction
4. Feature engineering and diagnostics
5. Expanding-window validation setup
6. Pooled Cross-Sectional OLS estimation
7. XGBoost estimation
8. Paired statistical model comparison
"@

$Readme |
    Set-Content `
        -Path ".\README.md" `
        -Encoding UTF8


# ============================================================
# 23. CREATE BASIC GITIGNORE
# ============================================================

$GitIgnore = @"
venv/
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
Thumbs.db
*.tmp
*.log
"@

$GitIgnore |
    Set-Content `
        -Path ".\.gitignore" `
        -Encoding UTF8


# ============================================================
# 24. FINAL MESSAGE
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host "PROFESSIONAL PROJECT RESTRUCTURE COMPLETED"
Write-Host "============================================================"
Write-Host "Snapshot created at:"
Write-Host $SnapshotRoot
Write-Host ""
Write-Host "Corrected raw panel location:"
Write-Host ".\data\raw\JSE_Broad_Master_Panel.csv"
Write-Host ""
Write-Host "IMPORTANT:"
Write-Host "Do not rerun Scripts 02-08 yet."
Write-Host "Their input and output paths must first be updated."
Write-Host "============================================================"