# ============================================================
# HONOURS PROJECT FILE ORGANISATION
# ============================================================

$ProjectRoot = Get-Location

$AuditFolder     = Join-Path $ProjectRoot "outputs\audits"
$FigureFolder    = Join-Path $ProjectRoot "outputs\figures"
$DuplicateFolder = Join-Path $ProjectRoot "archive\duplicates"
$LegacyFolder    = Join-Path $ProjectRoot "archive\legacy"

$folders = @(
    $AuditFolder,
    $FigureFolder,
    $DuplicateFolder,
    $LegacyFolder
)

foreach ($folder in $folders) {
    New-Item -ItemType Directory -Path $folder -Force | Out-Null
}


# ============================================================
# 1. MOVE TRUE DUPLICATES
# ============================================================

$duplicateFiles = @(
    "JSE_Active_Universe_Panel_Copy.csv",
    "JSE_Data_Quality_Issues - Copy.csv",
    "JSE_Market_Date_Audit - Copy.csv",
    "JSE_Removed_Non_Trading_Dates - Copy.csv",
    "JSE_Ticker_Trading_Coverage_Audit - Copy.csv"
)

foreach ($file in $duplicateFiles) {
    $source = Join-Path $ProjectRoot $file

    if (Test-Path -LiteralPath $source) {
        Move-Item `
            -LiteralPath $source `
            -Destination $DuplicateFolder `
            -Force

        Write-Host "Archived duplicate: $file"
    }
}


# ============================================================
# 2. MOVE AUDIT AND DIAGNOSTIC OUTPUTS
# ============================================================

$auditFiles = @(
    # Script 01
    "JSE_Data_Quality_Issues.csv",
    "JSE_Extraction_Ticker_Summary.csv",
    "JSE_Failed_Tickers.csv",

    # Script 02
    "JSE_Market_Date_Audit.csv",
    "JSE_Removed_Non_Trading_Dates.csv",
    "JSE_Ticker_Trading_Coverage_Audit.csv",
    "JSE_Active_Universe_Daily_Counts.csv",
    "JSE_Quarterly_Liquidity_Rankings.csv",
    "JSE_Quarterly_Universe_Turnover.csv",

    # Script 03
    "JSE_Target_Daily_Diagnostics.csv",
    "JSE_Target_Tie_Diagnostics.csv"
)

foreach ($file in $auditFiles) {
    $source = Join-Path $ProjectRoot $file

    if (Test-Path -LiteralPath $source) {
        Move-Item `
            -LiteralPath $source `
            -Destination $AuditFolder `
            -Force

        Write-Host "Moved audit output: $file"
    }
}


# ============================================================
# 3. MOVE CURRENT ACCEPTED FIGURES
# ============================================================

$currentFigures = @(
    "figure_4_1_quarterly_universe_turnover.png",
    "figure_4_2_target_distribution.png"
)

foreach ($file in $currentFigures) {
    $source = Join-Path $ProjectRoot $file

    if (Test-Path -LiteralPath $source) {
        Move-Item `
            -LiteralPath $source `
            -Destination $FigureFolder `
            -Force

        Write-Host "Moved current figure: $file"
    }
}


# ============================================================
# 4. ARCHIVE LEGACY OUTPUTS
# ============================================================

$legacyFiles = @(
    "Figure_4.1_Pipeline_Horizontal.png",
    "Figure_4.2_target_distribution_Advanced.png",
    "Figure_4.3_macro_regime.png",
    "Figure_4.4_correlation_heatmap.png",
    "HorizontalFlowchart.py",

    # Old modelling outputs, where present
    "JSE_Panel_Features_Engineered.csv",
    "JSE_Master_Merged_Panel.csv",
    "JSE_Train_Set.csv",
    "JSE_Test_Set.csv",
    "JSE_Test_Set_OLS_Predictions.csv",
    "JSE_Test_Set_XGB_Predictions.csv"
)

foreach ($file in $legacyFiles) {
    $source = Join-Path $ProjectRoot $file

    if (Test-Path -LiteralPath $source) {
        Move-Item `
            -LiteralPath $source `
            -Destination $LegacyFolder `
            -Force

        Write-Host "Archived legacy file: $file"
    }
}


# ============================================================
# 5. SUMMARY
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host "PROJECT ORGANISATION COMPLETE"
Write-Host "============================================================"
Write-Host "Audit outputs:      outputs\audits"
Write-Host "Figures:            outputs\figures"
Write-Host "Duplicate files:    archive\duplicates"
Write-Host "Legacy files:       archive\legacy"
Write-Host ""
Write-Host "The essential pipeline files remain in the project root."
Write-Host "============================================================"