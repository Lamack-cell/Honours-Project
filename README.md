# JSE Cross-Sectional Ranking Study

This project compares Pooled Cross-Sectional OLS and XGBoost for
one-day-ahead cross-sectional equity ranking on a dynamically constructed
liquid Johannesburg Stock Exchange universe.

## Project structure

- scripts/ â€” numbered research scripts
- data/raw/ â€” frozen source data
- data/interim/ â€” active-universe and target datasets
- data/processed/ â€” model-ready feature panel and validation schedules
- esults/ â€” OLS, XGBoost and statistical-comparison outputs
- outputs/audits/ â€” script-specific audit files
- outputs/figures/ â€” thesis-ready Chapter 4 and Chapter 5 figures
- outputs/logs/ â€” execution logs
- docs/ â€” thesis chapters and research documentation
- rchive/ â€” snapshots, superseded files and optional extensions

## Primary pipeline

1. Data extraction and correction
2. Dynamic liquidity-screened universe construction
3. Forward-return and percentile-target construction
4. Feature engineering and diagnostics
5. Expanding-window validation setup
6. Pooled Cross-Sectional OLS estimation
7. XGBoost estimation
8. Paired statistical model comparison
