# JSE Cross-Sectional Ranking Study

### Pooled Cross-Sectional OLS vs XGBoost for One-Day-Ahead Equity Ranking on the Johannesburg Stock Exchange

This repository contains the research code and supporting materials for an Honours research study comparing **Pooled Cross-Sectional Ordinary Least Squares (OLS)** with **XGBoost** for **one-day-ahead cross-sectional equity ranking** on the Johannesburg Stock Exchange (JSE).

The study evaluates whether a nonlinear machine-learning model provides improved **out-of-sample cross-sectional ranking performance** relative to a pooled linear benchmark when both models are evaluated under the same chronological, walk-forward framework.

---

## Research Question

> **Does XGBoost provide superior out-of-sample one-day-ahead cross-sectional equity ranking performance compared with a pooled OLS benchmark on a dynamically constructed liquid JSE universe?**

The study focuses on the **relative ranking of securities**, rather than forecasting the absolute return of an individual security.

The analysis therefore evaluates predictive performance using statistical out-of-sample measures rather than constructing or evaluating an investment portfolio.

---

## Research Design

The empirical pipeline combines:

* A dynamically constructed, liquidity-screened JSE equity universe
* One-day-ahead forward returns
* Cross-sectional percentile ranking of the target
* Technical and trading-based predictors
* Cross-sectional feature standardisation
* Expanding-window walk-forward validation
* Pooled Cross-Sectional OLS
* XGBoost regression
* Paired out-of-sample statistical comparison

The models are trained and evaluated using the **same information sets, predictors, target definition and chronological validation structure**, allowing their ranking performance to be compared on a like-for-like basis.

---

## Primary Pipeline

```text
JSE Market Data
       │
       ▼
1. Data Extraction & Quality Correction
       │
       ▼
2. Dynamic Liquidity-Screened Universe
       │
       ▼
3. Forward Return & Percentile Target
       │
       ▼
4. Feature Engineering & Diagnostics
       │
       ▼
5. Expanding Walk-Forward Validation
       │
       ├───────────────┐
       ▼               ▼
6. Pooled OLS     7. XGBoost
       │               │
       └───────┬───────┘
               ▼
8. Paired Statistical Model Comparison
               │
               ▼
      Out-of-Sample Evaluation
```

---

## Models

### Pooled Cross-Sectional OLS

Pooled OLS provides the primary **linear benchmark**.

The model estimates the relationship between the cross-sectionally standardised predictors and the one-day-ahead percentile-ranked return target across the available securities and dates within each training window.

### XGBoost

XGBoost provides the primary **nonlinear machine-learning model**.

The implementation uses gradient-boosted decision trees to model potentially nonlinear relationships and interactions between the predictors and the cross-sectional ranking target.

Hyperparameters are selected using **chronological inner validation**, rather than random train-test splitting.

---

## Target Variable

The primary target is the **one-day-ahead raw return**, transformed into a **cross-sectional percentile rank** within the active universe.

This converts the prediction problem from absolute return forecasting into a relative ranking problem:

> Which securities are expected to perform relatively better or worse than their cross-sectional peers on the following trading day?

The ranking target is particularly appropriate for comparing models whose objective is to distinguish relative security performance.

---

## Predictor Set

The primary feature set consists of nine daily cross-sectionally standardised predictors:

| Feature                  | Description                    |
| ------------------------ | ------------------------------ |
| `Mom_1M`                 | One-month momentum             |
| `Mom_3M`                 | Three-month momentum           |
| `Mom_6M`                 | Six-month momentum             |
| `Vol_21D`                | 21-day volatility              |
| `RSI_14`                 | 14-day Relative Strength Index |
| `SMA_50_Ratio`           | Price relative to 50-day SMA   |
| `SMA_200_Ratio`          | Price relative to 200-day SMA  |
| `Volume_Shock_21D`       | 21-day volume shock            |
| `Value_Traded_Shock_63D` | 63-day value-traded shock      |

Within-window treatment and diagnostic procedures are applied according to the methodology documented in the thesis.

---

## Validation Framework

The study uses an **expanding-window walk-forward design** to preserve the temporal ordering of financial information.

The procedure is designed to avoid look-ahead bias and to ensure that model evaluation reflects a realistic out-of-sample forecasting environment.

The framework includes:

* Chronological training and test periods
* Expanding training windows
* Purging around out-of-sample boundaries where required
* Inner chronological validation for XGBoost hyperparameter selection
* No random train-test splitting
* Consistent test periods across models
* Aggregation of daily out-of-sample ranking performance

The final comparison is therefore based on observations that were not used to estimate the corresponding model or select its hyperparameters.

---

## Model Evaluation

The primary evaluation focuses on **out-of-sample cross-sectional ranking performance**.

Performance is assessed using measures including:

* Daily Information Coefficient (IC)
* Mean out-of-sample IC
* Ranking-error measures
* Daily paired model differences
* Window-level performance
* Statistical inference on model-performance differences

The final comparison evaluates whether any observed difference between OLS and XGBoost is statistically distinguishable from sampling variation.

---

## Scope

This project is deliberately focused on **statistical prediction and model comparison**.

It does **not** attempt to:

* Construct an investment portfolio
* Optimise portfolio weights
* Estimate transaction costs
* Attribute realised economic returns to a trading strategy
* Develop a live trading system
* Forecast individual stock prices in isolation

The central question is whether the two statistical models differ in their ability to **rank securities out of sample**.

---

## Project Structure

```text
Honors Project/
│
├── scripts/
│   └── Numbered research and modelling pipeline
│
├── data/
│   ├── raw/
│   │   └── Source and extracted market data
│   ├── interim/
│   │   └── Universe and target construction datasets
│   └── processed/
│       └── Model-ready feature and validation datasets
│
├── results/
│   ├── 07_ols/
│   ├── 08_xgboost/
│   └── 09_comparison/
│       └── Final model-comparison outputs
│
├── outputs/
│   ├── audits/
│   │   └── Data, feature and model diagnostics
│   ├── figures/
│   │   └── Thesis-ready figures
│   └── logs/
│       └── Pipeline execution logs
│
├── docs/
│   └── Thesis chapters and research documentation
│
├── archive/
│   └── Historical snapshots, superseded pipelines
│       and alternative validation experiments
│
└── README.md
```

---

## Pipeline Stages

| Stage | Purpose                                           |
| ----- | ------------------------------------------------- |
| `01`  | Data extraction, quality checks and correction    |
| `02`  | Dynamic liquidity-screened universe construction  |
| `03`  | Forward-return and percentile-target construction |
| `04`  | Risk-free-rate processing and alignment           |
| `05`  | Feature engineering and diagnostics               |
| `06`  | Walk-forward validation setup                     |
| `07`  | Pooled Cross-Sectional OLS estimation             |
| `08`  | XGBoost estimation                                |
| `09`  | Paired statistical model comparison               |

The numbered structure is intended to make the empirical workflow **reproducible and auditable**, with intermediate diagnostics generated throughout the pipeline.

---

## Reproducibility

The research pipeline is organised so that the major stages of the empirical analysis can be followed sequentially from data preparation through final model comparison.

Where possible, generated outputs are accompanied by audit files documenting:

* Data quality
* Universe membership
* Feature construction
* Missingness
* Correlations
* Multicollinearity
* Model eligibility
* Validation windows
* Hyperparameter selection
* Model estimation
* Out-of-sample performance

This allows the empirical results reported in the thesis to be traced back through the underlying research pipeline.

---

## Technologies

The project primarily uses:

* **Python** — data processing, feature engineering and machine learning
* **R** — statistical analysis and supporting diagnostics where applicable
* **SAS** — supporting statistical analysis where applicable
* **Git / GitHub** — version control and research reproducibility

Key Python libraries include tools from the scientific Python and machine-learning ecosystem, including **pandas, NumPy, scikit-learn and XGBoost**.

---

## Academic Context

This project forms part of an **Honours research study in Mathematical Statistics**.

The methodological foundation draws on established literature in:

* Empirical asset pricing
* Cross-sectional return prediction
* Machine learning in finance
* Gradient boosting
* Forecast evaluation
* Time-series and cross-sectional statistical inference

The repository is intended to provide access to the computational implementation supporting the accompanying research paper.

---

## Repository Status

**Status:** Active research project

**Primary comparison:** Pooled Cross-Sectional OLS vs XGBoost

**Prediction horizon:** One trading day ahead

**Market:** Johannesburg Stock Exchange (JSE)

**Task:** Cross-sectional equity ranking

**Validation:** Expanding-window walk-forward out-of-sample evaluation

---

## Citation

If you use or reference the code or methodology from this repository, please refer to the accompanying Honours research paper.

---

## Author

**Resimate Lamack Maluleke**

Honours Mathematical Statistics
Nelson Mandela University

---

### Repository

The complete research implementation and supporting computational materials are available in this repository.

**GitHub:** `https://github.com/Lamack-cell/Honours-Project`
