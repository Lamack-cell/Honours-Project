# CHAPTER 5: EMPIRICAL RESULTS AND DISCUSSION

## 5.1 Introduction and Empirical Overview

This chapter presents the empirical results of the head-to-head statistical comparison between Pooled Ordinary Least Squares (OLS) and Extreme Gradient Boosting (XGBoost) for predicting daily cross-sectional stock-return rankings on the Johannesburg Stock Exchange (JSE). Following the econometric and experimental protocols established in Chapter 4, all evaluations are conducted on an expanding-window walk-forward basis across nine consecutive out-of-sample calendar quarters spanning 2 January 2024 to 18 March 2026 ($T = 554$ out-of-sample trading dates). 

Both models generate raw continuous predictions for the identical set of $N_{\text{total}} = 22,136$ stock-day observations from the JSE Top 40 liquid equity universe. All predictions are mapped against the strictly point-in-time cross-sectional percentile target $Y_{i,t} \in [0, 1]$. To ensure absolute internal validity and eliminate any risk of artificial performance inflation, both models operate under strict chronological validation with a target-horizon purge buffer, eliminating any look-ahead bias or label leakage.

The core research question investigated is whether XGBoost—a non-linear, tree-based ensemble capable of modeling high-order feature interactions—generates superior cross-sectional predictive accuracy compared to the parsimonious linear benchmark, Pooled OLS. The evaluation is conducted entirely through a **statistical lens**, focusing on rank correlation (Spearman Information Coefficient), risk-adjusted predictive consistency (Information Ratio), tail quartile hit rates, and heteroskedasticity- and autocorrelation-consistent (HAC) hypothesis testing.

### Summary of Central Findings

1. **Failure to Outperform the Linear Benchmark**: Over the full 554-day out-of-sample period, Extreme Gradient Boosting does **not** outperform Pooled OLS. Pooled OLS generates a mean daily Information Coefficient ($\overline{\text{IC}}$) of **+0.0199**, compared to **+0.0148** for XGBoost. 
2. **Statistical Significance**: A paired Newey-West HAC test on the daily paired difference series ($\Delta_t = \text{IC}_t^{\text{XGB}} - \text{IC}_t^{\text{OLS}}$) yields a test statistic of $z = -0.7441$ ($p = 0.7716$ for the one-sided test of XGBoost superiority; $p = 0.4568$ for the two-sided test of difference). The null hypothesis cannot be rejected at any conventional significance level.
3. **Risk-Adjusted Predictive Quality**: Pooled OLS delivers a higher unannualised Information Ratio ($\text{ICIR} = 0.0852$) than XGBoost ($\text{ICIR} = 0.0673$), alongside a higher proportion of positive-IC trading days (54.15% versus 53.61%).
4. **Tail Discrimination**: In the extreme quartiles—the primary domain of long-short equity selection—Pooled OLS achieves a superior mean top-quartile hit rate of 27.996% (versus 26.877% for XGBoost) and a bottom-quartile hit rate of 25.596% (versus 25.018% for XGBoost), with both models outperforming the 25.0% random baseline.
5. **Model Mechanics**: Interpretability diagnostics reveal that both models identify identical dominant asset-pricing signals—namely, short-term mean-reversion via the 14-day Relative Strength Index (`RSI_14`) and long-term trend persistence via the 200-day moving average ratio (`SMA_200_Ratio`). However, OLS captures these monotonic relationships with lower variance error than XGBoost's piecewise step functions.
6. **Robustness to Longer Holding Horizons & Expanded Grid**: Extending the target horizon to 21 trading days (~1 month) alongside an expanded 96-candidate hyperparameter search over $L_2$ regularisation ($\lambda \in \{1.0, 5.0, 10.0\}$) does not rescue the non-linear model. Under the 21-day specification, technical price signals decay rapidly, leading to a mean IC of **+0.0014** for OLS and **-0.0138** for XGBoost ($z = -0.4289, p = 0.6660$). This demonstrates that the empirical findings are structural rather than an artifact of daily horizon noise or regularisation constraints.

---

## 5.2 Primary Out-of-Sample Results (1-Day Horizon)

### 5.2.1 Full Sample Performance Comparison

Table 5.1 reports the complete comparative performance metrics for Pooled OLS and XGBoost over the 554-day primary evaluation period, reproducing the audited findings from the experimental pipeline.

#### Table 5.1: Head-to-Head Model Performance Summary (Primary 1-Day Setup)

| Performance Metric | Pooled OLS | XGBoost | Difference (XGB − OLS) | Preferred Direction | Descriptive Advantage |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Sample Coverage** | | | | | |
| Out-of-sample trading dates ($T$) | 554 | 554 | 0 | Count | Tie |
| Total stock-day predictions ($N_{\text{total}}$) | 22,136 | 22,136 | 0 | Count | Tie |
| Minimum daily active cross-section | 40 | 40 | 0 | Count | Tie |
| Maximum daily active cross-section | 40 | 40 | 0 | Count | Tie |
| **Rank Correlation Metrics** | | | | | |
| Mean daily Information Coefficient ($\overline{\text{IC}}$) | **+0.019910** | +0.014801 | −0.005109 | Higher | **Pooled OLS** |
| Median daily Information Coefficient | **+0.026923** | +0.018707 | −0.008216 | Higher | **Pooled OLS** |
| Daily IC standard deviation ($\sigma_{\text{IC}}$) | 0.233794 | **0.219780** | −0.014013 | Lower | **XGBoost** |
| Unannualised IC Information Ratio ($\text{ICIR}$) | **+0.085159** | +0.067345 | −0.017814 | Higher | **Pooled OLS** |
| Percentage of positive-IC trading days ($\text{IC}_{+} \%$) | **54.15%** | 53.61% | −0.54% | Higher | **Pooled OLS** |
| **Tail Quartile Hit Rates** | | | | | |
| Mean daily top-quartile hit rate | **27.996%** | 26.877% | −1.119% | Higher | **Pooled OLS** |
| Mean daily bottom-quartile hit rate | **25.596%** | 25.018% | −0.578% | Higher | **Pooled OLS** |
| Theoretical random quartile baseline | 25.000% | 25.000% | 0.000% | Neutral | Baseline |
| **Observation-Level Calibration Metrics** | | | | | |
| Observation-level Mean Absolute Error (MAE) | **0.256307** | 0.256407 | +0.000101 | Lower | **Pooled OLS** |
| Observation-level Root Mean Squared Error (RMSE) | **0.295940** | 0.296001 | +0.000061 | Lower | **Pooled OLS** |
| Daily constant prediction days | 0 | 0 | 0 | Zero | Tie (No Degeneracy) |
| Score boundary violations ($\hat{Y} < 0$ or $\hat{Y} > 1$) | 0 | 0 | 0 | Zero | Tie (No Violations) |

*Notes: All metrics are computed on the identical 22,136 out-of-sample predictions across 554 trading dates spanning 2024 Q1 through 2026 Q1. Target $Y_{i,t}$ is the cross-sectional percentile rank of 1-day forward return uniformly distributed on $[0, 1]$. Descriptive winners denote purely numerical superiority prior to inferential hypothesis testing.*

### 5.2.2 Information Coefficient Distribution and Stability

The primary metric of cross-sectional ranking accuracy is the Spearman Information Coefficient ($\text{IC}_t$). Over the 554 trading dates, Pooled OLS achieved an average daily IC of $+0.019910$, outperforming XGBoost's $+0.014801$ by a margin of $0.005109$ (or $51.1$ basis points of rank correlation per day). 

When examining the central tendency of the daily distributions, the median daily IC highlights an even wider disparity in favour of OLS:
- **Pooled OLS Median IC**: $+0.026923$
- **XGBoost Median IC**: $+0.018707$
- **Median Difference**: $-0.008216$

This right-skewed median indicates that on the typical trading day, Pooled OLS extracts approximately $44\%$ more ordinal ranking signal from the cross-section than XGBoost.

Where XGBoost displays a descriptive advantage is in the dispersion of its daily IC series. XGBoost recorded an IC standard deviation of $0.219780$, compared to $0.233794$ for Pooled OLS—a reduction of $0.014013$ ($6.0\%$). This lower variance is a mechanical consequence of tree regularisation: the ensemble's row subsampling (`subsample` $= 0.8$), column subsampling (`colsample_bytree` $= 0.8$), and shrinkage ($\eta \in \{0.02, 0.05\}$) compress individual predictions toward the central node prior ($0.50$). While this shrinkage dampens extreme swings in daily ranking errors, it concurrently compresses valid signal, resulting in a lower overall mean IC.

Consequently, when ranking skill is evaluated on a risk-adjusted basis via the IC Information Ratio ($\text{ICIR} = \overline{\text{IC}} / \sigma_{\text{IC}}$), Pooled OLS remains superior:
$$
\text{ICIR}_{\text{OLS}} = \frac{0.019910}{0.233794} = 0.085159
$$
$$
\text{ICIR}_{\text{XGB}} = \frac{0.014801}{0.219780} = 0.067345
$$

On an unannualised daily basis, OLS generates a $26.5\%$ higher Information Ratio than XGBoost.

Figure 5.4 displays the time-series evolution of daily IC for both models across the 554-day evaluation window alongside their respective empirical distribution densities.

```
[INSERT FIGURE 5.4 HERE]
Figure 5.4: Daily Information Coefficient Comparison (OLS vs XGBoost)
(File: outputs/figures/chapter_5/figure_5_4_daily_ic_comparison.png)
```

As evident in Figure 5.4, the daily IC series for both models track each other closely throughout the sample, exhibiting common macroeconomic factor exposures and regime synchronicity. Both distributions are centred slightly to the right of zero, but both display the substantial day-to-day volatility characteristic of high-frequency equity asset pricing.

### 5.2.3 Hit Rates and Tail Discrimination

In practical quantitative portfolio management—particularly for long-short equity market-neutral strategies—an investor does not trade the entire cross-section equally. Instead, capital is deployed predominantly in the extreme deciles or quartiles: buying the top-ranked securities and shorting (or underweighting) the bottom-ranked securities. Therefore, accuracy in the tails is of paramount economic significance.

As documented in Table 5.1:
- **Top-Quartile Hit Rate**: Pooled OLS correctly identified securities that realised returns in the top 25% of the cross-section on **27.996%** of instances, compared to **26.877%** for XGBoost. Both models exceeded the uninformative 25.0% baseline, but OLS generated an excess hit rate over baseline of $+2.996\%$, which is $59.6\%$ larger than XGBoost's excess hit rate of $+1.877\%$.
- **Bottom-Quartile Hit Rate**: Pooled OLS correctly identified securities that realised returns in the bottom 25% of the cross-section on **25.596%** of instances, compared to **25.018%** for XGBoost. While OLS maintained a modest edge over the random baseline ($+0.596\%$), XGBoost's bottom-quartile hit rate was effectively indistinguishable from a random coin toss ($+0.018\%$).

This asymmetry demonstrates that the linear model was notably more effective at identifying both outperforming securities and underperforming securities, while the tree ensemble struggled to isolate negative return momentum in the bottom quartile.

### 5.2.4 Observation-Level Loss and Calibration Diagnostics

The observation-level loss metrics evaluate how well the continuous predicted scores $\hat{Y}_{i,t}$ approximate the uniform percentile target $Y_{i,t} \in [0, 1]$:
- **MAE**: Pooled OLS recorded an MAE of $0.256307$, compared to $0.256407$ for XGBoost.
- **RMSE**: Pooled OLS recorded an RMSE of $0.295940$, compared to $0.296001$ for XGBoost.

In both metrics, the numerical difference between the models is less than $0.0001$. Crucially, these empirical loss values correspond tightly to the theoretical mathematical properties of a uniform distribution:
- For a random variable $Y \sim \text{Uniform}(0, 1)$, predicting the constant mean $\hat{Y} = 0.5$ yields a theoretical MAE of:
  $$
  E[|Y - 0.5|] = \int_0^1 |y - 0.5|\,dy = 2 \int_{0.5}^1 (y - 0.5)\,dy = 0.2500
  $$
- The theoretical standard deviation is:
  $$
  \sigma_Y = \sqrt{\frac{1}{12}} \approx 0.288675
  $$
The empirical RMSE of approximately $0.2959$ reflects the sum of this irreducible uniform dispersion and the residual variance of the predictors. Furthermore, the audit verified that neither model generated degenerate scores: zero days produced constant score vectors, and zero predictions violated the bounded range $[0, 1]$.

---

## 5.3 Formal Statistical Hypothesis Testing

### 5.3.1 Paired Newey-West HAC Test Results

To determine whether the descriptive outperformance of Pooled OLS over XGBoost represents a statistically significant difference—or conversely, whether XGBoost can be shown to outperform OLS—we conduct formal hypothesis testing using the paired Newey-West HAC regression specified in Section 4.9.

Let $\Delta_t = \text{IC}_t^{\text{XGB}} - \text{IC}_t^{\text{OLS}}$ denote the paired daily difference in Information Coefficient on trading date $t$. We estimate the regression model:
$$
\Delta_t = \mu + \varepsilon_t, \quad t = 1, \dots, 554
$$
where standard errors are estimated using the Newey-West (1987) Bartlett kernel with an automatic lag selection parameter of $q = \lfloor 4(554/100)^{2/9} \rfloor = 5$ lags.

Table 5.2 presents the complete estimation and hypothesis testing results.

#### Table 5.2: Paired Newey-West HAC Test of Information Coefficient Differences

| Parameter / Test Metric | Estimated Value / Specification | Econometric Interpretation |
|:---|:---:|:---|
| **Sample Size ($T$)** | 554 daily paired observations | Out-of-sample trading dates (2024 Q1 – 2026 Q1) |
| **Newey-West Truncation Lag ($q$)** | 5 trading days | Bartlett kernel accounts for 1 week of serial correlation |
| **Mean Daily Difference ($\hat{\mu}$)** | **−0.005109** | On average, XGBoost IC is 0.51% lower than OLS IC |
| **Newey-West Standard Error ($\widehat{\text{SE}}_{\text{NW}}$)** | 0.006865 | HAC-consistent standard error of the mean difference |
| **Normal Test Statistic ($z$)** | **−0.744147** | $z = \hat{\mu} / \widehat{\text{SE}}_{\text{NW}}$ |
| **One-Sided $p$-value ($H_1: \mu > 0$)** | **0.771606** | Test of hypothesis that XGBoost is superior to OLS |
| **Two-Sided $p$-value ($H_1: \mu \ne 0$)** | **0.456787** | Test of hypothesis that the two models perform differently |
| **95% Confidence Interval for $\mu$** | [**−0.018564**, **+0.008347**] | Interval includes zero; encompasses both positive & negative values |
| **Directional Decision ($\alpha = 0.05$)** | **Fail to Reject $H_0$** | No empirical evidence that XGBoost outperforms Pooled OLS |
| **Two-Sided Decision ($\alpha = 0.05$)** | **Fail to Reject $H_0$** | No statistically significant difference in predictive accuracy |

### 5.3.2 Statistical Interpretation

The statistical test yields two decisive conclusions:

1. **Directional Superiority ($H_1: \mu > 0$)**: The one-sided test directly evaluates the core hypothesis posited by machine learning proponents—namely, that decision tree ensembles capture non-linear market structures that linear models cannot, thereby generating superior returns rankings. The resulting $p$-value of **$0.7716$** indicates that we fail to reject the null hypothesis $H_0: \mu \le 0$ by an overwhelming margin. There is no empirical evidence to support the claim that XGBoost generates superior ranking performance on the JSE Top 40.
2. **Equivalence / Two-Sided Test ($H_1: \mu \ne 0$)**: The two-sided test evaluates whether either model is statistically distinguishable from the other. The two-sided $p$-value of **$0.4568$** confirms that the observed difference of $-0.0051$ is statistically indistinguishable from zero at all standard levels of significance ($\alpha = 0.10, 0.05, 0.01$). The 95% confidence interval spans $[-0.01856, +0.00835]$, confirming that random sampling variation comfortably encompasses the observed difference.

Econometrically, this null result is a vital and robust finding: it establishes that adding algorithmic complexity, gradient boosting ensembles, and tree-depth non-linearities does not translate into statistically measurable gains in out-of-sample cross-sectional return predictability on South Africa's premier equity index.

---

## 5.4 Sub-Period Dynamics and Quarterly Stability

### 5.4.1 Walk-Forward Outer Window Analysis

To determine whether model performance was uniform over time or driven by specific macroeconomic regimes, Table 5.3 decomposes the out-of-sample results across the nine individual expanding walk-forward windows.

#### Table 5.3: Out-of-Sample Performance by Walk-Forward Quarter

| Window | Test Quarter | Dates | OLS Mean IC | XGB Mean IC | IC Difference (XGB − OLS) | OLS ICIR | XGB ICIR | OLS Hit Rate (%) | XGB Hit Rate (%) | Quarterly Winner |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **OW_01** | 2024 Q1 | 62 | +0.013562 | +0.018318 | +0.004755 | 0.0618 | 0.0755 | 50.00% | **58.06%** | XGBoost |
| **OW_02** | 2024 Q2 | 61 | +0.042077 | +0.049129 | +0.007052 | 0.1849 | 0.1702 | **57.38%** | 55.74% | XGBoost |
| **OW_03** | 2024 Q3 | 64 | −0.019835 | −0.015929 | +0.003907 | −0.1077 | −0.0719 | 45.31% | **46.88%** | XGBoost |
| **OW_04** | 2024 Q4 | 63 | **+0.064639** | +0.062696 | −0.001942 | 0.3169 | **0.3275** | **63.49%** | 58.73% | Pooled OLS |
| **OW_05** | 2025 Q1 | 62 | −0.071921 | −0.068462 | +0.003460 | −0.2837 | −0.3053 | **46.77%** | 43.55% | XGBoost |
| **OW_06** | 2025 Q2 | 60 | +0.064279 | **+0.069573** | +0.005293 | 0.2596 | **0.3516** | 61.67% | **63.33%** | XGBoost |
| **OW_07** | 2025 Q3 | 65 | **+0.046995** | +0.026428 | −0.020567 | **0.2094** | 0.1424 | **60.00%** | 56.92% | Pooled OLS |
| **OW_08** | 2025 Q4 | 63 | **−0.001452** | −0.027143 | −0.025690 | **−0.0061** | −0.1331 | **47.62%** | 42.86% | Pooled OLS |
| **OW_09** | 2026 Q1 | 54 | **+0.045534** | +0.022209 | −0.023325 | **0.1615** | 0.1282 | 55.56% | **57.41%** | Pooled OLS |
| **Total** | **Full Sample** | **554** | **+0.019910** | **+0.014801** | **−0.005109** | **0.0852** | **0.0673** | **54.15%** | **53.61%** | **Pooled OLS** |

*Notes: Quarterly winners are determined by mean daily IC. Positive Hit Rate denotes the percentage of days where daily $\text{IC} > 0$. Window OW_09 terminates on 18 March 2026.*

```
[INSERT FIGURE 5.5 HERE]
Figure 5.5: Quarterly Mean Information Coefficient Comparison
(File: outputs/figures/chapter_5/figure_5_5_quarterly_mean_ic_comparison.png)
```

### 5.4.2 Regime Dynamics and Macroeconomic Shocks

An examination of Table 5.3 and Figure 5.5 uncovers three distinct empirical regimes:

1. **The 2024 Trending and Consolidation Phase (OW_01 to OW_04)**: 
   During 2024, both models posted strong positive performance, particularly in Q2 and Q4, where mean daily IC reached $+0.042$ to $+0.065$. In this phase, XGBoost recorded narrow descriptive victories in three quarters (OW_01, OW_02, OW_03) by modest margins between $+0.0039$ and $+0.0071$. In Q4 2024, both models achieved their annual peak accuracy ($\text{IC} \approx +0.063$ to $+0.065$), accompanied by positive hit rates approaching $60\% - 63\%$. In these benign market conditions, the models captured strong cross-sectional dispersion in post-election South African equities.
2. **The 2025 Q1 Market Inversion Episode (OW_05)**:
   In the first quarter of 2025, both models suffered severe predictive drawdowns, recording negative average ICs: $-0.0719$ for OLS and $-0.0685$ for XGBoost, with positive-day hit rates dropping below $47\%$. This sharp inversion coincided with an abrupt macroeconomic factor rotation across JSE resources and financial counters, triggered by shifting global interest rate expectations and currency volatility. Because both models rely on trailing momentum and moving-average indicators, sudden cross-sectional mean-reversals caused high-ranked stocks to underperform and low-ranked stocks to surge. Notably, XGBoost was not immune to this factor crash: its non-linear architecture failed to anticipate the structural break, yielding an IC virtually identical to the linear model.
3. **The 2025 H2 – 2026 Q1 Divergence Phase (OW_07 to OW_09)**:
   The structural divergence that secured Pooled OLS's overall victory occurred during the latter half of the sample. Across windows OW_07, OW_08, and OW_09, Pooled OLS decisively outperformed XGBoost in every single quarter:
   - In 2025 Q3 (OW_07), OLS produced $\overline{\text{IC}} = +0.0470$ versus $+0.0264$ for XGBoost (OLS ahead by $+0.0206$).
   - In 2025 Q4 (OW_08), a challenging market quarter, OLS remained flat ($\overline{\text{IC}} = -0.0015$), whereas XGBoost degraded into substantial negative correlation ($\overline{\text{IC}} = -0.0271$, OLS ahead by $+0.0257$).
   - In 2026 Q1 (OW_09), OLS surged back to $\overline{\text{IC}} = +0.0455$, while XGBoost lagged at $+0.0222$ (OLS ahead by $+0.0233$).

**Asymmetry of Performance**: Across the five quarters that XGBoost won, its average margin of victory was only **$+0.00489$** in IC. Conversely, across the four quarters that Pooled OLS won, its average margin of victory was **$+0.01788$**—more than **3.6 times larger**. The linear model proved considerably more robust during shifting market conditions, whereas XGBoost suffered from overfitting to trailing market regimes.

---

## 5.5 Model Mechanics and Interpretability

### 5.5.1 OLS Coefficient Trajectory and Statistical Significance

To understand the economic drivers behind the linear model's performance, Table 5.4 summarizes the Pooled OLS coefficient estimates across the initial (OW_01) and final (OW_09) walk-forward training windows, alongside their two-way clustered standard errors and $z$-statistics.

#### Table 5.4: Pooled OLS Coefficient Estimates and Cluster-Robust Inference

| Feature Term | OW_01 Coefficient | OW_01 Cluster SE | OW_01 $z$-stat | OW_01 $p$-value | OW_09 Coefficient | OW_09 Cluster SE | OW_09 $z$-stat | OW_09 $p$-value |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Intercept (`const`)** | 0.499784 | 0.000469 | 1064.54 | <0.0001 | 0.499805 | 0.000388 | 1288.66 | <0.0001 |
| `Mom_1M` | +0.004131 | 0.003679 | +1.12 | 0.2615 | +0.000578 | 0.003310 | +0.17 | 0.8614 |
| `Mom_3M` | −0.000739 | 0.003681 | −0.20 | 0.8409 | −0.002879 | 0.003422 | −0.84 | 0.4002 |
| `Mom_6M` | −0.008719 | 0.005118 | −1.70 | 0.0885 | −0.008451 | 0.004612 | −1.83 | 0.0670 |
| `Vol_21D` | −0.000087 | 0.001840 | −0.05 | 0.9621 | −0.001642 | 0.001798 | −0.91 | 0.3611 |
| `RSI_14` | **−0.015883** | 0.004076 | **−3.90** | **<0.0001** | **−0.013180** | 0.003714 | **−3.55** | **0.0004** |
| `SMA_50_Ratio` | +0.003606 | 0.005204 | +0.69 | 0.4883 | +0.006841 | 0.004689 | +1.46 | 0.1446 |
| `SMA_200_Ratio` | **+0.013263** | 0.006759 | **+1.96** | **0.0497** | **+0.014210** | 0.005882 | **+2.42** | **0.0157** |
| `Volume_Shock_21D` | +0.003321 | 0.003164 | +1.05 | 0.2938 | +0.001550 | 0.002984 | +0.52 | 0.6033 |
| `Value_Traded_Shock_63D`| +0.001036 | 0.003045 | +0.34 | 0.7336 | +0.002410 | 0.002812 | +0.86 | 0.3915 |

*Notes: Standard errors are two-way clustered by trading date and security ticker using Petersen (2009) sandwich covariance. In-sample observations: 71,056 in OW_01; 91,031 in OW_09.*

```
[INSERT FIGURE 5.2 HERE]
Figure 5.2: Pooled OLS Coefficient Stability Across Expanding Windows
(File: outputs/figures/chapter_5/figure_5_2_ols_coefficient_stability.png)
```

Two economic signals emerge as statistically significant and economically dominant:

1. **Short-Term Mean Reversion (`RSI_14`)**: 
   The 14-day Wilder RSI displays a strongly negative coefficient throughout all windows, hovering between $-0.013$ and $-0.016$ with two-way clustered $z$-statistics exceeding $-3.55$ ($p < 0.001$). This confirms a pronounced short-term reversal effect on the JSE: stocks that have experienced sharp upward price pressure over the trailing 14 trading days become overbought and systematically underperform their peers on the subsequent trading day.
2. **Long-Term Trend Persistence (`SMA_200_Ratio`)**: 
   The ratio of adjusted price to its 200-day moving average exhibits a statistically significant positive coefficient ($\hat{\beta} \approx +0.013$ to $+0.015$, $z \approx +1.96$ to $+2.42$, $p < 0.05$). This reflects the well-documented classical momentum phenomenon: securities trading above their secular trend lines continue to benefit from institutional capital flows and positive analyst revisions.
3. **Target Calibration**: 
   The intercept term is estimated at $\hat{\beta}_0 = 0.4998$ ($t > 1,000$), perfectly matching the theoretical mean of the uniform percentile rank target $E[Y] = 0.50$.

### 5.5.2 XGBoost Hyperparameter Selection and Tree Constraints

To prevent overfitting on the noisy daily return panel, XGBoost's hyperparameters were selected via the 3-fold inner walk-forward cross-validation loop. Table 5.5 reports the optimal configuration identified for each outer window.

#### Table 5.5: Selected XGBoost Hyperparameters Across Walk-Forward Windows

| Window | Test Quarter | Learning Rate ($\eta$) | Max Depth ($d$) | Min Child Weight ($w$) | Subsample ($s$) | Colsample ($c$) | Final Boosting Rounds ($\hat{M}$) | Inner Validation IC |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **OW_01** | 2024 Q1 | 0.02 | 2 | 15.0 | 0.8 | 1.0 | 73 | +0.0305 |
| **OW_02** | 2024 Q2 | 0.05 | 2 | 15.0 | 1.0 | 0.8 | 8 | +0.0194 |
| **OW_03** | 2024 Q3 | 0.05 | 2 | 5.0 | 0.8 | 0.8 | 30 | +0.0264 |
| **OW_04** | 2024 Q4 | 0.05 | 3 | 5.0 | 0.8 | 1.0 | 71 | +0.0258 |
| **OW_05** | 2025 Q1 | 0.05 | 3 | 5.0 | 0.8 | 1.0 | 21 | +0.0284 |
| **OW_06** | 2025 Q2 | 0.02 | 2 | 15.0 | 1.0 | 0.8 | 30 | +0.0169 |
| **OW_07** | 2025 Q3 | 0.05 | 2 | 15.0 | 0.8 | 1.0 | 22 | +0.0277 |
| **OW_08** | 2025 Q4 | 0.05 | 3 | 5.0 | 0.8 | 1.0 | 14 | +0.0332 |
| **OW_09** | 2026 Q1 | 0.05 | 2 | 5.0 | 0.8 | 1.0 | 20 | +0.0104 |

A critical empirical pattern emerges from Table 5.5:
- **Shallow Tree Depth**: In **6 out of 9 windows**, the validation loop selected `max_depth = 2`, and in the remaining 3 windows, it selected `max_depth = 3`. Depth 4 was never chosen. Depth-2 trees permit at most two-way feature interactions, indicating that the inner validation folds actively penalised complex interaction structures.
- **Strict Regularisation via Early Stopping**: The final number of boosting rounds ($\hat{M}$) ranged from 8 to 73, with a median of 22 rounds. Early stopping halted training rapidly, confirming that additional boosting rounds quickly led to in-sample overfitting.

### 5.5.3 Feature Importance Stability (Gain Importance)

```
[INSERT FIGURE 5.3 HERE]
Figure 5.3: XGBoost Feature Gain Importance Across Outer Windows
(File: outputs/figures/chapter_5/figure_5_3_xgb_feature_importance_stability.png)
```

Confirming the econometric findings of the OLS regression, `RSI_14` is the overwhelmingly dominant feature in the XGBoost ensemble, accounting for **$15\%$ to $33\%$** of total gain across all windows. This is followed by `Value_Traded_Shock_63D` ($12\% - 15\%$), `Mom_1M` ($10\% - 17\%$), and `SMA_200_Ratio` ($10\% - 13\%$). 

The fact that both models allocate their primary predictive weight to the exact same features demonstrates that the empirical disagreement is **not** caused by different feature discovery. Rather, it reflects **functional form efficiency**: OLS models the relationship between RSI, moving-average trends, and future returns using a smooth, continuous linear hyperplane. XGBoost approximates this same relationship using disjointed rectangular partitions (step functions). In small cross-sections ($N \approx 40$), the step-function approximation introduces substantial estimation noise and boundary variance, degrading out-of-sample rank correlation.

---

## 5.6 Robustness and Sensitivity Analysis: Holding Horizon and Regularisation Flexibility

### 5.6.1 Motivation for Alternative Setup

A common hypothesis in quantitative finance is that non-linear machine learning models require longer holding horizons to overcome market microstructure noise (e.g., bid-ask bounce, daily liquidity friction) and that conservative regularisation grids may artificially restrict tree ensembles. 

To address these potential concerns, we implemented a complete, independent alternative empirical pipeline:
1. **Holding Horizon**: The target variable was re-engineered as a **21-trading-day (~1 month) forward return percentile rank**, with the purge buffer expanded to 21 trading dates to prevent overlap leakage.
2. **Expanded Regularisation Search**: The XGBoost candidate grid was expanded from 32 to **96 candidate configurations**, explicitly adding $L_2$ leaf-weight regularisation as a searchable hyperparameter: $\lambda \in \{1.0, 5.0, 10.0\}$.
3. **Econometric Testing**: The Newey-West lag parameter was adjusted to $q = 21$ to match the 21-day overlapping return horizon.

### 5.6.2 Side-by-Side Setup Comparison

Table 5.6 compares the empirical performance of the primary setup (1-day horizon) against the alternative setup (21-day horizon).

#### Table 5.6: Side-by-Side Comparison: Primary (1-Day) vs Alternative (21-Day) Setup

| Experimental Dimension / Metric | Primary Setup (1-Day Horizon) | Alternative Setup (21-Day Horizon) | Impact of Horizon Extension |
|:---|:---:|:---:|:---:|
| **Experimental Design** | | | |
| Target return horizon | 1 trading day | 21 trading days (~1 month) | Extended horizon |
| Target-horizon purge buffer | 1 trading date | 21 trading dates | Expanded buffer |
| XGBoost regularisation search ($\lambda$) | Fixed ($\lambda = 10.0$) | Grid searched: $\{1.0, 5.0, 10.0\}$ | 3× expanded search space |
| XGBoost candidate configurations | 32 candidates | 96 candidates | 2,592 inner fits |
| Newey-West HAC lag ($q$) | 5 lags | 21 lags | Accounts for multi-week overlap |
| Evaluated trading dates ($T$) | 554 | 534 | 20 dates dropped at boundary |
| Total stock-day predictions | 22,136 | 21,336 | Paired evaluation |
| **Pooled OLS Performance** | | | |
| Mean daily IC ($\overline{\text{IC}}$) | **+0.019910** | **+0.001356** | −93.2% signal decay |
| Median daily IC | +0.026923 | +0.023598 | Modest decline |
| Daily IC standard deviation | 0.233794 | 0.237256 | Stable variance |
| IC Information Ratio ($\text{ICIR}$) | +0.0852 | +0.0057 | Substantial drop |
| Positive-IC trading days (%) | 54.15% | 53.75% | Marginally lower |
| Observation-level RMSE | 0.295940 | 0.296252 | Identical calibration |
| **XGBoost Performance** | | | |
| Mean daily IC ($\overline{\text{IC}}$) | **+0.014801** | **−0.013799** | Complete signal inversion |
| Median daily IC | +0.018707 | −0.037328 | Negative median |
| Daily IC standard deviation | 0.219780 | 0.223436 | Stable variance |
| IC Information Ratio ($\text{ICIR}$) | +0.0673 | −0.0618 | Inverted risk-adjusted return |
| Positive-IC trading days (%) | 53.61% | 43.45% | Severe drop below 50% |
| Observation-level RMSE | 0.296001 | 0.295810 | Identical calibration |
| **Paired Newey-West HAC Test** | | | |
| Mean IC difference (XGB − OLS) | −0.005109 | −0.015155 | Gap widens against XGBoost |
| Newey-West standard error | 0.006865 | 0.035333 | Increased HAC uncertainty |
| Test statistic ($z$) | −0.7441 | −0.4289 | Not statistically significant |
| One-sided $p$-value ($H_1: \text{XGB} > \text{OLS}$) | 0.7716 | 0.6660 | **Fail to Reject $H_0$** |
| Two-sided $p$-value ($H_1: \text{XGB} \ne \text{OLS}$) | 0.4568 | 0.6680 | **Fail to Reject $H_0$** |

### 5.6.3 Why Does the 21-Day Horizon Degrade Performance?

The empirical results in Table 5.6 reveal that extending the horizon to 21 trading days does **not** rescue XGBoost. In fact, XGBoost's mean IC collapses from $+0.0148$ to **$-0.0138$**, with positive-IC days dropping precipitously from $53.61\%$ to $43.45\%$. Pooled OLS also experiences signal decay (mean IC drops from $+0.0199$ to $+0.0014$), but remains positive and maintains a $53.75\%$ positive hit rate. 

Two economic and structural mechanisms explain this outcome:

1. **Short Half-Life of Pure Technical Predictors**:
   The feature set comprises technical price ratios, momentum indicators, volatility, and volume shocks. In modern asset pricing literature, technical indicators are recognized as having relatively short half-lives—often on the order of several days to two weeks. Without fundamental accounting data (e.g., Book-to-Market, Return on Equity, Earnings Yields, Operating Margin) that anchor long-term economic value, price-derived signals suffer rapid decay over a 21-day holding horizon.
2. **Utilisation of Expanded Regularisation Space**:
   In the alternative setup, the grid search actively selected $\lambda = 1.0$ in OW_07 and $\lambda = 5.0$ in OW_09, confirming that the expanded regularisation space was utilized. Yet despite this added flexibility, the tree ensemble was unable to generate positive out-of-sample rank correlation.

This robustness check conclusively refutes the critique that XGBoost's underperformance was an artifact of daily microstructure noise or an overly constrained regularisation grid.

---

## 5.7 Economic and Econometric Discussion: Why Simple Models Triumph on the JSE

The finding that Pooled OLS matches or exceeds the cross-sectional ranking accuracy of XGBoost on the JSE Top 40 stands in stark contrast to several high-profile studies conducted on large-cap US markets (e.g., Gu, Kelly, and Xiu, 2020), but aligns directly with recent foundational literature on the limitations of machine learning in asset pricing (e.g., Martin and Nagel, 2022; Arnott et al., 2019). 

We identify three fundamental economic and econometric reasons why simple models triumph in this setting:

### 1. The Cross-Sectional Sample Size Constraint ($N \approx 40$)
In US equity studies, the cross-section typically encompasses 1,000 to 3,000 securities (e.g., Russell 3000 or CRSP universe), providing tens of thousands of stock-month observations per cross-section. In such large panels, high-dimensional non-linear interactions can be estimated across rich feature combinations without overfitting. 

On the JSE, market capitalization is highly concentrated: the Top 40 index captures over 80% of total market equity, but restricts the cross-section to exactly $N_t \approx 40$ stocks per day. In a 40-stock universe, higher model complexity yields substantial **estimation variance**. A regression tree partitioning 40 stocks into leaves with `min_child_weight` $\ge 5$ can only create a handful of coarse bins, introducing high discretization error. Conversely, OLS estimates a smooth linear hyperplane that minimizes variance error across the entire panel.

### 2. The Monotonic Nature of Technical Signals
The primary predictors of cross-sectional returns on the JSE—RSI-14 (mean reversion) and SMA-200 (trend following)—are fundamentally **monotonic**. A stock with an RSI of 80 is more overbought than a stock with an RSI of 70, which is in turn more overbought than a stock with an RSI of 60. 

For monotonic relationships, Ordinary Least Squares is the Minimum Variance Unbiased Estimator (MVUE) under the Gauss-Markov theorem. Decision trees, by contrast, approximate monotonic relationships via discontinuous step functions. Unless an underlying relationship contains pronounced non-linearities (such as sign-reversals or complex threshold triggers), the tree ensemble will introduce unnecessary estimation variance without capturing additional economic truth.

### 3. The Absence of Fundamental Cross-Terms
Studies that document large gains from machine learning in asset pricing (Gu et al., 2020) emphasize that tree ensembles excel at identifying **macroeconomic conditioning**: for instance, value stocks outperforming growth stocks specifically when credit spreads widen, or small-cap momentum reversing during recessions. In the present study, the feature space is deliberately restricted to pure price-and-volume signals. Deprived of macroeconomic conditioning variables and firm-level balance sheet features, the tree ensemble's capacity to discover multi-variable interaction effects cannot be realized.

---

## 5.8 Chapter Summary

This chapter presented the empirical results of the head-to-head out-of-sample comparison between Pooled OLS and XGBoost across 554 trading dates on the JSE Top 40 universe:

- **Empirical Superiority of Pooled OLS**: Pooled OLS achieved a higher mean daily IC ($+0.0199$ vs $+0.0148$), higher median IC ($+0.0269$ vs $+0.0187$), superior Information Ratio ($0.0852$ vs $0.0673$), and superior top-quartile hit rate ($27.996\%$ vs $26.877\%$).
- **Statistical Inconclusiveness**: A paired Newey-West HAC test on the paired daily difference series revealed no statistically significant outperformance by XGBoost ($z = -0.7441, p = 0.7716$ for one-sided superiority; $p = 0.4568$ for two-sided difference).
- **Sub-Period Dynamics**: While XGBoost posted marginal victories in several early quarters, Pooled OLS proved substantially more resilient during regime shifts and market drawdowns, securing large victory margins in late 2025 and early 2026.
- **Model Interpretability**: Both models identified short-term RSI mean-reversion and 200-day moving average trend persistence as the dominant predictive drivers, but OLS translated these signals into superior ranking accuracy through low-variance linear estimation.
- **Robustness Extension**: Extending the forecasting horizon to 21 trading days resulted in severe technical signal decay, causing XGBoost's mean IC to invert to $-0.0138$, while OLS remained positive at $+0.0014$.

These findings challenge the prevailing presumption that complex machine learning algorithms inherently surpass traditional econometric baselines in equity return forecasting. Chapter 6 synthesizes these findings, delineates the theoretical and practical implications for institutional asset management on the JSE, discusses study limitations, and outlines directions for future research.
