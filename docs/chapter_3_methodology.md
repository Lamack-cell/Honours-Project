# Chapter 3: Methodology

## 3.1 Introduction

This chapter set out the quantitative methodology that was used to evaluate whether a non-linear ensemble model, Extreme Gradient Boosting (XGBoost), produced superior out-of-sample cross-sectional asset rankings relative to a transparent linear benchmark, Pooled Cross-Sectional Ordinary Least Squares (OLS). The study was positioned within the cross-sectional return-prediction literature, where the objective was to estimate the future relative performance of securities within a common investment universe rather than to forecast the exact future price or return of an individual stock.

The central methodological comparison was therefore between two scoring functions. Following the comparative cross-sectional framework used by Fieberg et al. (2023), Pooled OLS was adopted as the linear benchmark. Fieberg et al. (2023) used linear regression as an established benchmark for cross-sectional stock-return forecasting and compared its performance with several machine-learning methods. XGBoost, introduced by Chen and Guestrin (2016), was selected as the non-linear alternative because it could model threshold effects, non-linear relationships and interactions without requiring these structures to be specified manually.

Pooled OLS estimated a single linear scoring rule across all eligible stock-day observations in each training window. XGBoost estimated a flexible non-linear scoring rule through an additive sequence of regression trees. Both models were trained on the same feature matrix, predicted the same target variable, used the same expanding-window test schedule and were evaluated on the same out-of-sample stock-date observations. This common experimental design ensured that differences in predictive performance primarily reflected the modelling architecture rather than differences in target construction, preprocessing, sample selection or evaluation periods.

An important clarification concerned the use of OLS as an asset-ranking model. OLS was not treated as a specialised ranking algorithm. Instead, it was used as a linear scoring model. For every out-of-sample trading date, the estimated OLS equation generated one continuous predicted score for each model-eligible security. These scores were subsequently sorted within the contemporaneous cross-section to form the predicted ordering of securities. The linear hyperplane estimated from the pooled training observations was therefore converted into a daily ranking mechanism.

XGBoost was applied using the same scoring-and-sorting framework. Although it represented a more flexible non-linear model, it was also estimated as a regression model on the percentile target. Its continuous predictions were ranked within each out-of-sample trading date in the same manner as the OLS predictions.

The methodology incorporated several safeguards that were required in financial forecasting research. These included data-quality and price-scale validation, dynamic point-in-time universe construction, feature and return calculation on complete security histories, training-window-only winsorisation, daily cross-sectional feature standardisation, expanding-window out-of-sample testing, purged training-test boundaries and nested chronological hyperparameter tuning.

These procedures were used to mitigate survivorship bias, look-ahead bias, target leakage and model-selection bias. The principal research question was whether XGBoost's additional flexibility produced statistically superior and more stable out-of-sample cross-sectional rankings relative to Pooled OLS under identical experimental conditions.

The study was statistical rather than portfolio-based. Predictive performance was evaluated using the daily Spearman Information Coefficient, the variability and consistency of the IC series, error-magnitude measures, quartile hit rates, quarterly stability and paired Newey-West inference.

## 3.2 Research Design and Forecasting Framework

### 3.2.1 Panel Structure

The empirical dataset was structured as a daily panel indexed by security $i$ and forecast-origin date $t$. Cross-sectional prediction studies commonly represented each stock-date observation by a vector of contemporaneously observable characteristics and a subsequently realised outcome (Gu, Kelly and Xiu, 2020; Fieberg et al., 2023).

Let $A_{t}$ be the active set of securities on date $t$, with active-universe size:

$$N_{t}=\vert{}A_{t}\vert{}$$

The liquidity-screened active universe contained 40 securities on every active date. Let $A_{t}^{M} \subseteq A_{t}$ denote the model-eligible subset for which the complete nine-feature vector was available, with:

$$N_{t}^{M}=\vert{}A_{t}^{M}\vert{}$$

The model-eligible cross-section varied between 38 and 40 securities because some newly available or recently selected securities lacked sufficient preceding history for the longest rolling indicators.

Historical daily Open, High, Low, Close, Adjusted Close and Volume observations were assembled for a researcher-curated collection of current and historical large- and mid-cap JSE securities. Seventy-three ticker symbols were requested, 62 returned usable histories and 11 returned no observations. The broad extracted panel contained 150,445 stock-day observations from 1 January 2016 to 19 March 2026.

Adjusted Close was used for the forward return and all return- or price-history-based predictors. Unadjusted Close was used only with reported Volume to form nominal value-traded measures. This distinction prevented ordinary corporate-action adjustments from being interpreted as return signals while preserving the contemporaneous quoted price in the trading-activity proxy. Open, High and Low were retained for quality-control checks but were not used in the final model specification.

Complete historical records of official FTSE/JSE Top 40 constituents were not available from the accessible public sources. The study therefore constructed a point-in-time, liquidity-screened proxy for the liquid large-cap JSE universe. This procedure mitigated the use of a static ex post constituent list, but it did not eliminate survivorship or data-availability bias because several requested ticker histories were unavailable (Brown et al., 1992; Hanauer and Kalsbach, 2023).

### 3.2.2 Prediction Horizon and Timing Convention

The forecasting horizon was one retained market trading date ahead. Features observed on forecast-origin date $t$ were used to predict the cross-sectional percentile outcome realised from $t$ to the next retained market date, denoted by $t+1$. The central relationship was:

$$\hat{Y}_{i,t+1\vert{}t}^{(m)}=g_{m}(X_{i,t})$$

where $X_{i,t}$ was the feature vector available for security $i$ on date $t$, $Y_{i,t+1}$ was the realised next-period percentile target, and $m$ denoted either OLS or XGBoost.

The notation $t+1|t$ indicated that the outcome was predicted for the next retained trading date using information available at $t$. For implementation, $Y_{i,t+1}$ was stored on the same dataset row as $X_{i,t}$, so the model-panel date recorded the forecast origin rather than the date on which the target became observable.

The design did not claim that predictions could have been calculated and executed at the same date-$t$ closing price. It tested the statistical association between information recorded for date $t$ and relative returns subsequently realised from $t$ to $t+1$.

### 3.2.3 Cross-Sectional Scoring and Ranking Logic

The objective was not to predict the exact future return of each stock. Instead, the models were required to predict which securities were likely to outperform or underperform their contemporaneous peers.

For each out-of-sample trading date, the fitted model generated one continuous score for every model-eligible security. These scores were then sorted independently within that date. Observations from other dates did not enter the daily sorting operation.

For model $m$, the predicted rank was defined as:

$$\overline{Rank}_{i,t+1\vert{}t}^{(m)}=rank_{ASC}(\hat{Y}_{i,t}^{(m)}\vert{}i\in A_{t}^{M})$$

where $A_{t}^{M}$ represented the model-eligible cross-section on date $t$.

The security with the lowest predicted score received the lowest predicted rank, while the security with the highest predicted score received the highest predicted rank. Average ranks were assigned where raw model predictions were tied.

Predicted ranks were also transformed into daily predicted percentiles:

$$\hat{P}_{i,t}^{(m)}=\frac{\overline{Rank}_{i,t+1\vert{}t}^{(m)}-1}{N_{t}^{M}-1}$$

where $N_{t}^{M}=|A_{t}^{M}|$ denoted the model-eligible cross-sectional size.

The same transformation was applied to both models. The raw predicted scores were retained for Spearman IC, MAE and RMSE calculations, while the predicted ranks and percentiles were stored for interpretation and audit purposes.

### 3.2.4 Dynamic Active-Universe Construction

The active universe was constructed dynamically using a point-in-time liquidity screen.

For security $i$ on trading date $t$, the nominal daily value-traded proxy was calculated as:

$$VT_{i,t}=Close_{i,t}\times Volume_{i,t}$$

A 63-trading-date rolling average was then calculated:

$$VT_{i,t}^{63}=\frac{1}{63}\sum_{s=t-62}^{t}VT_{i,s}$$

The 63-date window approximated one calendar quarter and reduced the influence of isolated daily trading-volume spikes.

At the final retained market date of each calendar quarter $q$, securities with a complete 63-date liquidity history were ranked in descending order according to $\overline{VT}_{i,t}^{63}$. The 40 securities with the largest values were selected for the following calendar quarter $q+1$.

The selection made at the end of quarter $q$ was never applied retrospectively to quarter $q$. It became effective only in the subsequent quarter. This ensured that the active universe for a given quarter was based only on information that had already been observed before that quarter began.

For example, liquidity recorded during the second quarter was not used to determine membership for the beginning of that same quarter. It was used only to determine membership for the third quarter.

The initial 2016 Q1 selection period did not contain a complete 63-date history for any security. The first usable selection was therefore obtained later, and the observed active-universe panel began on 1 July 2016.

The resulting active panel contained 97,160 stock-day observations across 2,429 trading dates from 1 July 2016 to 19 March 2026. Exactly 40 securities were present on every active-universe trading date.

Membership changed gradually. After the initial formation, the active universe experienced an average of approximately 1.45 security replacements per observed quarter. These changes reflected the point-in-time liquidity ranking rather than an official index-reconstitution record.

## 3.3 Data-Quality and Price-Scale Validation

Financial-data errors could have generated artificial returns, contaminated rolling indicators and distorted the daily ranking target. The broad panel was therefore subjected to a series of validation checks before active-universe construction, target engineering or feature construction.

The initial data audit examined:

- duplicate ticker-date observations;
- missing price and volume fields;
- non-positive prices;
- negative reported volumes;
- ticker-level historical coverage;
- market-wide non-trading dates;
- isolated zero-volume observations;
- internal consistency among Open, High, Low and Close; and
- extreme adjacent-date price changes.

No duplicate Date-Ticker observations were permitted. Price fields were required to be positive, while reported volume was required to be non-negative.

Internal OHLC checks assessed whether:

$$High_{i,t}\ge \max(Open_{i,t}, Close_{i,t})$$

and whether:

$$Low_{i,t}\le \min(Open_{i,t}, Close_{i,t})$$

The audit identified a substantial number of OHLC consistency flags. These flags were treated as warnings rather than automatic proof that the Adjusted Close observation was incorrect. Open, High and Low were not used in the final target or feature set. Consequently, flagged observations were not removed merely because one of these auxiliary fields failed an OHLC consistency check.

A separate price-scale procedure was applied to detect isolated mechanical changes by common multiplicative factors. Candidate observations were identified where Open, High, Low, Close and Adjusted Close changed by approximately the same factor, such as 100, while Volume remained unchanged and the adjacent price history supported a mechanical scale explanation.

Four isolated factor-of-100 price-scale errors were confirmed:

- DTC.JO on 10 January 2025;
- SLM.JO on 10 January 2025;
- VOD.JO on 10 January 2025; and
- SBK.JO on 25 April 2025.

For each confirmed observation, Open, High, Low, Close and Adjusted Close were multiplied by 100. Volume was left unchanged. The original values, corrected values and correction factor were preserved in an audit file.

The price-scale audit was rerun after the corrections had been applied. No remaining isolated mechanical scale candidates were detected.

Market-wide non-trading dates were identified before rolling liquidity calculations. A date was classified as non-trading where the broad panel contained no security with positive reported volume. Nineteen such dates were removed. This reduced the broad panel from 150,445 to 149,423 observations and from 2,571 to 2,552 retained market dates.

Isolated zero-volume security observations on otherwise active market dates were not treated as market closures. They remained in the broad panel because the market as a whole had traded on those dates.

A ticker-coverage audit subsequently confirmed that each security contained a complete sequence of retained broad-market dates between its first and final available observation. This check reduced the risk that rolling windows or next-date returns would inadvertently skip internal dates.

All confirmed price corrections and market-date exclusions were completed before the active universe, forward returns and rolling features were reconstructed.

## 3.4 Construction of the Target Variable

The target was designed to measure subsequent relative performance rather than the exact magnitude of the next-day return. This design was consistent with cross-sectional equity-prediction research in which stock-level characteristics were used to distinguish expected relative winners from expected relative losers (Gu, Kelly and Xiu, 2020; Fieberg et al., 2023). It was also related to ranking-based portfolio-selection research, where the relative ordering of securities represented the principal prediction problem (Song, Liu and Yang, 2017).

### 3.4.1 Forward Return Calculation

For each stock $i$ at time $t$, the one-day forward return is calculated using the adjusted closing price. Adjusted closing prices are preferred because they account for corporate actions such as dividends and stock splits.

The one-day forward return for stock $i$ at time $t$ is defined as:

$$R_{i,t+1}=\frac{P_{i,t+1}^{Adj}}{P_{i,t}^{Adj}}-1$$

where:

- $R_{i,t+1}$ is the realised forward return of stock $i$ from day $t$ to day $t+1$
- $P_{i,t}$ represents the adjusted closing price of stock $i$ at time $t$
- $P_{i,t+1}$ represents the adjusted closing price at the following trading day

So, the return measures how much the price changes from today to tomorrow, expressed as a percentage.

The forward return is used because machine learning models must predict future using past information and is used only to construct the target variable and to evaluate later performance. It is not included among the predictors available to the model at time $t$. Forward returns are calculated on the complete cleaned security histories before the active-universe filter is applied. This ordering is necessary because calculating returns after filtering could incorrectly connect two observations separated by an inactive period. For example, if a stock left the active universe for several months and later returned, the observations before and after the inactive interval must not be treated as consecutive daily prices.

A validation check confirms that $t+1$ is the next broad-market trading date, rather than simply the next row remaining after a security-specific filter has been applied.

The final date in each security history has no future price and is therefore excluded from the target dataset.

### 3.4.2 Cross-Sectional Ranking of Forward Returns

Once forward returns had been merged onto the point-in-time active universe, they were ranked within each trading date.

The realised daily rank was defined as:

$$Rank_{i,t+1}=rank_{ASC}(R_{i,t+1}\vert{}i\in A_{t})$$

The worst realised forward return received the lowest rank, while the best realised return received the highest rank.

Average ranks were assigned when two or more securities produced identical forward returns. This tie-handling rule avoided arbitrary ordering. Tied returns occurred primarily when several securities recorded exactly zero forward returns.

The target audit identified 84 dates containing at least one tied return, representing 3.46% of the 2,428 target dates. These ties did not invalidate the percentile construction because the daily mean rank remained centred.

### 3.4.3 Percentile Normalisation

The realised daily rank was converted into a percentile target bounded between zero and one:

$$Y_{i,t}=\frac{Rank_{i,t+1}-1}{N_{t}-1}$$

The target was indexed by feature date $t$, even though it was derived from the return realised from $t$ to $t+1$.

The interpretation was:

- $Y_{i,t}=0$ represented the lowest-ranked realised performer;
- $Y_{i,t}\approx0.5$ represented a middle-ranked performer; and
- $Y_{i,t}=1$ represented the highest-ranked realised performer.

The transformation made the target comparable across dates and maintained a common bounded response variable for OLS and XGBoost.

The active target cross-section contained exactly 40 securities on every date. In the absence of ties, the target therefore followed the discrete grid:

$$0, \frac{1}{39}, \frac{2}{39}, ..., \frac{38}{39}, 1$$

The target audit confirmed that:

- every target value lay in [0, 1];
- every daily target mean equalled 0.5;
- every daily cross-section contained a minimum target of 0;
- every daily cross-section contained a maximum target of 1; and
- the overall target mean equalled 0.5.

### 3.4.4 Rationale for a Ranking Target and Trade-Off

Daily stock returns are noisy, non-stationary, and affected by market-wide shocks. Directly forecasting the exact return of each stock would force the model to learn both the common market component and the idiosyncratic component. By converting returns into cross-sectional percentile ranks, the modelling task becomes more aligned with relative asset selection. The model is rewarded for correctly ordering stocks against their peers rather than for predicting the exact numerical size of tomorrow's return. Song, Liu and Yang (2017) similarly motivate rank-oriented portfolio selection by emphasising the usefulness of predicting relative stock performance for investment decisions.

However, the use of a ranking target also creates an important limitation. By normalising returns into daily percentile positions, the target discards information about the magnitude of return differences. A model is therefore rewarded for correctly ranking two stocks even when their realised returns differ only marginally. Conversely, a model that incorrectly orders two stocks with a very large economic return difference is treated in the same ordinal framework. Two securities separated by a very small return difference could receive different ranks, while the ordinal distance between ranks did not indicate the economic size of their realised return spread.

The study therefore evaluated statistical ranking skill only. It did not infer that a correct ranking necessarily implied a profitable or directly implementable trading strategy.

## 3.5 Feature Engineering, Standardisation and Leakage Prevention

### 3.5.1 Feature Engineering Framework

The feature-engineering pipeline transformed the cleaned broad panel into variables intended to capture momentum, volatility, technical strength and abnormal trading activity.

All rolling predictors were calculated on each security's complete cleaned historical sequence before the active-universe filter was applied. This prevented inactive-universe periods from being removed from the security's genuine market history.

For example, when a security left the active universe and later re-entered, its rolling indicators were still calculated from its continuous broad-market price history. The observations before and after the inactive membership period were not incorrectly treated as though the inactive period had never existed.

All features used information recorded at or before the end of feature date $t$. No price, volume, return or target value from date $t + 1$ entered the feature vector.

### 3.5.2 Primary Feature Set

The primary OLS and XGBoost models use the same nine predictors:

- one-month momentum;
- three-month momentum;
- six-month momentum;
- 21-day volatility;
- 14-day Relative Strength Index;
- 50-day moving-average ratio;
- 200-day moving-average ratio;
- 21-day volume shock; and
- 63-day value-traded shock.

**Momentum Measures**

One-month momentum is:

$$Mom\_1M_{i,t}=\frac{P_{i,t}^{Adj}}{P_{i,t-21}^{Adj}}-1$$

Three-month momentum is:

$$Mom\_3M_{i,t}=\frac{P_{i,t}^{Adj}}{P_{i,t-63}^{Adj}}-1$$

Six-month momentum is:

$$Mom\_6M_{i,t}=\frac{P_{i,t}^{Adj}}{P_{i,t-126}^{Adj}}-1$$

These features measure approximately one-, three- and six-month adjusted-price performance. The inclusion of momentum variables is motivated by the evidence of Jegadeesh and Titman (1993), who document persistence in the relative performance of recent winners and losers.

**Volatility**

Daily logarithmic returns are defined as:

$$r_{i,t}^{log}=\log\left(\frac{P_{i,t}^{Adj}}{P_{i,t-1}^{Adj}}\right)$$

The 21-day volatility feature is:

$$Vol\_21D_{i,t}=SD(r_{i,t-20}^{log}, ..., r_{i,t}^{log})$$

This variable captures differences in the recent magnitude and variability of stock returns.

**Relative Strength Index**

The 14-period Relative Strength Index was calculated using a Wilder-style exponentially smoothed gain-and-loss procedure.

Let:

$$\Delta P_{i,t}=P_{i,t}^{Adj}-P_{i,t-1}^{Adj}$$

Daily gains and losses were defined as:

$$Gain_{i,t}=\max(\Delta P_{i,t}, 0)$$

$$Loss_{i,t}=\max(-\Delta P_{i,t}, 0)$$

The smoothed average gain and loss used an exponential weighting parameter of:

$$\alpha=\frac{1}{14}$$

Relative strength was calculated as:

$$RS_{i,t}=\frac{AverageGain_{14,i,t}}{AverageLoss_{14,i,t}}$$

The RSI was then calculated as:

$$RSI\_14_{i,t} = 100 - \frac{100}{1 + RS_{i,t}}$$

Where the average loss was zero and the average gain was positive, RSI was assigned a value of 100. Where both average gain and average loss were zero, RSI was assigned a neutral value of 50.

The variable captured recent price strength and possible overbought or oversold conditions (Wilder, 1978). Because the implementation used exponentially weighted smoothing with $\alpha = 1/14$, it was described as Wilder-style RSI rather than as an exact replication of every possible textbook initialisation convention.

**Moving-Average Ratios**

The 50-day and 200-day simple moving averages were:

$$SMA_{50,i,t}=\frac{1}{50}\sum_{s=t-49}^{t}P_{i,s}^{Adj}$$

$$SMA_{200,i,t}=\frac{1}{200}\sum_{s=t-199}^{t}P_{i,s}^{Adj}$$

The 50-day moving-average ratio is:

$$SMA\_50\_Ratio_{i,t}=\frac{P_{i,t}^{Adj}}{SMA_{50,i,t}}$$

The 200-day moving-average ratio is:

$$SMA\_200\_Ratio_{i,t}=\frac{P_{i,t}^{Adj}}{SMA_{200,i,t}}$$

These ratios measure the stock's current adjusted price relative to its medium- and long-term historical trend. A value greater than one indicates that the current adjusted price is above the corresponding moving average. A value below one indicates that it is below the moving average.

**Volume Shock**

Reported volume was transformed using:

$$LV_{i,t}=\log(1+Volume_{i,t})$$

The 21-day volume shock was calculated as the difference between current log volume and the mean of the preceding 21 log-volume observations:

$$Volume\_Shock\_21D_{i,t} = LV_{i,t} - \frac{1}{21}\sum_{s=t-21}^{t-1}LV_{i,s}$$

The preceding mean excluded date $t$. The feature therefore measured whether current trading volume was unusually high or low relative to the immediately preceding history.

**Value-Traded Shock**

The nominal value-traded proxy was calculated as:

$$VT_{i,t}=Close_{i,t}\times Volume_{i,t}$$

Its logarithmic transformation was:

$$LVT_{i,t}=\log(1+VT_{i,t})$$

The 63-day value-traded shock was calculated as:

$$Value\_Traded\_Shock\_63D_{i,t}=LVT_{i,t}-\frac{1}{63}\sum_{s=t-63}^{t-1}LVT_{i,s}$$

The preceding 63-date mean excluded the current observation. The variable therefore measured unusually high or low nominal trading activity relative to the security's recent history. Unlike Volume Shock, Value-Traded Shock incorporated both reported share volume and the contemporaneous unadjusted quoted price.

### 3.5.3 Feature Warm-Up and Model Eligibility

Rolling features required sufficient preceding historical observations. The 200-day moving average imposed the longest warm-up requirement.

A stock-date observation entered the final modelling panel only when:

- the forward-return target was available;
- all nine raw predictors were available;
- all standardised predictors were finite; and
- the Date-Ticker combination was unique.

The active-universe panel contained 97,160 observations. Of these, 40 observations on the final active-universe date had no forward return and were therefore excluded from the target dataset, producing a target panel of 97,120 observations.

A total of 3,889 observations were subsequently excluded because at least one predictor lacked sufficient preceding history. The exclusions comprised:

- 3,040 observations from the initial feature warm-up period; and
- 849 later security-specific exclusions caused by insufficient historical coverage.

The final feature-engineered panel contained 93,231 observations across 2,352 trading dates from 19 October 2016 to 18 March 2026.

The daily model-eligible cross-section contained:

- 38 securities on 55 dates;
- 39 securities on 739 dates; and
- 40 securities on 1,558 dates.

The unweighted mean daily model cross-section was approximately 39.64 securities.

The warm-up rule was applied after rolling features had been calculated on complete histories. A security entering the active universe could therefore be included immediately where sufficient preceding broad-panel history already existed.

### 3.5.4 Cross-Sectional Standardisation

The models pool observations across stocks with different price levels, trading volumes, and volatility regimes. To make the features comparable across the daily cross-section, each continuous feature is standardised within each trading day because the study pools multiple stocks together, a raw price or volume metric for a massive mining conglomerate cannot be directly compared to a smaller retail stock. The use of daily cross-sectional transformations is consistent with cross-sectional equity prediction frameworks in which predictor variables are interpreted relative to the contemporaneous asset universe rather than solely in absolute firm-level units (Gu, Kelly and Xiu, 2020; Fieberg et al., 2023).

For each feature $k$, for each stock $i$, on each day $t$, the feature is standardised cross-sectionally (z-scored):

$$Z_{i,t,k}=\frac{X_{k,i,t}-\mu_{k,t}}{s_{k,t}}$$

Where:

- $Z_{i,t,k}$: The standardised value of feature $k$ for stock $i$ on day $t$.
- $X_{k,i,t}$: The raw calculated value of the feature.
- $\mu_{k,t}$: The cross-sectional mean of feature $k$ across all stocks on day $t$.
- $s_{k,t}$: The cross-sectional standard deviation of feature $k$ across all stocks on day $t$.

This converts each raw feature into a relative measure. For example, a standardised volume value of 2 means that the stock traded two cross-sectional standard deviations above the average stock on that day.

This standardisation does not use future returns and is therefore not a form of target leakage. It uses only contemporaneously observable feature values across the stocks available at time $t$. If $s_{k,t}$ equals zero for any feature on a given day, the standardised value for that feature is set to zero for all stocks on that day, because the feature contains no cross-sectional information.

### 3.5.5 Outlier Treatment

Financial predictors could contain extreme observations, particularly momentum and trading-activity shocks. However, no full sample winsorisation was performed during feature engineering.

Predictors were winsorised at the 1st and 99th percentiles only after the chronological training samples had been defined.

For each OLS outer window and each final outer-window XGBoost model:

- the 1st- and 99th-percentile thresholds were estimated using the current outer training observations only;
- the outer training predictors were clipped to those thresholds; and
- the same thresholds were applied unchanged to the associated outer test observations.

For every XGBoost inner-validation fold:

- thresholds were estimated using the inner training observations only;
- the inner training predictors were clipped;
- the same fold-specific thresholds were applied unchanged to the corresponding validation block; and
- thresholds were re-estimated independently for the next fold.

This procedure prevented the future distribution of predictor values from influencing preprocessing decisions. It was consistent with the broader principle that forecasting evaluations had to preserve the temporal ordering of all model-development steps (Tashman, 2000).

Because OLS and XGBoost used the same outer training samples, their final outer-window winsorisation thresholds were identical.

### 3.5.6 Predictor Dependence and Multicollinearity Diagnostics

Several predictors captured related forms of price trend or trading activity.

For example:

- one-month momentum, RSI and the 50-day moving-average ratio all contained information about recent price strength;
- six-month momentum and the 200-day moving-average ratio both reflected longer-horizon trend behaviour; and
- Volume Shock and Value-Traded Shock both incorporated reported trading volume.

Pairwise Spearman correlations, Variance Inflation Factors and condition diagnostics were therefore calculated for the final predictor matrix.

The diagnostic audit identified:

- five feature pairs with an absolute Spearman correlation of at least 0.80;
- five predictors with VIF values of at least 5;
- two predictors with VIF values of at least 10;
- a maximum VIF of 13.369;
- a full predictor correlation-matrix rank of nine;
- a correlation-matrix condition number of 94.323; and
- a maximum condition index of 9.712.

The predictor matrix was not singular. However, the results indicated that individual OLS coefficients could be sensitive to overlapping predictor information.

The complete nine-feature specification was retained because the objective was comparative out-of-sample prediction rather than structural or causal interpretation. Multicollinearity did not mechanically invalidate OLS predictions, but it reduced the reliability of interpreting individual coefficients as isolated effects (Wooldridge, 2010).

Similarly, correlated variables could substitute for one another in XGBoost tree splits. OLS coefficients and XGBoost gain-importance values were therefore interpreted as conditional model quantities rather than causal effects.

### 3.5.7 Leakage Prevention Rules

The feature and target pipeline followed the following leakage-prevention rules:

- confirmed price corrections were completed before target or feature construction;
- market-wide non-trading dates were removed before rolling calculations;
- forward returns were calculated on complete cleaned security histories before membership filtering;
- rolling predictors were calculated on complete histories before active-universe filtering;
- features used information associated with date $t$ or earlier;
- the target used the adjusted-close return from date $t$ to the next retained market date;
- daily standardisation did not use the target;
- full-sample winsorisation was not performed;
- winsorisation thresholds were estimated from the relevant training sample only;
- XGBoost hyperparameters were selected using inner chronological validation;
- one date was purged before every validation and test block; and
- each outer test observation was evaluated only after all training and model-selection decisions had been finalised.

## 3.6 Pooled Cross-Sectional Ordinary Least Squares

### 3.6.1 Purpose of the OLS Benchmark

Pooled Cross-Sectional OLS was used as the transparent linear benchmark. It tested whether the nine engineered features contained a stable linear association with subsequent relative stock performance.

The term "pooled" indicated that all security-day observations within a training window were combined into one regression system. Separate security-specific regressions were not estimated, and no security or date fixed effects were included.

The model therefore estimated one common intercept and one common coefficient for each standardised feature across the full training panel.

OLS provided an interpretable reference model against which the additional complexity of XGBoost could be evaluated. If XGBoost produced superior out-of-sample rankings, the improvement could have indicated that non-linearities, threshold effects or interactions contained predictive information that was not captured by the linear specification. Conversely, failure to outperform OLS would indicate that the additional flexibility had not produced a material predictive advantage for the selected features and horizon.

### 3.6.2 OLS Model Specification

Following the paradigm set by Fieberg et al. (2023), the traditional benchmark is a Pooled Cross-Sectional Ordinary Least Squares regression. "Pooled" means the model does not look at stocks individually; it aggregates all stocks and all historical days into one massive matrix to find universal linear relationships across the entire market.

For each outer training window, the pooled regression was specified as:

$$Y_{i,t+1}=\beta_{0}+\sum_{k=1}^{K}\beta_{k}Z_{i,t,k}+\epsilon_{i,t+1}$$

where:

- $Y_{i,t}$ represented the realised percentile target associated with feature date $t$;
- $\beta_{0}$ represented the common intercept;
- $\beta_{k}$ represented the coefficient assigned to standardised predictor $k$;
- $Z_{i,t,k}$ represented the standardised value of predictor $k$;
- $\epsilon_{i,t}$ represented the regression residual; and
- $K=9$ represented the number of predictors.

The coefficient vector was estimated by minimising the pooled residual sum of squares:

$$\hat{\beta}=\arg\min_{\beta}\sum_{t\in T_{train}}\sum_{i\in A_{t}^{M}}\left[Y_{i,t+1}-\beta_{0}-\sum_{k=1}^{K}\beta_{k}Z_{i,t,k}\right]^{2}$$

This followed the standard OLS estimation framework described by Wooldridge (2010).

The fitted raw OLS score for an out-of-sample observation was:

$$\hat{P}_{i,t+1\vert{}t}^{OLS}=\hat{\beta}_{0}+\sum_{k=1}^{K}\hat{\beta}_{k}Z_{i,t,k}$$

The coefficients were re-estimated independently for every outer expanding window.

### 3.6.3 How OLS Produces Asset Rankings

OLS was not used as a specialised ranking algorithm. It was used as a linear scoring model whose continuous predicted scores were sorted within each out-of-sample cross-section to produce daily rankings. This scoring-and-sorting procedure is an adaptation of the cross-sectional forecasting logic employed by Fieberg et al. (2023). Whereas their framework evaluates predicted cross-sectional stock returns, the present study applies the same general linear-scoring principle to a study-specific percentile-ranking target. The ranking logic follows the general framework described in Section 3.2.3 and is applied identically to both OLS and XGBoost.

### 3.6.4 Numerical Illustration of OLS Ranking

Suppose the fitted OLS model produces the following predicted scores for three stocks on a given day:

| Stock | Momentum signal | Volatility signal | OLS score | Predicted rank |
|-------|-----------------|-------------------|-----------|----------------|
| A     | High            | Medium            | 0.72      | 1              |
| B     | Low             | High              | 0.21      | 3              |
| C     | Medium          | Low               | 0.55      | 2              |

OLS has not directly labelled the stocks as first, second, and third. It has only produced scores. The ranking is obtained by sorting the scores: Stock A is ranked first because 0.72 is the highest score, Stock C is ranked second because 0.55 is the next highest score, and Stock B is ranked third because 0.21 is the lowest score. This example shows how a linear regression model can be used to construct a cross-sectional ranking.

### 3.6.5 Coefficient Inference and Stability

The pooled observations were not assumed to be fully independent. Securities observed on the same date could share common market shocks, while repeated observations for the same security could exhibit temporal dependence.

Two-way cluster-robust coefficient standard errors were therefore calculated by:

- trading date; and
- security ticker.

This covariance procedure allowed for within-date cross-sectional dependence and within-security temporal dependence.

For each coefficient, the following were stored:

- the estimated coefficient;
- the two-way cluster-robust standard error;
- a normal-approximation test statistic;
- a two-sided p-value; and
- a 95% confidence interval.

The coefficient results were used to describe the stability and uncertainty of the linear scoring rule across expanding training windows. They were not used as the primary statistical comparison between OLS and XGBoost.

Coefficient stability was also examined by comparing the signs and magnitudes of the nine feature coefficients across the nine outer windows.

### 3.6.6 Treatment of Out-of-Range OLS Predictions

Because OLS is a linear model, its predicted scores are not mechanically restricted to the interval from 0 to 1, even though the target variable is bounded (Wooldridge, 2010). This does not invalidate the ranking application because the ordering of the scores is more important than their absolute level. For ranking, Information Coefficient, and portfolio construction, the raw OLS scores are retained. For error-magnitude metrics such as RMSE and MAE, the main specification also uses the raw scores to avoid artificially altering the model output. A bounded robustness check may additionally clip predictions to the interval [0, 1] to confirm that conclusions are not driven by extreme OLS predictions.

## 3.7 Extreme Gradient Boosting

### 3.7.1 Purpose of XGBoost

XGBoost was used as the non-linear model. Unlike OLS, which imposed a constant additive coefficient for each feature, XGBoost could learn:

- non-linear predictor effects;
- threshold relationships;
- interactions among predictors;
- conditional effects; and
- different response patterns in different regions of the predictor space.

For example, the association between momentum and the target could have differed when volatility was high compared with when volatility was low. Such relationships would have required manually specified transformations and interactions in a conventional linear model.

XGBoost's flexibility also increased the risk of overfitting. Its hyperparameters and boosting duration were therefore selected through nested chronological validation rather than from the final out-of-sample test periods (Chen and Guestrin, 2016; Cawley and Talbot, 2010).

### 3.7.2 XGBoost Model Specification

XGBoost builds an additive ensemble of decision trees. The final prediction is the sum of the initial prediction and the weighted outputs of all $M$ sequential trees (Chen and Guestrin, 2016).

The prediction for stock $i$ at time $t+1$ is expressed as:

$$\hat{Y}_{i,t+1\vert{}t}^{XGB}=\hat{Y}_{0}+\sum_{m=1}^{M}\eta\cdot f_{m}(X_{i,t})$$

Where:

- $\hat{Y}_{i,t+1}^{XGB}$: The final predicted percentile rank for asset $i$ for the forward trading day $t+1$.
- $\hat{Y}_{0}$: The initial baseline prediction, which serves as the starting guess (typically set to the global mean of the target variable, $\overline{Y}=0.5$).
- $M$: The total number of sequential decision trees constructed in the ensemble.
- $\eta$: The learning rate (or shrinkage parameter), where $0<\eta\le1$. This scales the step size of each subsequent tree to prevent overfitting and ensure stable convergence.
- $f_{m}(X_{i,t})$: The mathematical mapping function of the $m$-th individual regression tree, which outputs a continuous prediction score based on the vector of standardized features $X_{i,t}$ observed at time $t$.
- $X_{i,t}^{k,std}$ is the nine-dimensional standardised feature vector.

**Regularised Objective (Loss) Function**

During the training phase, XGBoost optimizes its tree structures by minimizing a regularized objective function. This function balances empirical predictive accuracy against model complexity to prevent overfitting to historical market noise (Chen and Guestrin, 2016):

$$Loss=\sum_{j=1}^{n}(Y_{j}-\hat{Y}_{j})^{2}+\sum_{m=1}^{M}\Omega(f_{m})$$

Where:

- $Loss$: The global objective value to be minimized across the entire training sample.
- $n$: The total number of observations (the pooled rows combining all assets across all training days).
- $Y_{j}$: The actual, realised target percentile rank for observation $j$.
- $\hat{Y}_{j}$: The total aggregated prediction made by the ensemble up to the current boosting iteration for observation $j$.
- $(Y_{j}-\hat{Y}_{j})^{2}$: The Mean Squared Error (MSE) training loss, acting as the differentiable objective that measures empirical prediction errors.
- $\Omega(f_{m})$: The structural regularisation penalty applied to the $m$-th individual tree to control its complexity.

**Structural Regularisation Component**

The structural penalty can be represented as:

$$\Omega(f_{m})=\gamma T_{m}+\frac{1}{2}\lambda\sum_{r=1}^{T_{m}}w_{m,r}^{2}+\alpha\sum_{r=1}^{T_{m}}\vert{}w_{m,r}\vert{}$$

where:

- $T_{m}$ is the number of terminal leaves;
- $w_{m,r}$ is the prediction assigned to leaf $r$;
- $\gamma$ penalises additional leaves;
- $\lambda$ represents $L_{2}$ regularisation; and
- $\alpha$ represents $L_{1}$ regularisation.

### 3.7.3 XGBoost Ranking Procedure

The XGBoost model is evaluated using the same ranking procedure as OLS. On each out-of-sample trading day, XGBoost produces one predicted score for every stock in the active cross-section. These scores are then sorted cross-sectionally:

$$Predicted\_Rank_{i,t+1}^{XGB} = rank_{ASC}(\hat{Y}_{i,t+1}^{XGB} \vert{} i \in A_{t})$$

This ensures that the comparison between OLS and XGBoost is methodologically equivalent. Both models output predicted scores; both scores are transformed into rankings using the same sorting rule; and both are evaluated against the same realised percentile target. The use of gradient-boosted trees in a cross-sectional equity prediction setting is consistent with Fieberg et al. (2023). However, the application of the XGBoost score to the study-specific daily percentile-ranking target represents the particular implementation adopted in this research.

### 3.7.4 Hyperparameter Tuning

Hyperparameters were selected separately inside each outer training window. The prespecified grid varied:

- learning rate: {0.02, 0.05};
- maximum depth: {2, 3};
- minimum child weight: {5, 15};
- row subsample: {0.8, 1.0}; and
- column subsample: {0.8, 1.0}.

The Cartesian product produced 32 candidate configurations. The fixed values were reg_alpha = 0.10, reg_lambda = 10, gamma = 0, a squared-error regression objective, histogram-based tree construction, a maximum of 1,000 boosting rounds, early-stopping patience of 50 rounds and random seed 42.

Within each inner validation fold, RMSE was used as the early-stopping criterion to determine the number of boosting rounds. The best candidate configuration was then selected based on the highest mean daily Spearman Information Coefficient averaged across the three inner validation folds. Where two or more candidates achieved the same mean validation IC, tie-breaks favoured lower RMSE, shallower maximum depth, fewer median boosting rounds and lower candidate identifier.

Restricting the search to shallow, regularised trees was appropriate for a noisy target and a relatively small daily cross-section. Conducting model selection separately within each outer training sample prevented the final test quarters from influencing the tuning process (Cawley and Talbot, 2010; Chen and Guestrin, 2016).

### 3.7.5 Early Stopping

Early stopping is applied within every inner validation fold. For a given candidate configuration:

- the model is fitted on the inner training sample;
- performance is evaluated on the later inner validation block using RMSE;
- training stops when validation RMSE fails to improve for the pre-specified patience period of 50 boosting rounds; and
- the best boosting round is recorded for that fold.

Once the best candidate configuration has been selected based on mean validation IC across the three inner folds, the final number of boosting rounds is determined by taking the median of the three fold-specific best-round values for that candidate, rounding to the nearest integer and applying a floor of one.

The model is then refitted using the complete outer training sample for the determined number of boosting rounds and evaluated once on the untouched outer test quarter.

This procedure prevents the outer test sample from determining either the hyperparameters or the number of fitted trees.

### 3.7.6 Feature Importance

Gain-based feature importance is calculated for each final outer-window XGBoost model. For every model, the feature-importance values are normalised to sum to one.

Gain importance measures the reduction in the model objective associated with tree splits that use a particular feature.

It should not be interpreted as a causal measure. Correlated predictors may substitute for one another. A zero importance in one test window means only that the fitted tree ensemble did not use the variable in its splits for that window.

The stability of importance values across expanding windows is therefore considered in addition to the average importance level.

## 3.8 Nested Expanding-Window Validation Framework

### 3.8.1 Expanding Outer Windows

The study uses an expanding-window walk-forward framework. The final modelling panel covers the period from 19 October 2016 to 18 March 2026. Formal out-of-sample evaluation begins on 2 January 2024.

Nine outer test windows are used:

- 2024 Q1;
- 2024 Q2;
- 2024 Q3;
- 2024 Q4;
- 2025 Q1;
- 2025 Q2;
- 2025 Q3;
- 2025 Q4; and
- the available portion of 2026 Q1.

The first model is trained from 19 October 2016 to 28 December 2023. The date 29 December 2023 is purged, and the first test window runs from 2 January 2024 to 28 March 2024.

After the first test quarter, the training sample expands to incorporate the additional historical observations. The models are then re-estimated and tested on the subsequent quarter.

This process continues until the final available test date.

The general structure is:

- Window 1: train using the initial historical period and test on the first future quarter;
- Window 2: expand the training period and test on the next quarter;
- Subsequent windows: continue the chronological expansion until all eligible out-of-sample observations have been evaluated.

The outer test windows do not overlap. Every model-eligible observation from January 2024 onward receives exactly one out-of-sample prediction from OLS and one from XGBoost.

### 3.8.2 One-Date Target-Horizon Purge

The target for feature date $t$ uses the adjusted closing price on the following market date. Consequently, if the final training feature date immediately precedes the first test date, its target may contain the first test-date closing price.

To prevent this boundary overlap, one market date is purged immediately before every outer test period and before every inner validation block.

If $T_{w}^{test}$ denotes the first test date in window $w$, then the market date immediately before $T_{w}^{test}$ is designated as the purged boundary date. The training sample ends one market date before the purge date.

This ensures that no training-set target value incorporates any price information from the test period.

### 3.8.3 Inner Chronological Validation Structure

For each outer training window, XGBoost hyperparameters are selected through a nested chronological validation procedure. The final 378 eligible training dates are divided into three consecutive 126-trading-date validation blocks.

Each inner fold:

1. uses all training dates that precede the validation block as the inner training sample;
2. removes the single trading date immediately preceding the validation block as a purge boundary; and
3. evaluates each candidate configuration on the 126-date validation block.

A minimum of 504 inner training dates is required before the first validation block. This ensures that each inner fold has sufficient historical observations for stable model estimation.

The inner-fold structure mirrors the outer-window purge logic: no inner validation observation enters the inner training sample, and the purge boundary prevents the inner training target from incorporating information from the inner validation period.

### 3.8.4 Inner-Fold Preprocessing

Within each inner validation fold, winsorisation thresholds are estimated using the inner training observations only. The inner training predictors are clipped to those thresholds, and the same fold-specific thresholds are applied unchanged to the corresponding inner validation block. Thresholds are re-estimated independently for each subsequent fold.

This fold-specific preprocessing ensures that the future distribution of predictor values does not influence the model or its preprocessing during inner validation, consistent with the training-only winsorisation procedure described in Section 3.5.5.

### 3.8.5 Final Outer-Window Model Fitting

After the inner validation procedure identifies the best candidate configuration, the final XGBoost model for each outer window is fitted as follows:

1. The number of boosting rounds is set to $\text{round}(\text{median}(M_1^*, M_2^*, M_3^*))$, where $M_k^*$ denotes the best boosting round identified by early stopping in inner fold $k$, with a floor of one.
2. Winsorisation thresholds are re-estimated using the complete outer training sample.
3. The outer training predictors are winsorised using those thresholds, and the same thresholds are applied unchanged to the outer test predictors.
4. The model is refitted on the complete outer training sample using the selected hyperparameters and determined boosting rounds.
5. The refitted model is evaluated once on the outer test quarter.

OLS follows the same outer-window structure, except that no hyperparameter tuning or early stopping is required: the coefficients are simply re-estimated on each expanding outer training sample.

### 3.8.6 Alternative and Expanded Experimental Setups

In addition to the primary experimental setup, two robustness configurations were evaluated:

**Alternative Setup (21-Day Horizon, 96-Candidate Grid)**

The target variable was re-engineered as a 21-trading-day forward return percentile rank, with the purge buffer expanded to 21 trading dates to prevent overlap leakage. The XGBoost hyperparameter grid was expanded from 32 to 96 candidate configurations by adding $L_2$ leaf-weight regularisation as a searchable dimension: $\lambda \in \{1.0, 5.0, 10.0\}$. The Newey-West lag parameter was adjusted to $q = 21$ to account for the overlapping return horizon. All other procedures remained identical to the primary setup.

**Expanded Grid Setup (1-Day Horizon, 96-Candidate Grid)**

To isolate the effect of regularisation flexibility from the holding-horizon change, the primary 1-day target was retained while expanding the XGBoost grid to the same 96 candidates ($\lambda \in \{1.0, 5.0, 10.0\}$). This configuration tested whether the fixed $\lambda = 10$ in the primary setup artificially constrained the tree ensemble.

Both robustness setups used the same walk-forward outer windows, inner validation structure, purge boundaries, features and evaluation metrics as the primary pipeline. A three-way comparison across all three setups was performed to assess the sensitivity of the findings to horizon and regularisation choices.

## 3.9 Evaluation Metrics

### 3.9.1 Spearman Information Coefficient (IC)

The primary evaluation metric is the daily Spearman rank correlation coefficient between predicted scores and realised percentile targets. For each out-of-sample trading date $t$, the Information Coefficient is defined as:

$$IC_{t} = 1 - \frac{6\sum_{i=1}^{N_{t}^{M}} d_{i,t}^{2}}{N_{t}^{M}(N_{t}^{M^2} - 1)}$$

where $d_{i,t}$ is the difference between the predicted rank and the realised rank of security $i$ on date $t$, and $N_{t}^{M}$ is the model-eligible cross-sectional size.

The IC measures ordinal agreement between predicted and realised rankings. A positive IC indicates that the model's ordering of securities is positively associated with their subsequent relative performance. An IC of zero indicates no rank association, while a negative IC indicates that the model's ordering is inversely associated with realised outcomes.

The IC is invariant to monotonic transformations of the raw predicted scores. It therefore evaluates the model's ranking ability rather than the accuracy of its continuous point predictions (Grinold and Kahn, 2000).

### 3.9.2 Mean IC and IC Information Ratio

The mean IC across all $T$ out-of-sample trading dates summarises the average predictive signal:

$$\overline{IC} = \frac{1}{T}\sum_{t=1}^{T} IC_{t}$$

The IC Information Ratio (ICIR) adjusts the mean IC for its variability:

$$ICIR = \frac{\overline{IC}}{\sigma_{IC}}$$

where $\sigma_{IC}$ is the standard deviation of the daily IC series.

A higher ICIR indicates a more consistent and reliable predictive signal. The ICIR is closely related to the signal-to-noise ratio used in quantitative portfolio management (Grinold and Kahn, 2000). An ICIR near zero indicates that any mean ranking ability is overwhelmed by daily noise.

### 3.9.3 Error-Magnitude Metrics

Two observation-level error metrics supplement the rank-correlation measures:

The Mean Absolute Error across all $n$ out-of-sample stock-day observations is:

$$MAE = \frac{1}{n}\sum_{j=1}^{n} |Y_{j} - \hat{Y}_{j}|$$

The Root Mean Squared Error is:

$$RMSE = \sqrt{\frac{1}{n}\sum_{j=1}^{n} (Y_{j} - \hat{Y}_{j})^{2}}$$

where $Y_{j}$ is the realised percentile target and $\hat{Y}_{j}$ is the predicted score for observation $j$.

These metrics measure the absolute distance between predicted scores and realised targets. Because the target is uniformly distributed on $[0, 1]$, the theoretical MAE and RMSE of a constant mean prediction ($\hat{Y} = 0.5$) are $0.250$ and $\sqrt{1/12} \approx 0.289$, respectively. Empirical values near these baselines indicate that the models capture little point-forecast accuracy beyond the unconditional mean, even when the ranking signal (IC) may be positive (Hyndman and Koehler, 2006).

### 3.9.4 Top-Quartile Hit Rate

The top-quartile hit rate evaluates tail discrimination—the model's ability to identify the securities that will subsequently realise the highest returns.

For each out-of-sample trading date $t$, the top-quartile hit rate is defined as:

$$HitRate_{t} = \frac{|TopPredicted_{t} \cap TopRealised_{t}|}{|TopPredicted_{t}|}$$

where $TopPredicted_{t}$ denotes the set of securities with predicted scores in the top quartile on date $t$, and $TopRealised_{t}$ denotes the set of securities whose realised returns fell in the top quartile.

An analogous bottom-quartile hit rate is calculated for the lowest-ranked securities. The theoretical baseline for a random model is 25.0%. Hit rates above this baseline indicate that the model provides useful tail-discrimination ability for long-short portfolio construction.

### 3.9.5 Quarterly Stability

Performance is decomposed across the nine individual outer test windows to assess temporal stability. Quarterly mean IC, positive-IC proportions and quarterly hit rates are reported to identify whether overall performance is driven by broad consistency or by a small number of exceptional quarters.

### 3.9.6 Percentage of Positive-IC Trading Days

The proportion of out-of-sample trading days on which $IC_t > 0$ is reported as a supplementary consistency measure. A model with a positive mean IC but fewer than 50% positive-IC days generates its average signal from occasional large positive spikes rather than from frequent small successes. A higher proportion of positive-IC days indicates more stable day-to-day ranking skill.

## 3.10 Statistical Significance Testing

### 3.10.1 Paired Newey-West HAC Inference

The primary statistical test evaluates whether the mean daily IC difference between XGBoost and Pooled OLS is significantly different from zero.

The daily paired IC difference is defined as:

$$\Delta_t = IC_{t}^{XGB} - IC_{t}^{OLS}$$

where a positive value indicates that XGBoost achieved higher rank correlation than OLS on date $t$.

The null and alternative hypotheses for the directional test of XGBoost superiority are:

$$H_0: E[\Delta_t] \le 0$$
$$H_1: E[\Delta_t] > 0$$

A two-sided test ($H_1: E[\Delta_t] \ne 0$) is also conducted to evaluate whether either model is statistically distinguishable from the other.

Because daily financial performance measures may exhibit serial dependence and heteroskedasticity, inference based on ordinary independent-observation standard errors may be unreliable. The study therefore applies heteroskedasticity and autocorrelation consistent inference using the Newey-West covariance estimator (Newey and West, 1987).

The test is implemented by regressing the paired IC-difference series on a constant:

$$\Delta_t = \mu + \varepsilon_t, \quad t = 1, \dots, T$$

where $\mu = E[\Delta_t]$ is estimated by the sample mean of the difference series, and standard errors are estimated using the Newey-West HAC estimator with a Bartlett kernel.

### 3.10.2 Newey-West Lag Selection

The maximum lag parameter for the Newey-West estimator is determined using the sample-size-based formula:

$$q = \lfloor 4(T/100)^{2/9} \rfloor$$

where $T$ is the number of paired daily observations, with a minimum lag of 1. For the primary evaluation period of $T = 554$ trading dates, this formula yields $q = 5$ trading days. For the 21-day alternative setup, the lag parameter was set to $q = 21$ to match the overlapping return horizon.

The test statistic is:

$$z = \frac{\hat{\mu}}{\widehat{SE}_{NW}}$$

where $\widehat{SE}_{NW}$ is the Newey-West HAC standard error. Under the null hypothesis, $z$ is asymptotically standard normal. One-sided and two-sided p-values are computed from the standard normal distribution.

### 3.10.3 Loss-Difference Testing

For predictive loss comparisons, the daily loss difference is defined as:

$$d_{t} = Loss_{t}^{OLS} - Loss_{t}^{XGB}$$

where a positive value indicates that XGBoost has lower loss than OLS on day $t$. A Diebold-Mariano-style test (Diebold and Mariano, 1995) may then be applied to the daily loss-difference series, with robust standard errors to account for time dependence.

## 3.11 Reproducibility and Audit Trail

The empirical pipeline was implemented through separate Python scripts for data extraction, price-scale auditing and correction, universe construction, target engineering, target diagnostics, feature engineering, model-panel diagnostics, validation setup, OLS estimation, XGBoost tuning and estimation, and paired model comparison.

Each stage wrote core datasets, audit files and figures to dedicated project directories. Integrity checks covered duplicate keys, missing and non-finite values, price-scale candidates, active-universe counts, next-date return links, target bounds, daily target means, feature-standardisation checks, model cross-sectional sizes, purged boundaries, preprocessing thresholds, out-of-sample coverage and equality of OLS and XGBoost evaluation samples.

The XGBoost random seed was fixed at 42, and software versions were stored in the estimation audit. Shared targets, returns and cross-sectional sizes were compared at a tolerance of $10^{-10}$. A tolerance of $10^{-8}$ was used only when daily metrics were reproduced after predictions had been written to and read from CSV files, accommodating immaterial floating-point round-trip differences.

## 3.12 Methodological Limitations

Several limitations remained. First, 11 requested ticker histories were unavailable, so the dynamic universe mitigated but did not eliminate survivorship and data-availability bias. Second, the liquidity screen represented a researcher-constructed proxy and not an exact official FTSE/JSE Top 40 history. Third, the study used daily OHLCV data and did not include intraday prices, spreads, order-book conditions, fundamentals, analyst information or macroeconomic predictors.

Fourth, the one-period target was highly noisy, and some medium- and long-horizon indicators might not have been optimally aligned with next-day return variation. Fifth, the percentile transformation removed the magnitude of realised return differences. Sixth, features recorded for date $t$ were associated with the close-to-close return from $t$ to $t+1$, but the study did not claim same-close implementability.

Seventh, the complete-feature requirement reduced the model-eligible daily cross-section below 40 on some dates. Eighth, correlated predictors complicated individual coefficient and feature-importance interpretation. Ninth, the primary XGBoost setup used a prespecified 32-candidate grid. An expanded 96-candidate grid was evaluated as a robustness check, but the search space remained manually defined rather than optimised through Bayesian or random search methods. Finally, no economic portfolio backtest was implemented. The conclusions were therefore limited to comparative statistical ranking performance.

## 3.13 Chapter Summary

This chapter described a controlled methodology for comparing Pooled Cross-Sectional OLS with XGBoost for one-trading-date-ahead cross-sectional ranking on the JSE. A cleaned broad panel was used to construct a point-in-time liquidity-screened active universe. Adjusted-close forward returns were ranked within each date and transformed into a percentile target. Nine technical and trading-activity predictors were calculated on complete security histories, standardised within each daily model cross-section and winsorised using training-sample thresholds only.

OLS estimated a common linear scoring rule, while XGBoost estimated a regularised non-linear tree ensemble. Both models used identical outer training and test samples. XGBoost tuning was conducted through nested purged chronological validation using mean daily IC as the selection criterion, while RMSE served as the inner-fold early-stopping metric. The final number of boosting rounds was determined by the median of the fold-specific best-round values. Final evaluation used nine quarterly expanding windows covering 554 forecast-origin dates and 22,136 paired stock-day predictions.

Two robustness configurations supplemented the primary design: an alternative 21-day holding horizon with an expanded 96-candidate hyperparameter grid, and an expanded 96-candidate grid applied to the original 1-day horizon. A three-way comparison across all three setups assessed the sensitivity of the findings to both holding horizon and regularisation flexibility.

Daily Spearman IC was the primary performance measure. IC stability, MAE, RMSE, quartile hit rates and quarterly profiles were used as supplementary evidence. The mean paired IC difference was tested using Newey-West HAC inference with automatic lag selection. The methodology therefore provided a temporally ordered, reproducible framework for assessing whether XGBoost's non-linear flexibility produced a statistically meaningful improvement over the transparent OLS benchmark.
