---
title: "Assignment 1 - Forecasting Daily Traffic at the Baregg Tunnel"
author: "Sharon Varughese"
editor_options: 
  markdown: 
    wrap: 72
---

# Introduction

The data set shows the daily vehicle counts in the Baregg Tunnel which
is located in Switzerland's A1 motorway.This data set has strong weekly
seasonality patterns.

1.  Measures - Daily count of vehicles
2.  Time Span - Nov 2003 – Nov 2005
3.  Frequency - Daily observations
4.  Total observations - \~750
5.  The forecasting objective - The goal is to create a model that
    forecasts the daily traffic at Baregg Tunnel as accurately as
    possible. The analysis will include two models - the Naive model and
    the linear regression model - to determine which model is more
    accurate in this case.

# Data Exploration

```{r load-packages}
#── Load packages ────────────────────────
library(fpp3)    # loads tsibble, fable, feasts
#fpp3: tsibble, fable, feasts, and ggplot2.
```

```{r load-data}
# ── Read CSV & create a tsibble ──────────
baregg <- read.csv("BareggTunnel.csv") |>
  mutate(Day = dmy(Day)) |>
  as_tsibble(index = Day)
# dmy() from lubridate automatically parses "01 Nov 2003" format.
# as_tsibble(index = Day) declares the time column
# now R "knows" this is a time series.
```

```{r full-timeseries-plot, fig.cap="Figure 1: Complete time series showing daily vehicle count"}
# ── Quick plot ───────────────────────────
autoplot(baregg, Number.of.vehicles) +
  labs(title = "Baregg Tunnel: Daily Vehicle Count",
       y = "No. of Vehicles", x = "Date")
```

## Key Observations

1.  Relatively stable trend line which remains around 100k throughout
    the data, depicting regular fluctuations.
2.  There is strong 7-day seasonality, and clear cyclical patterns which
    consistently drop (weekend) and rise (weekdays).
3.  There are not many extreme values or outliers, except for a few such
    as 2004-01 drop.

# Methodology

## Partitioning Strategy

Training: Nov 2003 – Jun 2005 ~607 days (82%) 
Validation: Jul 2005 – Nov 2005 ~139 days (18%)

Covers 20 full weekly cycles. The model is trained on the historical
data to predict the future. The split also respects the time order
instead of trying to fit the data in a perfect 70 or 80 percent.

```{r partition-data}
# ── Partition ────────────────────────────

train <- baregg |>
filter(Day < ymd("2005-07-01"))
valid <- baregg |>
filter(Day >= ymd("2005-07-01"))
```

## Model 1: Naïve Forecast (Benchmark)

This benchmark model is quite simple and straightforward. This model
means tomorrow's forecast = today's value. This serves as an excellent
baseline benchmark where any model has to simply outperform Naïve to
justify its added complexity.

The Mean Absolute Scaled Error (MASE) metric explicitly measures
performance relative to the Naïve forecast, making it an essential
reference point. 
1. MASE<1 (Outperforms the Naïve Model ) 
2. MASE>1 (Fails to outperform the Model)

```{r naive-model}

# ── Fit the naïve model on TRAINING data ──

fit_naive <- train |>
model(Naive = NAIVE(Number.of.vehicles))
# ── Generate forecasts for validation ────

fc_naive <- fit_naive |>
  forecast(new_data = valid)

# ── Plot: forecast vs actual ─────────────

fc_naive |> autoplot(baregg, level = NULL) + labs(title = "Naïve Forecast vs Actual", y = "No. of Vehicles", x = "Date") + theme_minimal()

# ── Accuracy check ─────────────────

accuracy(fc_naive, baregg)
```

As seen in the plot, the naive model is a horizontal flat line that does not capture the cyclical pattern of the daily traffic data.

## Model 2: Linear Regression

The TSLM = Time Series Linear Model uses OLS regression where predictors
come from time itself. It captures trend which includes long term growth
or decline and day-of-week pattern/seasonality of the traffic. Moreover,
unlike naive, regression produces a wavy forecast to capture the weekly
traffic cycle.

The regression equation: y = β₀ + β₁·t + β₂·Mon + β₃·Tue + ... +
β₇·Sat + ε

```{r lm-model}
# ── Fit linear regression on TRAINING data

fit_lm <- train |>
model(
LinReg = TSLM(Number.of.vehicles ~
trend() + season("week"))
)

# ── Inspect the coefficients ─────────────

report(fit_lm)

# ── Forecast the validation period ───────

fc_lm <- fit_lm |>
forecast(new_data = valid)

# ── Plot: forecast vs actual ─────────────

fc_lm |>
autoplot(baregg, level = 95) +
labs(title = "Regression Forecast vs Actual",
y = "Vehicles", x = "Date") +
theme_minimal()

accuracy(fc_lm, baregg)
```

The linear regression model captures the daily traffic better than the
naive model with the pattern showing ups and downs, aligning with the
historical traffic data.

# Results

## Step 4

### Forecast Comparison and Accuracy Metrics
In the following code, we fit both the models together to generate forecasts on the validation set. 

```{r fit-both-models}
# Fit BOTH models in one call

fit_both <- train |>
  model(
    Naive = NAIVE(Number.of.vehicles),
    LinReg = TSLM(Number.of.vehicles ~ trend() + season("week"))
  )
# ── Forecast both on validation set ─────────────────────────────────
fc_both <- fit_both |> forecast(new_data = valid)

```

### Overlay Plot

```{r overlay-plot, fig.cap="Figure 2: Comparison of Naïve and Linear Regression forecasts against actual validation data"}
fc_both |>
  autoplot(baregg, level = NULL) +
  labs(title = "Comparison of Naïve vs Regression Models",
       y = "No. of Vehicles") +
  guides(colour =
           guide_legend(title = "Model"))
```
In the overlay plot, we can observe as follows -
1. The Naive model is a horizontal line which fails to capture the weekly traffic in the actual data.
2. The Linear regression model captures the wavy pattern of the weekly traffic closely, which makes it better in generating forecasts on the validation period.


### Accuracy Metrics for both models

```{r Accuracy Metrics for both models}
# ── Accuracy table: the key comparison ──────────────────────────────
accuracy(fc_both, baregg) |>
  select(.model, ME, RMSE, MAE, MAPE, MASE)
```

### Manual MASE Calculation

```{r mase-manual}
# ── MASE (manual — daily data needs this) ────

mae_naive_train <- mean(abs(diff(train$Number.of.vehicles)))
cat("MASE Naive:", round(mean(abs(valid$Number.of.vehicles - fc_naive$.mean)) / mae_naive_train, 3), "\n")
cat("MASE LinReg:", round(mean(abs(valid$Number.of.vehicles - fc_lm$.mean)) / mae_naive_train, 3), "\n")

cat("\nRecall: If MASE < 1 → beats naïve benchmark\n")
```

### Interpretation of results

1. MASE (Mean Absolute Scaled Error) 
- This metric is significant in generating a comparison for the Naive model
- If MASE < 1, then the model has outperformed Naive benchmark
- According to the results of the accuracy metrics of linear regression, MASE = 0.798, and the manual calculation of LinReg, MASE = 0.414, both outperform the naive benchmark.

2. ME (Mean Error)
- This metric tells the direction of bias.
- Positive is under-forecasts. Negative is over-forecasts.
- In the accuracy metrics, both models have a negative number which insinuates over forecasting.
- LinReg has a lower ME than naive, which indicates less bias in LinReg model.

3. MAPE (Mean Absolute % error)
-  This metric tells error as a percentage.
-  In the accuracy metrics, the LinReg model has a significantly lower error percentage (~3.7%) as compared to the naive model (~13.1%).

4. MAE (Mean Absolute Error)
- This metric tells you the average size of errors in actual units.
- In the accuracy metrics, the LinReg model has significantly lower (~3900) absolute errors than the naive model (~13600).

5. RMSE (Root Mean Squared Error)
- This metric is like MAE but penalizes large errors more heavily.
- RMSE ≥ MAE always.
- In the accuracy metrics, the LinReg model has significantly lower (~5870)  RMSE than the naive model (~16800).

## Step 5

### Compute & plot errors

```{r error-analysis}
fc_both |>
  accuracy(baregg, measures = list(
    ME = ME, MAE = MAE,
    RMSE = RMSE, MASE = MASE
  ))

```

### Residual diagnostics

```{r residual-diagnostics}
fit_both |>
  select(LinReg) |>
  ggtime::gg_tsresiduals() +
  labs(title = "LinReg Residuals")
```

### Interpretation of diagnostic plots

1. Does the forecast follow the actual pattern (ups and downs)?
 Yes, the Lin Reg forecast follows the up and down patterns closely. However, the naive model, which can be seen as a horizontal flat line, fails to capture the weekly cyclical patterns.
 
2. Is one model consistently above or below the actual values? What does this tell you about bias?
The accuracy metrics produced by the LinReg model (ME, MAPE,MAE, RMSE, MASE) are all lower compared to the naive model. The errors in the linReg model also seem to be closer to zero which indicates less bias in the forecasting. Although the ME is negative for both LinReg and Naive model, the former is significantly lower compared to the latter.

3. Are the forecast errors random or do they show a pattern over time?
The forecast errors in the Naive model show a systemic pattern whereas the LinReg model errors are random with no observable pattern. In the Naive model, when the traffic is high, the model under-forecasts and when the traffic is low, the model over-forecasts. This clearly suggests a predictable pattern overtime. However, the Linreg model seems to have random errors, which is better and desirable.

4. Is the histogram of errors approximately centered at zero?
Yes, the histogram of errors for linreg is mainly centered at zero, with a bell-shaped distribution. This tells us that the model is mostly unbiased and is more accurate in forecasting.

5. Does the ACF plot show any significant spikes? If so, what do they suggest?
There are observable spikes in lag 7, 14, 21, etc. which do suggest a pattern. This can be improved by using more sophisticated forecasting models such as ARIMA to improve the accuracy and reduce errors.

#Conclusion

In this analysis, the Linear regression model worked better in forecasting the daily traffic of Baregg tunnel than the naive model.
The linear regression model's MASE < 1, which indicates that the model outperformed the naive model.
In addition, the LinReg model was able to follow the weekly cyclical patterns of the actual data closely. The errors were also centered at zero with no systemic patterns. The model also showed low ME, MAE, and MAPE.

To make this model better -
1. The ARIMA modelling can take into account the autocorrelation in residuals and help in improving the overall accuracy of the predictions.
2. Taking into account the holiday patterns and seasonal adjustments can also help in improving the traffic predictions.


#References
Anthropic. (2025). Claude (version 4.5) [Large language model]. https://www.anthropic.com/claude
Dataset: Baregg Tunnel daily vehicle counts (November 2003 - November 2005)
 
 

 
 

