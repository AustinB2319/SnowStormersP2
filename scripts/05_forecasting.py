"""
05_forecasting.py
Snow Stormers – DS 4002, Spring 2026

Predictive modeling component: SARIMA time-series forecasting of monthly
federal inmate complaint counts.

Steps:
  1. Train/test split (2000–2020 train | 2021–2023 test)
  2. SARIMA order selection via AIC grid search
  3. Fit best SARIMA model on training data
  4. Evaluate on test set (MAE, RMSE, MAPE)
  5. Refit on full data and forecast 2024–2027
  6. (Optional) SARIMAX with inmate-to-officer ratio as exogenous covariate

Figures saved to figures/; model results printed to stdout.
Finish-line goal: MAPE < 15% on the 2021–2023 holdout set.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FIG_DIR  = os.path.join(os.path.dirname(__file__), "figures")

ACCENT = "#1f77b4"
RED    = "#d62728"
GREEN  = "#2ca02c"
ORANGE = "#ff7f0e"
CAPTION = "Source: BOP SENTRY database via Data Liberation Project (FOIA, 2024)"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.labelsize": 11,
})

# ── Load monthly complaints ──────────────────────────────────────────────────
monthly = pd.read_csv(f"{DATA_DIR}/monthly_complaints.csv", parse_dates=["year_month"])
monthly = monthly.set_index("year_month").sort_index()
ts = monthly["complaints"]

# ── Train / Test split ───────────────────────────────────────────────────────
TRAIN_END = "2020-12-01"
TEST_START = "2021-01-01"

train = ts[ts.index <= TRAIN_END]
test  = ts[ts.index >= TEST_START]
print(f"Training set: {train.index[0].date()} → {train.index[-1].date()}  ({len(train)} months)")
print(f"Test set:     {test.index[0].date()} → {test.index[-1].date()}  ({len(test)} months)")

# ── SARIMA order selection via AIC grid search ───────────────────────────────
print("\nSearching SARIMA orders … (this may take ~60 seconds)")

p_vals = range(0, 3)
d_vals = [1]
q_vals = range(0, 3)
P_vals = range(0, 2)
D_vals = [1]
Q_vals = range(0, 2)
S = 12

best_aic  = np.inf
best_order = None
best_sorder = None
results_list = []

for p, d, q, P, D, Q in itertools.product(p_vals, d_vals, q_vals,
                                            P_vals, D_vals, Q_vals):
    try:
        model = SARIMAX(train, order=(p, d, q),
                        seasonal_order=(P, D, Q, S),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        fit = model.fit(disp=False)
        results_list.append(((p, d, q), (P, D, Q, S), fit.aic))
        if fit.aic < best_aic:
            best_aic    = fit.aic
            best_order  = (p, d, q)
            best_sorder = (P, D, Q, S)
    except Exception:
        continue

results_list.sort(key=lambda x: x[2])
print(f"\nTop 5 SARIMA models by AIC:")
for order, sorder, aic in results_list[:5]:
    print(f"  SARIMA{order}×{sorder}  AIC={aic:.2f}")

print(f"\nSelected: SARIMA{best_order}×{best_sorder}  AIC={best_aic:.2f}")

# ── Fit best model on training data ─────────────────────────────────────────
model_train = SARIMAX(train,
                      order=best_order,
                      seasonal_order=best_sorder,
                      enforce_stationarity=False,
                      enforce_invertibility=False)
fit_train = model_train.fit(disp=False)

# ── Forecast on test set ─────────────────────────────────────────────────────
forecast_obj = fit_train.get_forecast(steps=len(test))
pred         = forecast_obj.predicted_mean
pred_ci      = forecast_obj.conf_int(alpha=0.10)   # 90% CI
pred.index   = test.index
pred_ci.index = test.index

# ── Evaluation metrics ───────────────────────────────────────────────────────
mae  = mean_absolute_error(test, pred)
rmse = np.sqrt(mean_squared_error(test, pred))
mape = (np.abs((test.values - pred.values) / test.values)).mean() * 100

print(f"\n── Test-Set Evaluation (2021–2023) ──────────────────")
print(f"  MAE:  {mae:,.0f} complaints/month")
print(f"  RMSE: {rmse:,.0f} complaints/month")
print(f"  MAPE: {mape:.1f}%")
print(f"  Finish-line goal: MAPE < 15%  →  {'✓ MET' if mape < 15 else '✗ NOT MET'}")

# ── Refit on full series → forecast 2024–2027 ────────────────────────────────
model_full = SARIMAX(ts,
                     order=best_order,
                     seasonal_order=best_sorder,
                     enforce_stationarity=False,
                     enforce_invertibility=False)
fit_full = model_full.fit(disp=False)

FORECAST_MONTHS = 48   # 4 years
forecast_full   = fit_full.get_forecast(steps=FORECAST_MONTHS)
fc_mean = forecast_full.predicted_mean
fc_ci   = forecast_full.conf_int(alpha=0.10)

# Build date index for the forecast period
last_date  = ts.index[-1]
fc_index   = pd.date_range(start=last_date + pd.DateOffset(months=1),
                            periods=FORECAST_MONTHS, freq="MS")
fc_mean.index = fc_ci.index = fc_index

# Save forecast to CSV
fc_df = pd.DataFrame({
    "year_month": fc_index,
    "forecast":   fc_mean.values,
    "lower_90":   fc_ci.iloc[:, 0].values,
    "upper_90":   fc_ci.iloc[:, 1].values,
})
fc_df.to_csv(f"{DATA_DIR}/forecast_2024_2027.csv", index=False)
print(f"\nSaved forecast to data/forecast_2024_2027.csv")

# ── Figure 17 – Test-set predictions vs actuals ──────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
# Show last 36 months of training for context
context = train[-36:]
ax.plot(context.index, context.values, color="lightsteelblue",
        linewidth=1, label="Training data (last 36 months)")
ax.plot(test.index,   test.values,  color=ACCENT, linewidth=2,
        label="Actual (2021–2023)")
ax.plot(pred.index,   pred.values,  color=RED, linewidth=2,
        linestyle="--", label=f"SARIMA forecast")
ax.fill_between(pred.index,
                pred_ci.iloc[:, 0], pred_ci.iloc[:, 1],
                color=RED, alpha=0.12, label="90% confidence interval")
ax.axvline(pd.Timestamp(TEST_START), color="gray", linewidth=1,
           linestyle=":", label="Train/test split")
ax.set_title(f"SARIMA{best_order}×{best_sorder} – Test-Set Forecast vs. Actuals\n"
             f"MAE={mae:,.0f}  RMSE={rmse:,.0f}  MAPE={mape:.1f}%")
ax.set_ylabel("Monthly Complaint Filings")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=9, framealpha=0.8)
ax.annotate(CAPTION, xy=(0, -0.13), xycoords="axes fraction",
            fontsize=8, color="gray")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig17_forecast_test.png", bbox_inches="tight")
plt.close()
print("Saved: fig17_forecast_test.png")

# ── Figure 18 – Full historical + 2024–2027 forecast ─────────────────────────
fig, ax = plt.subplots(figsize=(14, 5.5))
ax.fill_between(ts.index, ts.values, alpha=0.15, color=ACCENT)
ax.plot(ts.index, ts.values, color=ACCENT, linewidth=1,
        label="Observed (2000–2023)")
ax.plot(fc_mean.index, fc_mean.values, color=RED, linewidth=2.5,
        linestyle="--", label="Forecast (2024–2027)")
ax.fill_between(fc_mean.index,
                fc_ci.iloc[:, 0], fc_ci.iloc[:, 1],
                color=RED, alpha=0.15, label="90% confidence interval")
ax.axvline(pd.Timestamp("2024-01-01"), color="gray", linewidth=1,
           linestyle=":", label="Forecast start")

# Annotate average forecast level
avg_fc = fc_mean.mean()
ax.axhline(avg_fc, color=ORANGE, linewidth=1, linestyle="--",
           label=f"Avg forecast: {avg_fc:,.0f}/month")

ax.set_title("Federal Inmate Complaints: Historical Series & SARIMA Forecast (2024–2027)")
ax.set_ylabel("Monthly Complaint Filings")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=9, framealpha=0.8, loc="upper left")
ax.annotate(CAPTION, xy=(0, -0.12), xycoords="axes fraction",
            fontsize=8, color="gray")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig18_forecast_full.png", bbox_inches="tight")
plt.close()
print("Saved: fig18_forecast_full.png")

# ── Figure 19 – Annual forecast summary (2024–2027 vs recent years) ───────────
annual_hist = ts.resample("YE").sum()
annual_fc   = fc_mean.resample("YE").sum()

fig, ax = plt.subplots(figsize=(11, 4.5))
years_hist = annual_hist.index.year
years_fc   = annual_fc.index.year
ax.bar(years_hist[-6:], annual_hist.values[-6:], color=ACCENT,
       alpha=0.7, label="Observed annual total")
ax.bar(years_fc,        annual_fc.values,        color=RED,
       alpha=0.6, label="Forecasted annual total")

# Error bars from CI
annual_lower = fc_ci.iloc[:, 0].resample("YE").sum()
annual_upper = fc_ci.iloc[:, 1].resample("YE").sum()
yerr_low  = annual_fc.values - annual_lower.values
yerr_high = annual_upper.values - annual_fc.values
ax.errorbar(years_fc, annual_fc.values,
            yerr=[yerr_low, yerr_high],
            fmt="none", color="darkred", capsize=5, linewidth=1.5,
            label="90% CI")

for x, y in zip(years_fc, annual_fc.values):
    ax.text(x, y + 800, f"{y:,.0f}", ha="center", fontsize=8.5)
ax.set_title("Annual Complaint Filings: Recent History vs. SARIMA Forecast")
ax.set_ylabel("Annual Filings")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=9)
ax.annotate(CAPTION, xy=(0, -0.14), xycoords="axes fraction",
            fontsize=8, color="gray")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig19_annual_forecast.png", bbox_inches="tight")
plt.close()
print("Saved: fig19_annual_forecast.png")

print(f"\n── Forecast Summary (2024–2027) ─────────────────────")
for yr in [2024, 2025, 2026, 2027]:
    subset = fc_mean[fc_mean.index.year == yr]
    lo     = fc_ci.iloc[:, 0][fc_ci.index.year == yr]
    hi     = fc_ci.iloc[:, 1][fc_ci.index.year == yr]
    if len(subset) > 0:
        print(f"  {yr}: {subset.sum():,.0f} filings  "
              f"(90% CI: {lo.sum():,.0f} – {hi.sum():,.0f})")

print("\n✓ Forecasting complete.")
