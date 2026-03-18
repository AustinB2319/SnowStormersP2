"""
03_trend_analysis.py
Snow Stormers – DS 4002, Spring 2026

Time-series trend analysis of federal inmate complaint filings:
  • Rolling averages (3-month and 12-month)
  • STL decomposition (trend / seasonal / residual)
  • Z-score spike detection
  • Mann-Kendall monotonic trend test
  • Augmented Dickey-Fuller stationarity test
  • Period-over-period change analysis

Figures saved to figures/; statistical results printed to stdout.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
import pymannkendall as mk

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FIG_DIR  = os.path.join(os.path.dirname(__file__), "figures")

ACCENT   = "#1f77b4"
RED      = "#d62728"
GREEN    = "#2ca02c"
CAPTION  = "Source: BOP SENTRY database via Data Liberation Project (FOIA, 2024)"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
})

# ── Load monthly counts ──────────────────────────────────────────────────────
monthly = pd.read_csv(f"{DATA_DIR}/monthly_complaints.csv", parse_dates=["year_month"])
monthly = monthly.set_index("year_month").sort_index()
ts = monthly["complaints"]

# ═══════════════════════════════════════════════════════════════════════════
# Figure 8 – Rolling averages
# ═══════════════════════════════════════════════════════════════════════════
roll3  = ts.rolling(3,  center=True).mean()
roll12 = ts.rolling(12, center=True).mean()

fig, ax = plt.subplots(figsize=(13, 4.8))
ax.plot(ts.index, ts.values, color="lightsteelblue", linewidth=0.8,
        label="Monthly count", alpha=0.9)
ax.plot(roll3.index,  roll3.values,  color="#ff7f0e", linewidth=1.5,
        label="3-month rolling avg")
ax.plot(roll12.index, roll12.values, color=RED,       linewidth=2,
        label="12-month rolling avg")
ax.set_title("Monthly Complaints with Rolling Averages (2000–2023)")
ax.set_ylabel("Number of Filings")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=9, framealpha=0.8)
ax.annotate(CAPTION, xy=(0, -0.13), xycoords="axes fraction",
            fontsize=8, color="gray")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig8_rolling_averages.png", bbox_inches="tight")
plt.close()
print("Saved: fig8_rolling_averages.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 9 – STL decomposition
# ═══════════════════════════════════════════════════════════════════════════
stl    = STL(ts, period=12, robust=True)
result = stl.fit()

fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
components = [
    (ts.values,           "Observed",  ACCENT),
    (result.trend,        "Trend",     RED),
    (result.seasonal,     "Seasonal",  GREEN),
    (result.resid,        "Residual",  "gray"),
]
for ax, (y, title, color) in zip(axes, components):
    ax.plot(ts.index, y, color=color, linewidth=1)
    if title == "Residual":
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_ylabel(title, fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.suptitle("STL Decomposition of Monthly Complaint Filings (2000–2023)",
             fontsize=13, fontweight="bold", y=1.01)
axes[-1].annotate(CAPTION, xy=(0, -0.25), xycoords="axes fraction",
                  fontsize=8, color="gray")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig9_stl_decomposition.png", bbox_inches="tight")
plt.close()
print("Saved: fig9_stl_decomposition.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 10 – Spike detection (Z-score on 12-month rolling mean)
# ═══════════════════════════════════════════════════════════════════════════
ZSCORE_THRESH = 2.0

# Use deseasonalized series (trend + residual) for spike detection
deseason = pd.Series(result.trend + result.resid, index=ts.index)
z_scores  = pd.Series(stats.zscore(deseason.dropna()), index=deseason.dropna().index)
spikes    = z_scores[z_scores.abs() > ZSCORE_THRESH]

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

# Top panel: raw series with spikes highlighted
ax = axes[0]
ax.fill_between(ts.index, ts.values, alpha=0.15, color=ACCENT)
ax.plot(ts.index, ts.values, color=ACCENT, linewidth=1, label="Monthly filings")
ax.plot(roll12.index, roll12.values, color=RED, linewidth=2,
        label="12-month rolling avg")

pos_spikes = spikes[spikes > 0]
neg_spikes = spikes[spikes < 0]
ax.scatter(pos_spikes.index, ts.loc[pos_spikes.index], color=RED,   zorder=5,
           s=50, label=f"High spike (|z|>{ZSCORE_THRESH})", marker="^")
ax.scatter(neg_spikes.index, ts.loc[neg_spikes.index], color=GREEN, zorder=5,
           s=50, label=f"Low spike (|z|>{ZSCORE_THRESH})",  marker="v")

ax.set_title("Monthly Complaint Filings – Spike Detection (|Z| > 2)")
ax.set_ylabel("Number of Filings")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=9, framealpha=0.8)

# Bottom panel: Z-scores
ax = axes[1]
ax.bar(z_scores.index, z_scores.values, width=20,
       color=[RED if z > ZSCORE_THRESH else (GREEN if z < -ZSCORE_THRESH else "lightsteelblue")
              for z in z_scores.values], alpha=0.8)
ax.axhline( ZSCORE_THRESH, color=RED,   linewidth=1.2, linestyle="--",
            label=f"±{ZSCORE_THRESH} threshold")
ax.axhline(-ZSCORE_THRESH, color=GREEN, linewidth=1.2, linestyle="--")
ax.axhline(0, color="black", linewidth=0.7)
ax.set_ylabel("Z-Score")
ax.set_xlabel("")
ax.legend(fontsize=9)
axes[1].annotate(CAPTION, xy=(0, -0.18), xycoords="axes fraction",
                 fontsize=8, color="gray")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig10_spike_detection.png", bbox_inches="tight")
plt.close()
print("Saved: fig10_spike_detection.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 11 – Period comparison: 5-year average complaint counts
# ═══════════════════════════════════════════════════════════════════════════
periods = {
    "2000–2004": (2000, 2004),
    "2005–2009": (2005, 2009),
    "2010–2014": (2010, 2014),
    "2015–2019": (2015, 2019),
    "2020–2023": (2020, 2023),
}
annual_df = monthly.reset_index()
annual_df["year"] = annual_df["year_month"].dt.year

period_means = {}
for label, (y0, y1) in periods.items():
    subset = annual_df[(annual_df["year"] >= y0) & (annual_df["year"] <= y1)]
    period_means[label] = subset["complaints"].mean()

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(period_means.keys(), period_means.values(),
       color=sns.color_palette("Blues_d", len(period_means)),
       edgecolor="white")
ax.set_title("Average Monthly Complaint Count by 5-Year Period")
ax.set_ylabel("Avg Monthly Filings")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
for i, (label, val) in enumerate(period_means.items()):
    ax.text(i, val + 30, f"{val:,.0f}", ha="center", va="bottom", fontsize=9)
ax.annotate(CAPTION, xy=(0, -0.14), xycoords="axes fraction",
            fontsize=8, color="gray")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig11_period_comparison.png", bbox_inches="tight")
plt.close()
print("Saved: fig11_period_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════
# Statistical Tests – printed to console
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("STATISTICAL ANALYSIS RESULTS")
print("═" * 60)

# 1. Mann-Kendall trend test (annual totals)
annual = annual_df.groupby("year")["complaints"].sum()
mk_result = mk.original_test(annual.values)
print(f"\n1. Mann-Kendall Trend Test (annual totals)")
print(f"   Trend:     {mk_result.trend}")
print(f"   p-value:   {mk_result.p:.6f}")
print(f"   Tau:       {mk_result.Tau:.4f}")
print(f"   Sen slope: {mk_result.slope:.2f} complaints/year")

# Also run on monthly series
mk_monthly = mk.original_test(ts.values)
print(f"\n   Mann-Kendall on monthly series:")
print(f"   Trend:     {mk_monthly.trend}")
print(f"   p-value:   {mk_monthly.p:.6f}")
print(f"   Tau:       {mk_monthly.Tau:.4f}")

# 2. Augmented Dickey-Fuller test
print(f"\n2. Augmented Dickey-Fuller Test (stationarity)")
adf_stat, adf_p, adf_lags, adf_obs, adf_crit, _ = adfuller(ts, autolag="AIC")
print(f"   ADF Statistic: {adf_stat:.4f}")
print(f"   p-value:       {adf_p:.6f}")
print(f"   Critical values: {adf_crit}")
if adf_p < 0.05:
    print("   → Series is STATIONARY (reject H0 of unit root)")
else:
    print("   → Series is NON-STATIONARY (fail to reject H0)")

# 3. Spike summary
print(f"\n3. Spike Detection Summary (|Z| > {ZSCORE_THRESH})")
print(f"   Total spikes detected: {len(spikes)}")
print(f"   High-complaint spikes: {len(pos_spikes)}")
print(f"   Low-complaint spikes:  {len(neg_spikes)}")
if len(spikes) > 0:
    print(f"\n   High-complaint spike months:")
    for dt, z in spikes[spikes > 0].sort_values(ascending=False).head(10).items():
        raw = ts.loc[dt]
        print(f"   {dt.strftime('%Y-%m')}  z={z:.2f}  filings={raw:,.0f}")
    print(f"\n   Low-complaint spike months:")
    for dt, z in spikes[spikes < 0].sort_values().head(5).items():
        raw = ts.loc[dt]
        print(f"   {dt.strftime('%Y-%m')}  z={z:.2f}  filings={raw:,.0f}")

# 4. Period growth summary
print(f"\n4. Period-over-Period Growth")
annual_vals = annual.values
for i in range(1, len(annual)):
    yr   = annual.index[i]
    prev = annual.values[i-1]
    curr = annual.values[i]
    pct  = (curr - prev) / prev * 100
    if abs(pct) >= 15:
        print(f"   {yr}: {pct:+.1f}%  ({prev:,.0f} → {curr:,.0f})")

print("\n✓ Trend analysis complete.")
