"""
02_eda.py
Snow Stormers – DS 4002, Spring 2026

Exploratory Data Analysis of federal inmate complaint filings.
Reads the processed CSVs from data/ (produced by 01_data_processing.py)
and saves seven figures to figures/.

Figures produced:
  fig1  – Monthly complaint filings: full time series overview
  fig2  – Annual complaint totals with year-over-year % change
  fig3  – Seasonal pattern: average monthly filings by month of year
  fig4  – Monthly filings by appeal level (stacked area)
  fig5  – Top complaint subjects over time (line chart, top 8)
  fig6  – Top 15 facilities by total complaint volume
  fig7  – Year × Month heatmap of complaint volume
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
import seaborn as sns

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FIG_DIR  = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

ACCENT  = "#1f77b4"
RED     = "#d62728"
GREEN   = "#2ca02c"
ORANGE  = "#ff7f0e"
CAPTION = "Source: BOP SENTRY database via Data Liberation Project (FOIA, 2024)"

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

# ── Load processed data ───────────────────────────────────────────────────────
monthly = pd.read_csv(f"{DATA_DIR}/monthly_complaints.csv", parse_dates=["year_month"])
monthly = monthly.set_index("year_month").sort_index()
ts = monthly["complaints"]

monthly_level = pd.read_csv(f"{DATA_DIR}/monthly_by_level.csv", parse_dates=["year_month"])
annual_subject = pd.read_csv(f"{DATA_DIR}/annual_by_subject.csv")
annual = pd.read_csv(f"{DATA_DIR}/annual_complaints.csv")
annual_facility = pd.read_csv(f"{DATA_DIR}/annual_by_facility.csv")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 – Full monthly time series overview
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.fill_between(ts.index, ts.values, alpha=0.18, color=ACCENT)
ax.plot(ts.index, ts.values, color=ACCENT, linewidth=1)

# Annotate overall min and max
ax.scatter([ts.idxmax()], [ts.max()], color=RED, zorder=5, s=60)
ax.annotate(f"Peak: {ts.max():,.0f}\n({ts.idxmax().strftime('%b %Y')})",
            xy=(ts.idxmax(), ts.max()),
            xytext=(20, -35), textcoords="offset points",
            fontsize=8.5, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))

ax.scatter([ts.idxmin()], [ts.min()], color=GREEN, zorder=5, s=60)
ax.annotate(f"Min: {ts.min():,.0f}\n({ts.idxmin().strftime('%b %Y')})",
            xy=(ts.idxmin(), ts.min()),
            xytext=(20, 25), textcoords="offset points",
            fontsize=8.5, color=GREEN,
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.8))

ax.set_title("Federal Inmate Complaint Filings – Monthly Overview (2000–2023)")
ax.set_ylabel("Number of Filings")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.annotate(CAPTION, xy=(0, -0.13), xycoords="axes fraction",
            fontsize=8, color="gray")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig1_monthly_overview.png", bbox_inches="tight")
plt.close()
print("Saved: fig1_monthly_overview.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 – Annual totals with YoY % change
# ═══════════════════════════════════════════════════════════════════════════
annual = annual.sort_values("year").reset_index(drop=True)
annual["yoy_pct"] = annual["complaints"].pct_change() * 100

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                          gridspec_kw={"height_ratios": [3, 1.5]})

# Top: bar chart
ax = axes[0]
bar_colors = [RED if c > annual["complaints"].mean() else ACCENT
              for c in annual["complaints"]]
ax.bar(annual["year"], annual["complaints"], color=bar_colors, alpha=0.75)
ax.axhline(annual["complaints"].mean(), color="gray", linewidth=1,
           linestyle="--", label=f"24-yr avg: {annual['complaints'].mean():,.0f}")
ax.set_ylabel("Annual Filings")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_title("Annual Federal Inmate Complaint Filings (2000–2023)")
ax.legend(fontsize=9)

# Bottom: YoY % change
ax2 = axes[1]
colors = [RED if v > 0 else GREEN for v in annual["yoy_pct"].fillna(0)]
ax2.bar(annual["year"], annual["yoy_pct"].fillna(0), color=colors, alpha=0.75)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_ylabel("YoY Change (%)")
ax2.set_xlabel("Year")
ax2.annotate(CAPTION, xy=(0, -0.25), xycoords="axes fraction",
             fontsize=8, color="gray")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig2_annual_totals.png", bbox_inches="tight")
plt.close()
print("Saved: fig2_annual_totals.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 – Seasonal pattern: average filings by month of year
# ═══════════════════════════════════════════════════════════════════════════
monthly_df = monthly.reset_index()
monthly_df["month"] = monthly_df["year_month"].dt.month
monthly_df["year"]  = monthly_df["year_month"].dt.year

month_avg = monthly_df.groupby("month")["complaints"].agg(["mean", "std"]).reset_index()
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(month_avg["month"], month_avg["mean"],
       color=sns.color_palette("Blues_d", 12),
       yerr=month_avg["std"], capsize=4, error_kw={"linewidth": 1})
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names)
ax.set_title("Average Monthly Complaint Filings by Month of Year (2000–2023)")
ax.set_ylabel("Avg Filings (±1 SD)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.axhline(month_avg["mean"].mean(), color="gray", linewidth=1,
           linestyle="--", label="Annual monthly avg")
ax.legend(fontsize=9)
ax.annotate(CAPTION, xy=(0, -0.13), xycoords="axes fraction",
            fontsize=8, color="gray")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig3_seasonal_pattern.png", bbox_inches="tight")
plt.close()
print("Saved: fig3_seasonal_pattern.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 – Monthly filings by appeal level (stacked area)
# ═══════════════════════════════════════════════════════════════════════════
level_pivot = (
    monthly_level
    .pivot_table(index="year_month", columns="level_label",
                 values="complaints", aggfunc="sum")
    .fillna(0)
    .sort_index()
)

# Reorder columns by total volume
col_order = level_pivot.sum().sort_values(ascending=False).index
level_pivot = level_pivot[col_order]

palette = [ACCENT, ORANGE, GREEN]
fig, ax = plt.subplots(figsize=(13, 5))
ax.stackplot(level_pivot.index, level_pivot.T.values,
             labels=level_pivot.columns,
             colors=palette[:len(level_pivot.columns)], alpha=0.8)
ax.set_title("Monthly Complaint Filings by Appeal Level (2000–2023)")
ax.set_ylabel("Number of Filings")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
ax.annotate(CAPTION, xy=(0, -0.12), xycoords="axes fraction",
            fontsize=8, color="gray")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig4_filings_by_level.png", bbox_inches="tight")
plt.close()
print("Saved: fig4_filings_by_level.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 – Top complaint subjects over time (line chart)
# ═══════════════════════════════════════════════════════════════════════════
# Identify top 8 subjects by total volume across all years
top_subjects = (
    annual_subject
    .groupby("subject_label")["complaints"]
    .sum()
    .sort_values(ascending=False)
    .head(8)
    .index.tolist()
)

subject_pivot = (
    annual_subject[annual_subject["subject_label"].isin(top_subjects)]
    .pivot_table(index="year", columns="subject_label",
                 values="complaints", aggfunc="sum")
    .fillna(0)
)

fig, ax = plt.subplots(figsize=(13, 6))
palette_lines = sns.color_palette("tab10", len(top_subjects))
for col, color in zip(subject_pivot.columns, palette_lines):
    ax.plot(subject_pivot.index, subject_pivot[col], linewidth=1.8,
            label=col, color=color, marker="o", markersize=3)

ax.set_title("Top 8 Complaint Subjects Over Time (Annual Counts, 2000–2023)")
ax.set_xlabel("Year")
ax.set_ylabel("Annual Filings")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=8, framealpha=0.85, bbox_to_anchor=(1.01, 1), loc="upper left")
ax.annotate(CAPTION, xy=(0, -0.12), xycoords="axes fraction",
            fontsize=8, color="gray")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig5_subject_trends.png", bbox_inches="tight")
plt.close()
print("Saved: fig5_subject_trends.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 6 – Top 15 facilities by total complaint volume
# ═══════════════════════════════════════════════════════════════════════════
facility_totals = (
    annual_facility
    .groupby("CDFCLEVN")["complaints"]
    .sum()
    .sort_values(ascending=False)
    .head(15)
    .reset_index()
)
facility_totals.columns = ["facility", "complaints"]

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(facility_totals["facility"][::-1],
               facility_totals["complaints"][::-1],
               color=sns.color_palette("Blues_d", 15))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
for bar, val in zip(bars, facility_totals["complaints"][::-1]):
    ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}", va="center", fontsize=8)
ax.set_title("Top 15 Federal Facilities by Total Complaint Filings (2000–2023)")
ax.set_xlabel("Total Filings")
ax.annotate(CAPTION, xy=(0, -0.11), xycoords="axes fraction",
            fontsize=8, color="gray")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig6_top_facilities.png", bbox_inches="tight")
plt.close()
print("Saved: fig6_top_facilities.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 7 – Year × Month heatmap of complaint volume
# ═══════════════════════════════════════════════════════════════════════════
heatmap_data = (
    monthly_df
    .pivot_table(index="year", columns="month", values="complaints", aggfunc="sum")
)
heatmap_data.columns = month_names

fig, ax = plt.subplots(figsize=(13, 8))
sns.heatmap(
    heatmap_data,
    cmap="YlOrRd",
    linewidths=0.3,
    linecolor="white",
    annot=True,
    fmt=".0f",
    annot_kws={"size": 7},
    ax=ax,
    cbar_kws={"label": "Monthly Filings", "shrink": 0.6},
)
ax.set_title("Complaint Filing Volume: Year × Month Heatmap (2000–2023)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Month")
ax.set_ylabel("Year")
ax.tick_params(axis="x", labelsize=9)
ax.tick_params(axis="y", labelsize=9, rotation=0)
ax.annotate(CAPTION, xy=(0, -0.07), xycoords="axes fraction",
            fontsize=8, color="gray")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig7_year_month_heatmap.png", bbox_inches="tight")
plt.close()
print("Saved: fig7_year_month_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════
# Summary statistics printed to console
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("EDA SUMMARY STATISTICS")
print("═" * 60)
print(f"\nOverall monthly complaints:")
print(f"  Mean:   {ts.mean():,.0f}")
print(f"  Median: {ts.median():,.0f}")
print(f"  Std:    {ts.std():,.0f}")
print(f"  Min:    {ts.min():,.0f}  ({ts.idxmin().strftime('%Y-%m')})")
print(f"  Max:    {ts.max():,.0f}  ({ts.idxmax().strftime('%Y-%m')})")

print(f"\nAnnual totals:")
annual_sorted = annual.sort_values("complaints", ascending=False)
print(f"  Highest year: {int(annual_sorted.iloc[0]['year'])} "
      f"({annual_sorted.iloc[0]['complaints']:,.0f} filings)")
print(f"  Lowest year:  {int(annual_sorted.iloc[-1]['year'])} "
      f"({annual_sorted.iloc[-1]['complaints']:,.0f} filings)")

print(f"\nTop 5 complaint subjects (all years):")
top5 = (annual_subject.groupby("subject_label")["complaints"]
        .sum().sort_values(ascending=False).head(5))
for subj, cnt in top5.items():
    print(f"  {subj}: {cnt:,.0f}")

print(f"\nTop facility by volume: "
      f"{facility_totals.iloc[0]['facility']} "
      f"({facility_totals.iloc[0]['complaints']:,.0f} filings)")

print("\n✓ EDA complete.")
