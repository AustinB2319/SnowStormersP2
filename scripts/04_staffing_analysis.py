"""
04_staffing_analysis.py
Snow Stormers – DS 4002, Spring 2026

Merges annual complaint counts with BOP staffing estimates to examine
whether staffing conditions correlate with complaint volume.

Note on staffing data: BOP publishes annual statistics but does not release
a single machine-readable time-series dataset. The figures in data/bop_staffing.csv
are derived from published BOP Annual Reports, DOJ OIG reports, and the
Congressional Research Service report R48826 (January 2026). They reflect
approximate values and should be interpreted directionally.

Figures saved to figures/; correlation results printed to stdout.
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
from scipy import stats

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FIG_DIR  = os.path.join(os.path.dirname(__file__), "figures")

ACCENT  = "#1f77b4"
RED     = "#d62728"
ORANGE  = "#ff7f0e"
GREEN   = "#2ca02c"
CAPTION = ("Source: Complaint data – BOP SENTRY via Data Liberation Project (FOIA, 2024).\n"
           "Staffing data – BOP Annual Reports, DOJ OIG reports, CRS R48826 (approx.).")

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

# ── Load data ────────────────────────────────────────────────────────────────
annual      = pd.read_csv(f"{DATA_DIR}/annual_complaints.csv")
staffing    = pd.read_csv(f"{DATA_DIR}/bop_staffing.csv")

df = pd.merge(annual, staffing, on="year", how="inner")

# Derived metrics
df["complaints_per_1k_inmates"] = df["complaints"] / df["inmate_population"] * 1000
df["inmates_per_officer"]       = df["inmate_population"] / df["correctional_officers"]
df["pct_positions_vacant"]      = (
    (df["correctional_officers"].max() - df["correctional_officers"])
    / df["correctional_officers"].max() * 100
)

print(df[["year","complaints","inmate_population","correctional_officers",
          "complaints_per_1k_inmates","inmates_per_officer"]].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# Figure 12 – Dual-axis: annual complaints vs correctional officer count
# ═══════════════════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(13, 5))

color1 = ACCENT
ax1.bar(df["year"], df["complaints"], color=color1, alpha=0.55,
        label="Total complaints")
ax1.set_xlabel("Year")
ax1.set_ylabel("Annual Complaint Filings", color=color1)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax1.tick_params(axis="y", labelcolor=color1)

ax2 = ax1.twinx()
ax2.spines["top"].set_visible(False)
ax2.plot(df["year"], df["correctional_officers"], color=RED, linewidth=2.5,
         marker="o", markersize=5, label="Correctional officers")
ax2.set_ylabel("Approx. Correctional Officers", color=RED)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax2.tick_params(axis="y", labelcolor=RED)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

plt.title("Annual Complaint Filings vs. BOP Correctional Officer Count (2000–2023)")
ax1.annotate(CAPTION, xy=(0, -0.18), xycoords="axes fraction",
             fontsize=7.5, color="gray")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig12_complaints_vs_officers.png", bbox_inches="tight")
plt.close()
print("\nSaved: fig12_complaints_vs_officers.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 13 – Inmate-to-officer ratio over time
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(df["year"], df["inmates_per_officer"], color=ORANGE, linewidth=2.5,
        marker="o", markersize=5)
ax.fill_between(df["year"], df["inmates_per_officer"], alpha=0.12, color=ORANGE)
ax.axhline(df["inmates_per_officer"].mean(), color="gray", linewidth=1,
           linestyle="--", label=f"24-yr avg: {df['inmates_per_officer'].mean():.1f}")
ax.set_title("BOP Inmate-to-Correctional-Officer Ratio (2000–2023)")
ax.set_xlabel("Year")
ax.set_ylabel("Inmates per Officer")
ax.legend(fontsize=9)
ax.annotate(CAPTION, xy=(0, -0.14), xycoords="axes fraction",
            fontsize=7.5, color="gray")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig13_inmate_ratio.png", bbox_inches="tight")
plt.close()
print("Saved: fig13_inmate_ratio.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 14 – Complaints per 1,000 inmates over time
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(df["year"], df["complaints_per_1k_inmates"], color=RED, linewidth=2.5,
        marker="o", markersize=5)
ax.fill_between(df["year"], df["complaints_per_1k_inmates"], alpha=0.12, color=RED)
ax.axhline(df["complaints_per_1k_inmates"].mean(), color="gray", linewidth=1,
           linestyle="--",
           label=f"24-yr avg: {df['complaints_per_1k_inmates'].mean():.1f}")
ax.set_title("Complaint Filings per 1,000 Inmates (2000–2023, Population-Adjusted)")
ax.set_xlabel("Year")
ax.set_ylabel("Complaints per 1,000 Inmates")
ax.legend(fontsize=9)
ax.annotate(CAPTION, xy=(0, -0.14), xycoords="axes fraction",
            fontsize=7.5, color="gray")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig14_complaints_per_1k.png", bbox_inches="tight")
plt.close()
print("Saved: fig14_complaints_per_1k.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 15 – Scatter: inmate-to-officer ratio vs complaints per 1k inmates
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5.5))
sc = ax.scatter(df["inmates_per_officer"], df["complaints_per_1k_inmates"],
                c=df["year"], cmap="plasma", s=80, zorder=3, edgecolors="white",
                linewidths=0.5)
# Annotate each point with year
for _, row in df.iterrows():
    ax.annotate(str(int(row["year"])),
                (row["inmates_per_officer"], row["complaints_per_1k_inmates"]),
                textcoords="offset points", xytext=(5, 4), fontsize=7.5)

# Regression line
m, b, r, p, se = stats.linregress(df["inmates_per_officer"],
                                   df["complaints_per_1k_inmates"])
x_line = np.linspace(df["inmates_per_officer"].min(),
                      df["inmates_per_officer"].max(), 100)
ax.plot(x_line, m * x_line + b, color=RED, linewidth=1.5, linestyle="--",
        label=f"OLS fit  r={r:.2f}, p={p:.3f}")

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Year")
ax.set_title("Inmate-to-Officer Ratio vs.\nComplaints per 1,000 Inmates")
ax.set_xlabel("Inmates per Correctional Officer")
ax.set_ylabel("Complaints per 1,000 Inmates")
ax.legend(fontsize=9)
ax.annotate(CAPTION, xy=(0, -0.16), xycoords="axes fraction",
            fontsize=7.5, color="gray")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig15_ratio_vs_complaints_scatter.png", bbox_inches="tight")
plt.close()
print("Saved: fig15_ratio_vs_complaints_scatter.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 16 – Heatmap-style correlation matrix
# ═══════════════════════════════════════════════════════════════════════════
corr_df = df[["complaints", "complaints_per_1k_inmates", "inmates_per_officer",
              "inmate_population", "correctional_officers"]].rename(columns={
    "complaints": "Total\nComplaints",
    "complaints_per_1k_inmates": "Complaints\nper 1k Inmates",
    "inmates_per_officer": "Inmates\nper Officer",
    "inmate_population": "Inmate\nPopulation",
    "correctional_officers": "Correctional\nOfficers",
})
corr_mat = corr_df.corr()

fig, ax = plt.subplots(figsize=(7, 5.5))
mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Matrix: Complaints & Staffing Variables")
ax.annotate(CAPTION, xy=(0, -0.18), xycoords="axes fraction",
            fontsize=7.5, color="gray")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig16_correlation_matrix.png", bbox_inches="tight")
plt.close()
print("Saved: fig16_correlation_matrix.png")

# ═══════════════════════════════════════════════════════════════════════════
# Statistical correlation results
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("STAFFING CORRELATION RESULTS")
print("═" * 60)

pairs = [
    ("inmates_per_officer",       "complaints_per_1k_inmates",
     "Inmate-to-officer ratio × complaints per 1k inmates"),
    ("correctional_officers",     "complaints",
     "Officer count × total annual complaints"),
    ("inmate_population",         "complaints",
     "Inmate population × total complaints"),
    ("inmate_population",         "complaints_per_1k_inmates",
     "Inmate population × complaints per 1k inmates"),
]
for col_x, col_y, label in pairs:
    r, p = stats.pearsonr(df[col_x], df[col_y])
    rs, ps = stats.spearmanr(df[col_x], df[col_y])
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
    print(f"\n  {label}")
    print(f"    Pearson r={r:+.3f}  p={p:.4f} {sig}")
    print(f"    Spearman ρ={rs:+.3f}  p={ps:.4f}")

print("\n  Key metrics by year:")
print(df[["year","complaints","complaints_per_1k_inmates",
          "inmates_per_officer"]].to_string(index=False))
print("\n✓ Staffing analysis complete.")
