# Federal Inmate Complaint Trend Analysis
### DS 4002 — Spring 2026 | Snow Stormers

This repository contains a data science study analyzing federal inmate complaint trends over a 24-year period (2000–2023) to determine whether spikes in complaint frequency serve as measurable indicators of institutional stress in federal prisons — particularly in relation to staffing shortages.

**Research Question:** How do federal inmate complaints change over time, and can temporal spikes signal institutional stress potentially associated with staffing shortages in federal prisons?

---

## Repository Contents

This repository contains all data, scripts, and figures needed to reproduce the full analysis pipeline — from raw complaint filings through exploratory data analysis, trend detection, staffing correlation, and SARIMA forecasting. The study applies time-series decomposition, anomaly detection, correlation analysis, and predictive modeling to approximately 1.8 million Bureau of Prisons (BOP) SENTRY complaint records.

---

## Section 1: Software and Platform

### Language
- **Python 3.9**

### Required Packages
Install the following packages before running any scripts:

| Package | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, and aggregation |
| `numpy` | Numerical computations |
| `pyarrow` | Reading `.parquet` files |
| `matplotlib` | Plotting and visualization |
| `seaborn` | Statistical visualizations |
| `scipy` | Statistical tests (Pearson/Spearman correlations) |
| `statsmodels` | STL decomposition, ADF test, SARIMA/SARIMAX modeling |
| `scikit-learn` | Model evaluation metrics (MAE, RMSE) |
| `pymannkendall` | Mann-Kendall monotonic trend test |
| `tzdata` | Timezone support for time-series data |

Install all dependencies at once:
```bash
pip install pandas numpy pyarrow matplotlib seaborn scipy statsmodels scikit-learn pymannkendall tzdata
```

### Platform
- **macOS** (developed and tested on macOS with Python 3.9 via virtual environment)
- Compatible with **Windows** and **Linux** with no modification to the scripts

---

## Section 2: Map of Documentation

The following tree illustrates the full folder and file hierarchy of this project:

```
data science project 2/
│
├── README.md                          # Project orientation and reproduction guide
│
├── complaint-filings.csv              # Raw complaint data: 1,783,999 rows, 37 columns (2000–2023)
├── complaint-filings.parquet          # Compressed version of the raw data (~25 MB vs. ~201 MB)
│
├── 01_data_processing.py              # Script 1: Load raw data, clean, and produce aggregated CSVs
├── 02_eda.py                          # Script 2: Exploratory data analysis — produce figures 1–7
├── 03_trend_analysis.py               # Script 3: Time-series trend detection and anomaly identification
├── 04_staffing_analysis.py            # Script 4: Correlation of complaints with BOP staffing metrics
├── 05_forecasting.py                  # Script 5: SARIMA model — forecast complaint volumes 2024–2027
│
├── data/                              # Processed datasets (outputs from Script 1)
│   ├── monthly_complaints.csv         # Monthly total complaint counts (2000–2023)
│   ├── monthly_by_level.csv           # Monthly complaint counts by appeal level (A/R/F)
│   ├── annual_complaints.csv          # Total annual complaint counts by year (2000–2023)
│   ├── annual_by_subject.csv          # Annual complaint counts broken down by subject category
│   ├── annual_by_facility.csv         # Annual complaint counts broken down by facility
│   └── forecast_2024_2027.csv         # SARIMA monthly predictions with 90% confidence intervals
│
└── figures/                           # All output visualizations (PNG)
    ├── fig1_monthly_complaints.png    # Monthly complaint volume over time (line + fill)
    ├── fig2_annual_totals.png         # Annual complaint totals bar chart (2000–2023)
    ├── fig3_by_appeal_level.png       # Monthly complaints by appeal level (stacked area)
    ├── fig4_top_subjects.png          # Top 10 complaint categories by total filings
    ├── fig5_subject_trends.png        # Annual filings by top 5 subject categories (stacked area)
    ├── fig6_top_facilities.png        # Top 15 facilities by total complaint count
    ├── fig7_monthly_seasonality.png   # Average complaints by calendar month (with ± 1 SD)
    ├── fig8_rolling_averages.png      # 3-month and 12-month rolling averages
    ├── fig9_stl_decomposition.png     # STL decomposition: trend, seasonal, and residual components
    ├── fig10_spike_detection.png      # Z-score anomaly detection (spikes flagged at |Z| > 2.0)
    ├── fig11_period_comparison.png    # Average monthly complaints per 5-year period
    ├── fig12_complaints_vs_officers.png  # Dual-axis: complaint volume vs. officer headcount
    ├── fig13_inmate_ratio.png         # Inmate-to-officer ratio trend (2000–2023)
    ├── fig14_complaints_per_1k.png    # Population-normalized complaint rate per 1,000 inmates
    ├── fig15_ratio_vs_complaints_scatter.png  # Scatter plot: ratio vs. complaints with OLS regression
    ├── fig16_correlation_matrix.png   # Heatmap of correlations across 5 key institutional variables
    ├── fig17_forecast_test.png        # SARIMA validation performance on 2021–2023 holdout set
    ├── fig18_forecast_full.png        # Full 2000–2023 history + 2024–2027 forecast with CI bands
    └── fig19_annual_forecast.png      # Bar chart: recent annual history vs. forecasted annual totals
```

---

## Section 3: Instructions for Reproducing Results

Follow these steps in order. Each script depends on the outputs of the one before it.

### Step 1 — Set Up Your Environment

1. Ensure you have **Python 3.9** installed. Verify with:
   ```bash
   python --version
   ```
2. Navigate to the project folder in your terminal.
3. Install all required packages (see Section 1):
   ```bash
   pip install pandas numpy pyarrow matplotlib seaborn scipy statsmodels scikit-learn pymannkendall tzdata
   ```

### Step 2 — Run the Data Processing Script

Run `01_data_processing.py` first. This script reads the raw complaint data, parses dates, applies subject and appeal-level labels, and produces the cleaned, aggregated CSV files used by all subsequent scripts.

```bash
python 01_data_processing.py
```

**Input:** `complaint-filings.parquet`
**Output (written to `data/`):**
- `monthly_complaints.csv`
- `monthly_by_level.csv`
- `annual_complaints.csv`
- `annual_by_subject.csv`
- `annual_by_facility.csv`

> **Note:** If `complaint-filings.parquet` is not present, the script will fall back to reading `complaint-filings.csv`, which will take significantly longer due to file size (~201 MB).

### Step 3 — Run the Exploratory Data Analysis Script

Run `02_eda.py` to generate all exploratory visualizations from the processed data produced in Step 2.

```bash
python 02_eda.py
```

**Input:** `data/monthly_complaints.csv`, `data/monthly_by_level.csv`, `data/annual_by_subject.csv`, `data/annual_by_facility.csv`, `data/annual_complaints.csv`
**What it does:**
- Plots overall monthly complaint volume as a time-series line chart (fig1)
- Plots annual complaint totals as a bar chart (fig2)
- Produces a stacked area chart of complaints by appeal level (Administrative, Regional, Final) (fig3)
- Ranks the top 10 complaint subject categories by total filings (fig4)
- Shows annual filing trends for the top 5 subject categories as a stacked area chart (fig5)
- Ranks the top 15 facilities by total complaint count (fig6)
- Plots average complaint counts by calendar month with ± 1 standard deviation bars to reveal seasonality (fig7)

**Output:** Figures `fig1` through `fig7` saved to `figures/`. No console output beyond save confirmations.

### Step 4 — Run the Trend Analysis Script

Run `03_trend_analysis.py` to perform time-series analysis on the monthly complaint data produced in Step 2.

```bash
python 03_trend_analysis.py
```

**Input:** `data/annual_complaints.csv`, `data/monthly_by_level.csv`
**What it does:**
- Computes 3-month and 12-month rolling averages
- Applies STL decomposition to extract trend, seasonal, and residual components
- Flags anomalous months using Z-score spike detection (threshold: |Z| > 2.0)
- Runs the Mann-Kendall test for monotonic trend significance
- Runs the Augmented Dickey-Fuller (ADF) test for stationarity
- Computes period-over-period growth rates (year-over-year)

**Output:** Figures `fig8` through `fig11` saved to `figures/`, plus statistical summaries printed to the console (Mann-Kendall results, ADF p-value, spike summaries, high-growth periods ≥ 15% YoY).

### Step 5 — Run the Staffing Correlation Script

Run `04_staffing_analysis.py` to examine whether complaint trends correlate with BOP staffing levels.

```bash
python 04_staffing_analysis.py
```

**Input:** `data/annual_complaints.csv`, `data/bop_staffing.csv`
**What it does:**
- Computes normalized complaint rates (complaints per 1,000 inmates)
- Calculates inmate-to-correctional-officer ratio by year
- Tests Pearson and Spearman correlations between complaint metrics and staffing variables
- Generates dual-axis and scatter plots for visual inspection

**Output:** Figures `fig12` through `fig16` saved to `figures/`, plus correlation coefficients and a key metrics table printed to the console.

> **Note:** The staffing data (`bop_staffing.csv`) is derived from BOP Annual Reports, DOJ OIG publications, and Congressional Research Service report CRS R48826. These figures are directional approximations from published sources.

### Step 6 — Run the Forecasting Script

Run `05_forecasting.py` to train a SARIMA model on historical complaint data and generate predictions through 2027.

```bash
python 05_forecasting.py
```

**Input:** `data/monthly_by_level.csv` (or aggregate monthly series from Step 2)
**What it does:**
- Splits data into a training set (2000–2020) and a test set (2021–2023)
- Performs grid search over SARIMA orders optimized by AIC
- Fits the best model to the full 2000–2023 series
- Generates a 48-month forecast (January 2024 – December 2027) with 90% confidence intervals
- Evaluates model accuracy using MAE, RMSE, and MAPE (target: MAPE < 15%)

**Output:**
- Figures `fig17`, `fig18`, and `fig19` saved to `figures/`
- `data/forecast_2024_2027.csv` with monthly predicted values and confidence bounds
- Model performance metrics and top 5 SARIMA candidate models printed to the console

### Expected Final Outputs

After completing all five steps, your `figures/` folder should contain 19 PNG files (`fig1` through `fig19`), and your `data/` folder should contain all processed CSVs including the forecast file. The figures and console outputs together constitute the full results of the study.


### References
- [1] Congressional Research Service, Correctional Officer Staffing in Federal Prisons: Background and Issues (R48826), EveryCRSReport, Jan. 26, 2026. [Online]. Available: https://www.everycrsreport.com/reports/R48826.html#_Toc220490174
  
- [2] Data Liberation Project, Federal Inmate Complaints Dataset, 2024. [Online]. Available: https://www.data-liberation-project.org/datasets/federal-inmate-complaints/
  
- [3] Bureau of Prisons, BOP Annual Statistics, Federal Bureau of Prisons. [Online]. Available: https://www.bop.gov/about/statistics/
  
- [4] U.S. Department of Justice, Office of the Inspector General, Top Management and Performance Challenges Facing the Department of Justice, DOJ OIG, 2023. [Online]. Available: https://oig.justice.gov/reports/2023/challenges23.pdf
