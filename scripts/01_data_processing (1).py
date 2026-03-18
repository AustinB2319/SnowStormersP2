"""
01_data_processing.py
Snow Stormers – DS 4002, Spring 2026

Loads the raw federal inmate complaint filings (parquet), cleans the data,
and writes aggregated time-series CSVs to data/ for use by subsequent scripts.
"""

import os
import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Subject-code labels (BOP Administrative Remedy Program) ─────────────────
SUBJECT_LABELS = {
    10: "Admissions & Orientation",
    11: "Staff Accountability",
    12: "Visiting",
    13: "Quarters / Conditions",
    14: "Food Services",
    15: "Commissary",
    16: "Recreation",
    17: "Legal Activities",
    18: "Staff Conduct",
    19: "Transfer / Designation",
    20: "Discipline / Inst. Operations",
    21: "Work / Education Assignment",
    22: "Sentence Computation",
    23: "Community Programs",
    24: "Release / Detainer",
    25: "Mail / Telephone",
    26: "Medical",
    27: "Mental Health",
    28: "Dental",
    29: "Education Programs",
    30: "Property",
    31: "Religious Programs",
    32: "Financial Responsibility",
    33: "Special Housing / Seg.",
    34: "Other / Miscellaneous",
    35: "Correspondence",
    36: "Inst. Supplement",
}

LEVEL_LABELS = {
    "A": "Administrative (Warden)",
    "R": "Regional (Director)",
    "F": "Final (Central Office)",
}

STATUS_LABELS = {
    "REJ": "Rejected",
    "CLD": "Closed – Denied",
    "CLO": "Closed",
    "CLG": "Closed – Granted",
    "ACC": "Accepted",
}

# ── Load parquet (faster than CSV) ──────────────────────────────────────────
print("Loading complaint-filings.parquet …")
df = pq.read_table("complaint-filings.parquet").to_pandas()
print(f"  Rows loaded: {len(df):,}")

# ── Parse date ───────────────────────────────────────────────────────────────
df["date"] = pd.to_datetime(df["sitdtrcv"], errors="coerce")
df = df.dropna(subset=["date"])

# Keep 2000-01 through 2023-12 (full calendar years for clean analysis)
df = df[(df["date"] >= "2000-01-01") & (df["date"] < "2024-01-01")]

df["year_month"] = df["date"].dt.to_period("M")
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

print(f"  After date filter: {len(df):,}")
print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")

# ── Label appeal levels and statuses ─────────────────────────────────────────
df["level_label"] = df["ITERLVL"].map(LEVEL_LABELS).fillna(df["ITERLVL"])
df["status_label"] = df["CDSTATUS"].map(STATUS_LABELS).fillna(df["CDSTATUS"])

# Convert subject code to int where possible
df["subject_code"] = pd.to_numeric(df["CDSUB1PR"], errors="coerce")
df["subject_label"] = df["subject_code"].map(SUBJECT_LABELS).fillna("Unknown")

# ── 1. Overall monthly counts ─────────────────────────────────────────────────
monthly = (
    df.groupby("year_month")
    .size()
    .reset_index(name="complaints")
)
monthly["year_month"] = monthly["year_month"].dt.to_timestamp()
monthly = monthly.sort_values("year_month").reset_index(drop=True)
monthly.to_csv(f"{DATA_DIR}/monthly_complaints.csv", index=False)
print(f"\nSaved: monthly_complaints.csv  ({len(monthly)} months)")

# ── 2. Monthly counts by appeal level ────────────────────────────────────────
monthly_level = (
    df.groupby(["year_month", "level_label"])
    .size()
    .reset_index(name="complaints")
)
monthly_level["year_month"] = monthly_level["year_month"].dt.to_timestamp()
monthly_level = monthly_level.sort_values("year_month")
monthly_level.to_csv(f"{DATA_DIR}/monthly_by_level.csv", index=False)
print(f"Saved: monthly_by_level.csv")

# ── 3. Annual counts by subject ───────────────────────────────────────────────
annual_subject = (
    df.groupby(["year", "subject_label"])
    .size()
    .reset_index(name="complaints")
)
annual_subject.to_csv(f"{DATA_DIR}/annual_by_subject.csv", index=False)
print(f"Saved: annual_by_subject.csv")

# ── 4. Top facilities ─────────────────────────────────────────────────────────
top_facilities = (
    df.groupby(["year", "CDFCLEVN"])
    .size()
    .reset_index(name="complaints")
)
top_facilities.to_csv(f"{DATA_DIR}/annual_by_facility.csv", index=False)
print(f"Saved: annual_by_facility.csv")

# ── 5. Annual totals (for staffing merge) ─────────────────────────────────────
annual = (
    df.groupby("year")
    .size()
    .reset_index(name="complaints")
)
annual.to_csv(f"{DATA_DIR}/annual_complaints.csv", index=False)
print(f"Saved: annual_complaints.csv")

# ── 6. Summary statistics ─────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────")
print(f"Total filings analyzed: {len(df):,}")
print(f"Calendar years covered: {df['year'].min()} – {df['year'].max()}")
print(f"Unique facilities: {df['CDFCLEVN'].nunique():,}")
print(f"\nAppeal level breakdown:\n{df['level_label'].value_counts()}")
print(f"\nStatus breakdown:\n{df['status_label'].value_counts()}")
print(f"\nTop 10 complaint subjects:\n{df['subject_label'].value_counts().head(10)}")
print("\n✓ Data processing complete.")
