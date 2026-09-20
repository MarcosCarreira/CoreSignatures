"""
curves_load.py -- readers for the two term structures of Section 5.

  DI1  : Brazilian one-day interbank deposit futures (B3), daily settlement
         rates. Source file: 2023/df_all_DI1.csv (columns Date, MATURITY in
         years, PX_SETTLE in % p.a., FUT_BUS_DAYS_VAL, VOLUME,
         RT_OPEN_INTEREST, CONTRACT, ...). B3 publishes settlement prices;
         only the derived series (date, tau, rate) is redistributed.
  H15  : US Treasury constant-maturity yields, Federal Reserve H.15 release.
         Source file: 2023/Rates/FRB_H15.csv (plain-CSV export of the
         downloaded FRB_H15.xlsx: one row per date, one column per tenor code;
         tenors 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y; % p.a.).
         The .xlsx is read only if the .csv is absent (needs openpyxl).

Both loaders return a tidy DataFrame with columns [date, tau, rate], tau in
years, rate in percent, one row per market point, sorted by date then tau.
`curve_on(df, date)` returns the (tau, rate) arrays of one date.

No arguments. Run the file to print a summary of both files (a few seconds).
Defaults to the two dated snapshots in python/data/ (the dates used in the paper).
Explicit path arguments may still be used with the original full source files.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))          # ~/CoreSignaturesImproved
DI1_CSV = os.path.join(_HERE, "data", "di1_source_snapshot.csv")
H15_CSV = os.path.join(_HERE, "data", "h15_source_snapshot.csv")
H15_XLSX = os.path.join(_ROOT, "2023", "Rates", "FRB_H15.xlsx")   # fallback only
DERIVED_DIR = os.path.join(_HERE, "data")                          # derived series for the repo

H15_TENORS = {  # H.15 column code -> tau in years
    "RIFLGFCM01_N.B": 1 / 12, "RIFLGFCM03_N.B": 0.25, "RIFLGFCM06_N.B": 0.5,
    "RIFLGFCY01_N.B": 1.0, "RIFLGFCY02_N.B": 2.0, "RIFLGFCY03_N.B": 3.0,
    "RIFLGFCY05_N.B": 5.0, "RIFLGFCY07_N.B": 7.0, "RIFLGFCY10_N.B": 10.0,
    "RIFLGFCY20_N.B": 20.0, "RIFLGFCY30_N.B": 30.0,
}
DI1_MIN_TAU = 0.0          # keep every listed contract; set e.g. 1/12 to drop the front month
DI1_MAX_TAU = 15.0         # contracts beyond this are illiquid and sparse


def load_di1(path: str = DI1_CSV) -> pd.DataFrame:
    """DI1 settlement rates as [date, tau, rate]; one row per contract per day."""
    df = pd.read_csv(path, usecols=["Date", "MATURITY", "PX_SETTLE"],
                     parse_dates=["Date"])
    df = df.rename(columns={"Date": "date", "MATURITY": "tau", "PX_SETTLE": "rate"})
    df = df.dropna(subset=["tau", "rate"])
    df = df[(df["tau"] >= DI1_MIN_TAU) & (df["tau"] <= DI1_MAX_TAU) & (df["rate"] > 0)]
    df = df.sort_values(["date", "tau"]).drop_duplicates(["date", "tau"])
    return df.reset_index(drop=True)


def load_h15(path: str = H15_CSV) -> pd.DataFrame:
    """H.15 constant-maturity yields as [date, tau, rate]; one row per tenor per day."""
    if os.path.exists(path):
        raw = pd.read_csv(path)                        # plain CSV export (no openpyxl needed)
    else:
        raw = pd.read_excel(H15_XLSX, header=5)        # row 6 holds 'Time Period' + codes
    raw = raw.rename(columns={"Time Period": "date"})
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"])
    long = raw.melt(id_vars="date", var_name="code", value_name="rate")
    long["tau"] = long["code"].map(H15_TENORS)
    long = long.dropna(subset=["tau"])
    long["rate"] = pd.to_numeric(long["rate"], errors="coerce")
    long = long.dropna(subset=["rate"])
    return long[["date", "tau", "rate"]].sort_values(["date", "tau"]).reset_index(drop=True)


def curve_on(df: pd.DataFrame, date: str) -> tuple[np.ndarray, np.ndarray]:
    """(tau, rate) arrays of one date; raises if the date is absent."""
    d = pd.Timestamp(date)
    c = df[df["date"] == d].sort_values("tau")
    if c.empty:
        near = df["date"].iloc[(df["date"] - d).abs().argsort().iloc[0]]
        raise KeyError(f"{date} not in file; nearest available date is {near.date()}")
    return c["tau"].to_numpy(float), c["rate"].to_numpy(float)


def dates_with_at_least(df: pd.DataFrame, n_points: int) -> pd.DatetimeIndex:
    """Dates carrying at least n_points market points (for the time-series script)."""
    counts = df.groupby("date").size()
    return pd.DatetimeIndex(counts[counts >= n_points].index)


def write_derived(df: pd.DataFrame, name: str) -> str:
    """Write the derived [date, tau, rate] series to data/<name>.csv (repo-shippable)."""
    os.makedirs(DERIVED_DIR, exist_ok=True)
    out = os.path.join(DERIVED_DIR, f"{name}.csv")
    df.to_csv(out, index=False, float_format="%.6f")
    return out


if __name__ == "__main__":
    t0 = time.time()
    for name, loader in (("DI1", load_di1), ("H15", load_h15)):
        df = loader()
        per_day = df.groupby("date").size()
        print(f"{name}: {len(df):,} points, {per_day.size:,} dates, "
              f"{df['date'].min().date()} to {df['date'].max().date()}, "
              f"{per_day.min()}-{per_day.max()} points/day, tau {df['tau'].min():.3f}-{df['tau'].max():.1f}y")
        out = write_derived(df, f"{name.lower()}_curve_points")
        print(f"      derived series written to {os.path.relpath(out, _HERE)}")
    print(f"done in {time.time() - t0:.1f}s")
