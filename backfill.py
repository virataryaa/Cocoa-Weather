"""
Cocoa Weather — One-Time Backfill
===================================
Fetches 2016-2026 + normals for IVC, Ghana, Ecuador.
Saves one parquet file per origin in ./data/

Run once:
    python backfill.py

After this, use daily_update.py for incremental refreshes.
"""

import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from pathlib import Path

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
PARQUET_DIR = Path(__file__).parent / "data"
API_URL     = "https://api.weatherdesk.xweather.com/2e621a7f-2b1e-4f3e-af6a-5a986a68b398/services/gwi/v1/timeseries"
MAX_WORKERS = 20

FETCH_YEARS = [
    "2016", "2017", "2018", "2019", "2020",
    "2021", "2022", "2023", "2024", "2025", "2026",
    "normals",
]

# -------------------------------------------------------
# ORIGINS
# -------------------------------------------------------
ORIGINS = {
    "IVC": {
        "file": "ivc.parquet",
        "stations": {
            "65557": "IVC", "65560": "IVC", "65562": "IVC", "65563": "IVC",
            "65585": "IVC", "65594": "IVC", "65599": "IVC",
        },
    },
    "Ghana": {
        "file": "ghana.parquet",
        "stations": {
            "65432": "Ghana", "65439": "Ghana", "65442": "Ghana",
            "65445": "Ghana", "65459": "Ghana", "65467": "Ghana",
        },
    },
    "Ecuador": {
        "file": "ecuador.parquet",
        "stations": {
            "84050": "Ecuador", "84105": "Ecuador",
            "84135": "Ecuador", "84140": "Ecuador",
        },
    },
}

# -------------------------------------------------------
# FETCH
# -------------------------------------------------------
def _fetch_station(station: str, parameter: str) -> list:
    params = {
        "station": station, "parameter": parameter,
        "start": "01-01", "end": "12-31", "model": "0", "metric": "1",
    }
    r = requests.get(API_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("output", {})

    records = []
    for api_year in FETCH_YEARS:
        if api_year not in data:
            continue
        label = "Normal (Maxar)" if api_year == "normals" else api_year
        for d in data[api_year]:
            rec = {"station": station, "year": label, "date": d["date"]}
            if parameter == "PRCP":
                rec["prcp"]     = d.get("prcp")
                rec["prcp_sum"] = d.get("prcp_sum")
            elif parameter == "TAVG":
                rec["tavg"] = d.get("tavg")
            elif parameter == "TMIN":
                rec["tmin"] = d.get("tmin")
            else:
                rec["tmax"] = d.get("tmax")
            records.append(rec)
    return records


def _fetch_origin(origin_name: str, cfg: dict) -> pd.DataFrame:
    station_region = cfg["stations"]
    stations       = list(station_region.keys())
    buckets        = {"PRCP": [], "TAVG": [], "TMIN": [], "TMAX": []}
    errors         = []

    tasks = [(s, p) for s in stations for p in buckets]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_fetch_station, s, p): (s, p) for s, p in tasks}
        for fut in as_completed(futures):
            stn, param = futures[fut]
            try:
                buckets[param].extend(fut.result())
            except Exception as e:
                errors.append(f"{stn}/{param}: {e}")

    if errors:
        print(f"  {len(errors)} error(s) (first 3): {errors[:3]}")

    frames = {p: pd.DataFrame(rows) for p, rows in buckets.items() if rows}
    if not frames:
        return pd.DataFrame()

    df = reduce(lambda l, r: l.merge(r, on=["station", "year", "date"], how="outer"),
                frames.values())
    for col in ["prcp", "prcp_sum", "tavg", "tmin", "tmax"]:
        if col not in df.columns:
            df[col] = pd.NA
    df["region"] = df["station"].map(station_region)
    return df[["station", "region", "year", "date", "prcp", "prcp_sum", "tavg", "tmin", "tmax"]]


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {PARQUET_DIR}\n")

    for origin_name, cfg in ORIGINS.items():
        n = len(cfg["stations"])
        print(f"[{origin_name}]  {n} stations x 4 params x {len(FETCH_YEARS)} years ...")
        df = _fetch_origin(origin_name, cfg)
        if df.empty:
            print(f"  No data returned -- skipping.\n")
            continue
        out = PARQUET_DIR / cfg["file"]
        df.to_parquet(out, index=False)
        print(f"  {len(df):,} rows saved -> {cfg['file']}")
        print(f"  Years: {sorted(df['year'].unique())}\n")

    print("Backfill complete.")


if __name__ == "__main__":
    main()
