"""
Cocoa Weather — Daily Update
==============================
Refreshes current calendar year data in all parquet files.
Run daily via Task Scheduler (daily_push.bat).
"""

import requests
import pandas as pd
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from pathlib import Path

PARQUET_DIR  = Path(__file__).parent / "data"
API_URL      = "https://api.weatherdesk.xweather.com/2e621a7f-2b1e-4f3e-af6a-5a986a68b398/services/gwi/v1/timeseries"
MAX_WORKERS  = 20
CURRENT_YEAR = str(datetime.date.today().year)

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


def _fetch_station(station: str, parameter: str) -> list:
    params = {
        "station": station, "parameter": parameter,
        "start": "01-01", "end": "12-31", "model": "0", "metric": "1",
    }
    r = requests.get(API_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("output", {})

    records = []
    if CURRENT_YEAR in data:
        for d in data[CURRENT_YEAR]:
            rec = {"station": station, "year": CURRENT_YEAR, "date": d["date"]}
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


def _update_origin(origin_name: str, cfg: dict):
    parquet_path   = PARQUET_DIR / cfg["file"]
    station_region = cfg["stations"]
    stations       = list(station_region.keys())

    if not parquet_path.exists():
        print(f"  Parquet not found. Run backfill.py first.")
        return

    buckets = {"PRCP": [], "TAVG": [], "TMIN": [], "TMAX": []}
    errors  = []
    tasks   = [(s, p) for s in stations for p in buckets]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_fetch_station, s, p): (s, p) for s, p in tasks}
        for fut in as_completed(futures):
            stn, param = futures[fut]
            try:
                buckets[param].extend(fut.result())
            except Exception as e:
                errors.append(f"{stn}/{param}: {e}")

    if errors:
        print(f"  {len(errors)} error(s): {errors[:3]}")

    frames = {p: pd.DataFrame(rows) for p, rows in buckets.items() if rows}
    if not frames:
        print(f"  No data returned for {CURRENT_YEAR}.")
        return

    new_df = reduce(lambda l, r: l.merge(r, on=["station", "year", "date"], how="outer"),
                    frames.values())
    for col in ["prcp", "prcp_sum", "tavg", "tmin", "tmax"]:
        if col not in new_df.columns:
            new_df[col] = pd.NA

    new_df["region"] = new_df["station"].map(station_region)
    new_df = new_df[["station", "region", "year", "date", "prcp", "prcp_sum", "tavg", "tmin", "tmax"]]

    existing = pd.read_parquet(parquet_path)
    existing = existing[existing["year"] != CURRENT_YEAR]
    updated  = pd.concat([existing, new_df], ignore_index=True)
    updated.to_parquet(parquet_path, index=False)
    print(f"  {len(new_df):,} rows updated for {CURRENT_YEAR} -> {cfg['file']}")


def main():
    today = datetime.date.today()
    print(f"Daily update -- {today}  (refreshing year {CURRENT_YEAR})\n")
    for origin_name, cfg in ORIGINS.items():
        print(f"[{origin_name}]")
        _update_origin(origin_name, cfg)
    print("\nDone.")


if __name__ == "__main__":
    main()
