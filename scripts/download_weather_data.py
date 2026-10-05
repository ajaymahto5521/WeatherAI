"""Download 2 years of hourly New Delhi weather from Open-Meteo (no API key needed).

Writes data/raw/delhi_weather_hourly.csv. Run: python -m scripts.download_weather_data
"""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

LATITUDE = 28.6139
LONGITUDE = 77.2090
LOCATION_NAME = "New Delhi, India"

END_DATE = date.today() - timedelta(days=1)
START_DATE = END_DATE - timedelta(days=365 * 2)

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "windspeed_10m",
    "precipitation",
    "cloudcover",
    "pressure_msl",
    "shortwave_radiation",
    "et0_fao_evapotranspiration",
]

OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "delhi_weather_hourly.csv"

API_URL = "https://archive-api.open-meteo.com/v1/archive"


def download() -> pd.DataFrame:
    print(f"Fetching hourly data for {LOCATION_NAME}, {START_DATE} -> {END_DATE}")

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "Asia/Kolkata",  # timestamps in local time
    }
    response = requests.get(API_URL, params=params, timeout=60)
    response.raise_for_status()

    df = pd.DataFrame(response.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df.index.name = "time"

    print(f"Downloaded {len(df):,} rows x {len(df.columns)} columns "
          f"({df.index[0]} -> {df.index[-1]})")
    return df


def validate(df: pd.DataFrame) -> None:
    """Fail loudly rather than train on bad data."""
    assert len(df) >= 15_000, f"Expected >=15,000 rows, got {len(df)}"
    print(f"  [OK] row count: {len(df):,}")

    for col in HOURLY_VARIABLES:
        assert col in df.columns, f"Missing column: {col}"
    print(f"  [OK] all {len(HOURLY_VARIABLES)} variables present")

    temp = df["temperature_2m"].dropna()
    assert temp.min() > -5, f"Temperature too low: {temp.min():.1f}C"
    assert temp.max() < 55, f"Temperature too high: {temp.max():.1f}C"
    print(f"  [OK] temperature range: {temp.min():.1f}C -> {temp.max():.1f}C")

    for col, pct in (df.isnull().mean() * 100).items():
        if pct > 5:
            print(f"  [WARN] {col} has {pct:.1f}% missing values")
        else:
            print(f"  [OK] {col}: {pct:.1f}% missing")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = download()
    validate(df)
    print(df.describe().round(2).to_string())

    df.to_csv(OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
