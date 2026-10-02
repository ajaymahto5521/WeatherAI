"""
scripts/download_weather_data.py
=================================
Downloads historical hourly weather data from the Open-Meteo Archive API.

Open-Meteo: https://open-meteo.com
  - Free, no API key required
  - Historical data back to 1940
  - Hourly resolution

Location: New Delhi, India (28.6139°N, 77.2090°E)
Period:   Last 2 years of hourly data

Output:
  data/raw/delhi_weather_hourly.csv

Usage:
    python -m scripts.download_weather_data

What each column means:
  time                 — ISO 8601 datetime (UTC+5:30 local, stored as UTC)
  temperature_2m       — Air temperature at 2m height in °C  [TARGET]
  relative_humidity_2m — Relative humidity at 2m in %
  windspeed_10m        — Wind speed at 10m in km/h
  precipitation        — Rainfall in the past hour in mm
  cloudcover           — Total cloud cover in %
  pressure_msl         — Atmospheric pressure at mean sea level in hPa
  shortwave_radiation  — Solar radiation in W/m²
  et0_fao_evapotranspiration — Reference evapotranspiration (mm) — proxy for dryness
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import requests
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────

# New Delhi coordinates
LATITUDE  = 28.6139
LONGITUDE = 77.2090
LOCATION_NAME = "New Delhi, India"

# Fetch 2 years of data ending yesterday
END_DATE   = date.today() - timedelta(days=1)
START_DATE = END_DATE - timedelta(days=365 * 2)

# Variables to fetch — each will become a column in our CSV
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

# Output path
OUTPUT_DIR  = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "delhi_weather_hourly.csv"

# Open-Meteo Archive API endpoint
API_URL = "https://archive-api.open-meteo.com/v1/archive"


# ── Main ───────────────────────────────────────────────────────────────────

def download() -> pd.DataFrame:
    """
    Calls the Open-Meteo Archive API and returns a DataFrame.

    The API returns JSON like:
    {
      "hourly": {
        "time": ["2023-01-01T00:00", ...],
        "temperature_2m": [18.5, 17.9, ...],
        ...
      }
    }
    We convert this to a tidy DataFrame where each row is one hour.
    """
    print(f"Fetching hourly weather data for {LOCATION_NAME}")
    print(f"Period: {START_DATE} → {END_DATE}")
    print(f"Variables: {', '.join(HOURLY_VARIABLES)}")
    print()

    params = {
        "latitude":  LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE.isoformat(),
        "end_date":   END_DATE.isoformat(),
        "hourly":     ",".join(HOURLY_VARIABLES),
        "timezone":   "Asia/Kolkata",   # IST — so timestamps are in local time
    }

    print("Calling Open-Meteo API...")
    response = requests.get(API_URL, params=params, timeout=60)

    # Raise an exception if the API returned an error status
    # This is better than silently getting back empty data
    response.raise_for_status()

    data = response.json()

    # The API puts all variables inside a "hourly" key
    hourly = data["hourly"]

    # Build DataFrame — each key in hourly becomes a column
    df = pd.DataFrame(hourly)

    # "time" comes back as a string like "2023-01-01T00:00"
    # Convert it to a proper datetime so we can do time-based operations
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df.index.name = "time"

    print(f"Downloaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"Date range: {df.index[0]} → {df.index[-1]}")
    print()

    return df


def validate(df: pd.DataFrame) -> None:
    """
    Basic sanity checks on the downloaded data.
    Fail loudly if something looks wrong — don't silently use bad data.
    """
    print("── Validation ─────────────────────────────────────────────")

    # Check we have a reasonable number of rows
    # 2 years * 365 days * 24 hours = ~17,520 rows
    assert len(df) >= 15_000, f"Expected ≥15,000 rows, got {len(df)}"
    print(f"  ✓ Row count: {len(df):,}")

    # Check all expected columns exist
    for col in HOURLY_VARIABLES:
        assert col in df.columns, f"Missing column: {col}"
    print(f"  ✓ All {len(HOURLY_VARIABLES)} variables present")

    # Check temperature is in a plausible range for New Delhi
    temp = df["temperature_2m"].dropna()
    assert temp.min() > -5,  f"Temperature too low: {temp.min():.1f}°C"
    assert temp.max() < 55,  f"Temperature too high: {temp.max():.1f}°C"
    print(f"  ✓ Temperature range: {temp.min():.1f}°C → {temp.max():.1f}°C")

    # Check missing value percentage
    missing_pct = df.isnull().mean() * 100
    for col, pct in missing_pct.items():
        if pct > 5:
            print(f"  ⚠ Warning: {col} has {pct:.1f}% missing values")
        else:
            print(f"  ✓ {col}: {pct:.1f}% missing")

    print()


def summarize(df: pd.DataFrame) -> None:
    """Print a concise summary so you can see what you're working with."""
    print("── Data Summary ────────────────────────────────────────────")
    print(df.describe().round(2).to_string())
    print()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = download()
    validate(df)
    summarize(df)

    df.to_csv(OUTPUT_FILE)
    print(f"✓ Saved to {OUTPUT_FILE}  ({OUTPUT_FILE.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
