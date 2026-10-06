"""Agent tools: RAG search, live weather (OpenWeatherMap), Delhi next-hour forecast.

ToolResult.text goes to the LLM, .data to the API. Tools return failures as
text instead of raising.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import requests

from app.config import settings
from app.models.forecasting import ForecastingModel, WeatherConditions

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
DELHI_LAT, DELHI_LON = 28.6139, 77.2090
HTTP_TIMEOUT = 15


@dataclass
class ToolResult:
    text: str
    data: dict[str, Any] = field(default_factory=dict)


def _http_get_json(url: str, params: dict) -> dict:
    """All network calls go through here so tests can stub it."""
    resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_live_weather(city: str) -> ToolResult:
    if not settings.openweather_api_key:
        return ToolResult("Live weather is unavailable: OPENWEATHER_API_KEY is not configured.")
    try:
        d = _http_get_json(
            "https://api.openweathermap.org/data/2.5/weather",
            {"q": city, "appid": settings.openweather_api_key, "units": "metric"},
        )
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return ToolResult(f"City '{city}' was not found.")
        return ToolResult(f"Live weather lookup failed: {exc}")
    except requests.RequestException as exc:
        return ToolResult(f"Live weather lookup failed: {exc}")

    main, wind = d.get("main", {}), d.get("wind", {})
    desc = (d.get("weather") or [{}])[0].get("description", "n/a")
    text = (
        f"Current weather in {d.get('name', city)}: {main.get('temp')}°C "
        f"(feels like {main.get('feels_like')}°C), {desc}, humidity {main.get('humidity')}%, "
        f"wind {wind.get('speed')} m/s, pressure {main.get('pressure')} hPa."
    )
    # same phrasing/units the severity classifier was trained on
    wind_kmh = round((wind.get("speed") or 0) * 3.6)
    rain_mm = (d.get("rain") or {}).get("1h", 0)
    severity_text = (
        f"{desc.capitalize()}. Winds {wind_kmh} km/h. Rainfall {rain_mm} mm. "
        f"Temperature around {round(main.get('temp') or 0)}°C."
    )
    return ToolResult(
        text,
        {"city": d.get("name", city), "temperature": main.get("temp"), "severity_text": severity_text},
    )


HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "windspeed_10m", "precipitation",
    "cloudcover", "pressure_msl", "shortwave_radiation", "et0_fao_evapotranspiration",
]


def build_conditions(hourly: dict[str, list], i: int) -> WeatherConditions:
    """Model inputs for row i of an Open-Meteo hourly block; must match engineer_features() in training/train_forecaster.py."""
    if i < 24:
        raise ValueError("need at least 24 prior hours of history")
    t = hourly["temperature_2m"]
    rad = hourly["shortwave_radiation"]

    def v(name: str, k: int) -> float:
        x = hourly[name][k]
        if x is None:
            raise ValueError(f"missing {name} at index {k}")
        return float(x)

    temps = [v("temperature_2m", k) for k in range(i - 23, i + 1)]  # last 24h incl. now
    last3 = temps[-3:]
    return WeatherConditions(
        temperature=temps[-1],
        humidity=v("relative_humidity_2m", i),
        wind_speed=v("windspeed_10m", i),
        precipitation=v("precipitation", i),
        cloud_cover=v("cloudcover", i),
        pressure=v("pressure_msl", i),
        radiation=v("shortwave_radiation", i),
        evapotranspiration=v("et0_fao_evapotranspiration", i),
        timestamp=datetime.fromisoformat(hourly["time"][i]),
        temp_lag_1h=v("temperature_2m", i - 1),
        temp_lag_2h=v("temperature_2m", i - 2),
        temp_lag_3h=v("temperature_2m", i - 3),
        temp_lag_6h=v("temperature_2m", i - 6),
        temp_lag_24h=v("temperature_2m", i - 24),
        temp_rmean_3h=statistics.fmean(last3),
        temp_rmean_6h=statistics.fmean(temps[-6:]),
        temp_rmean_12h=statistics.fmean(temps[-12:]),
        temp_rstd_3h=statistics.stdev(last3),
        temp_trend_3h=temps[-1] - v("temperature_2m", i - 3),
        rad_rmean_3h=statistics.fmean(v("shortwave_radiation", k) for k in range(i - 2, i + 1)),
        hum_lag_1h=v("relative_humidity_2m", i - 1),
        wind_lag_1h=v("windspeed_10m", i - 1),
        rad_lag_1h=v("shortwave_radiation", i - 1),
        pres_lag_1h=v("pressure_msl", i - 1),
    )


_forecaster: ForecastingModel | None = None


def forecast_temperature(now: datetime | None = None) -> ToolResult:
    global _forecaster
    now = (now or datetime.now(IST)).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    try:
        d = _http_get_json(
            "https://api.open-meteo.com/v1/forecast",
            {
                "latitude": DELHI_LAT, "longitude": DELHI_LON,
                "hourly": ",".join(HOURLY_VARS),
                "past_days": 2, "forecast_days": 1, "timezone": "Asia/Kolkata",
            },
        )
        hourly = d["hourly"]
        times = [datetime.fromisoformat(s) for s in hourly["time"]]
        if now not in times:
            return ToolResult("Forecast unavailable: current hour missing from observation feed.")
        cond = build_conditions(hourly, times.index(now))
        if _forecaster is None:
            _forecaster = ForecastingModel()
        result = _forecaster.predict(cond)
    except FileNotFoundError as exc:
        return ToolResult(f"Forecast unavailable: {exc}")
    except (requests.RequestException, KeyError, ValueError) as exc:
        return ToolResult(f"Forecast unavailable: {exc}")

    text = (
        f"Model forecast for {result.location}: {result.predicted_temperature}°C at "
        f"{result.prediction_datetime} (currently {cond.temperature}°C). {result.confidence_note}"
    )
    return ToolResult(
        text,
        {
            "predicted_temperature": result.predicted_temperature,
            "prediction_date": result.prediction_datetime,
            "location": result.location,
            "model_name": result.model_type,
        },
    )


def make_rag_tool(retriever) -> Callable[[str], ToolResult]:
    def search_weather_knowledge(query: str) -> ToolResult:
        hits = [h for h in retriever.retrieve(query) if h.score >= 0.05]
        if not hits:
            return ToolResult("No relevant passages found in the reference documents.")
        text = "\n\n".join(
            f"[{i}] ({h.chunk.document_name} — {h.chunk.section})\n{h.chunk.text}"
            for i, h in enumerate(hits, 1)
        )
        return ToolResult(text, {"sources": [
            {"document": h.chunk.document_name, "section": h.chunk.section,
             "relevance_score": round(h.score, 4)} for h in hits
        ]})
    return search_weather_knowledge


def _schema(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props, "required": required},
    }}


TOOL_SCHEMAS = [
    _schema(
        "search_weather_knowledge",
        "Search reference documents for weather terms, alerts (watch vs warning), the "
        "climate of India, and site safety SOPs (wind/crane limits, heatwave, flood).",
        {"query": {"type": "string", "description": "What to look up."}}, ["query"],
    ),
    _schema(
        "get_live_weather",
        "Get the CURRENT weather conditions for a city.",
        {"city": {"type": "string", "description": "City name, e.g. 'Mumbai'."}}, ["city"],
    ),
    _schema(
        "forecast_temperature",
        "Predict the NEXT-HOUR temperature for New Delhi using the trained ML model. "
        "Only supports New Delhi.",
        {}, [],
    ),
]


def build_registry(retriever) -> dict[str, Callable[..., ToolResult]]:
    return {
        "search_weather_knowledge": make_rag_tool(retriever),
        "get_live_weather": get_live_weather,
        "forecast_temperature": forecast_temperature,
    }
