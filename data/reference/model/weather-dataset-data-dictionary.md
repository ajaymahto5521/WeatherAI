# Weather Dataset: Data Dictionary

Project document (written for this assistant; not an external source).
File: `data/raw/weatherHistory.csv`. Hourly observations, 96,453 rows, local timestamps from 2006-01-01 to 2016-12-31.
The file does not state its location. It matches the widely shared "Weather in Szeged 2006-2016" dataset (Hungary), which fits its Central European time offsets (+0100 in winter, +0200 in summer).

## Columns

| Column | Unit | Meaning | Notes |
|---|---|---|---|
| Formatted Date | local time with UTC offset | Timestamp of the hourly observation | Parsed to UTC. 24 duplicate timestamps were removed, and a 3-hour gap was filled by interpolation. |
| Summary | text | Short description of the conditions (27 values, e.g. Partly Cloudy, Mostly Cloudy, Overcast, Clear, Foggy) | Categorical. |
| Precip Type | text | Type of precipitation: rain or snow | 517 missing values. |
| Temperature (C) | degrees Celsius | Measured air temperature | Range about -21.8 to 39.9, mean 11.9. |
| Apparent Temperature (C) | degrees Celsius | How warm or cold it feels (combines temperature, humidity and wind) | Nearly identical to temperature (correlation 0.99). Not used as a feature because it leaks the temperature target. |
| Humidity | fraction 0 to 1 | Relative humidity | 22 zero values were treated as missing. |
| Wind Speed (km/h) | kilometres per hour | Wind speed | Range 0 to 63.9, mean 10.8. To convert to knots divide by 1.852 (the Beaufort scale uses knots). |
| Wind Bearing (degrees) | degrees 0 to 359 | Direction the wind comes from | 0 and 360 are the same direction. |
| Visibility (km) | kilometres | Horizontal visibility | Range 0 to 16.1. |
| Loud Cover | fraction | Cloud cover (the column name is a typo in the source) | Always 0, so it carries no information and is dropped. |
| Pressure (millibars) | millibars (hPa) | Atmospheric pressure | 1,288 rows were 0 (sensor dropouts, not real values) and were treated as missing and interpolated. Normal range is about 974 to 1046. |
| Daily Summary | text | One-sentence description of the day | Categorical text, not used as a feature. |

## Cleaning applied before modeling
1. Timestamps converted to UTC, sorted, duplicates removed.
2. Constant column Loud Cover dropped.
3. Zero values in Pressure and Humidity treated as missing.
4. Reindexed to a complete hourly grid, so there are 96,432 rows after cleaning.
5. Numeric gaps linearly interpolated, text gaps filled with the previous value.

## Related reference material
- Heat index, wind chill and Beaufort scale pages in `weather_basics/` explain how temperature, humidity and wind combine into what it feels like and into wind categories.
