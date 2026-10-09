# Weather Forecast Model: Model Card

Project document (written for this assistant; not an external source).
Numbers below come from the evaluation in `notebooks/02_sklearn_baseline.ipynb`. The final model has not yet been saved to `models/`.

## What the model predicts
- **Target:** air temperature in degrees Celsius, **24 hours ahead** (horizon H = 24).
- **Granularity:** hourly. One prediction per hour, for a single location.
- **Intended use:** a quick reference estimate of tomorrow's temperature for operations staff. It is not an official forecast and does not replace National Weather Service forecasts or warnings.

## Training data
- Hourly weather observations, 2006-01-01 to 2016-12-31 (local time). About 96,400 hourly rows after cleaning (96,453 raw rows).
- Variables used: temperature, humidity, wind speed, pressure, plus time-of-day and season features and lagged temperatures.
- See `weather-dataset-data-dictionary.md` for each column.

## Features (11)
- Current observations: temperature (C), humidity, wind speed (km/h), pressure (millibars).
- Time features: hour of day and month of year, encoded as sine and cosine so that 23:00 is next to 00:00 and December is next to January.
- Lag features: temperature 1 hour ago, temperature 24 hours ago, and the mean temperature over the last 24 hours.
- Excluded on purpose: apparent temperature (it is computed from air temperature, humidity and wind, so it leaks the answer), and cloud cover (constant in this dataset).

## Models compared
Evaluation used 5-fold time-series cross-validation (always train on the past, test on the following period). Lower RMSE is better. RMSE is in degrees Celsius.

| Model | Mean RMSE (C) |
|---|---|
| Persistence baseline (tomorrow's temperature = current temperature) | 3.14 |
| Linear regression | 2.90 |
| Gradient boosting (default) | 2.85 |
| Gradient boosting (tuned: shallow trees, 400 iterations, learning rate 0.05) | 2.80 |
| Random forest | 2.95 |
| Extra trees | 2.92 |

Every model beat the persistence baseline in all five folds. The best model is about 11 percent better than persistence, which means a typical 24-hour-ahead error of about 2.8 C.

## Findings that matter for users
- The most important input is the current temperature. Time of year and time of day come next. Lag features added almost nothing.
- Differences between the better models are small (about 0.05 C) compared with variation between time periods (about 0.2 C).
- Using only one split can mislead: on one test split persistence scored 2.99 and boosting 2.60, while cross-validation gave 3.14 and 2.85.

## Limitations
- Trained on one location only. Do not use it for other places.
- Uses only past observations from one weather station. It cannot see approaching fronts, storms or pressure systems, so errors are largest when the weather changes abruptly.
- Forecast error is about 2.8 C on average and can be much larger on individual days.
- Learned from 2006 to 2016 data; climate and station conditions may have changed.
- Predicts temperature only, not rain, wind, storms or alerts.
