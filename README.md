# 🌦️ New Capital Weather Prediction Challenge

## 🏙️ Background

Indonesia's ambitious plan to relocate its capital city from Jakarta to East Kalimantan (IKN) is driven by the need to address flooding, overpopulation, and environmental issues. However, the progress of this mega-project is heavily influenced by unpredictable weather—floods, extreme rain, and strong winds can delay development significantly.

This project aims to build a robust **time-series-based weather prediction model** focused on predicting hourly rainfall (`rain_1h`) to assist in project planning and risk management.

---

## 📁 Dataset Overview

The data consists of hourly weather records with 341,880 rows. Key columns include:
- **Numerical features**: `temp`, `feels`, `humidity`, `pressure`, `wind speed`, etc.
- **Timestamp**: `datetime_iso` used to extract cyclical and temporal features.
- **Target**: `rain_1h` (amount of rain in the past 1 hour in mm)

Some columns (like `sea_level`, `snow_3h`, `visibility`) were dropped due to high missing values or irrelevance.

---

## 🧹 Data Preprocessing

- Removed rows or columns with irrelevant values (e.g., `"unknown"`, `"not recorded"`, `-1`)
- Converted numerical columns from strings using regex and cast to `float`
- Handled outliers (e.g., `temp > 100°C`, `wind_deg > 360`)
- Replaced unrealistic values with `NaN` and applied **interpolation** followed by **backward fill**
- Standardized numeric features using **MinMaxScaler**

---

## 🔁 Feature Engineering

- **Temporal decomposition**: `hour`, `day`, `week`, `month`, `year`
- **Cyclical encoding**:
  - `hour_x`, `hour_y` using sin/cos of 24-hour cycle
  - `day_x`, `day_y` (within month), `month_x`, `month_y`, `week_x`, `week_y`
  - `wind_deg_x`, `wind_deg_y` for directionality
- **Interaction features**:
  - `wind_u`, `wind_v`: Decomposed wind speed into vector components
  - `temp_feels_diff`, `dew_temp_diff`, `humidity_temp_ratio`
  - `cloud_pressure_interaction`, `temp_range`, etc.
- **Aggregated rolling averages** for temperature, humidity, and pressure by day, week, and month

---

## 🤖 Modeling Approach

The final prediction model used:
- **LightGBM Regressor** for fast and accurate modeling
- **TimeSeriesSplit** to respect temporal order in cross-validation
- **Optuna** for hyperparameter optimization
- Target (`rain_1h`) and features scaled using **MinMaxScaler**
- Model evaluation based on **Root Mean Squared Error (RMSE)**

```python
# Example RMSE output
Final RMSE on training data: 0.5912
