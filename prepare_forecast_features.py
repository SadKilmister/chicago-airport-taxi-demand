import json
import logging
import math
import os
from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


ROOT = Path(__file__).resolve().parent

PANEL_PATH = (
    ROOT
    / "marts"
    / "hourly_airport_demand"
    / "airport_hourly_panel.csv"
)
PANEL_QUALITY_PATH = (
    ROOT
    / "marts"
    / "hourly_airport_demand"
    / "airport_panel_quality.json"
)

OUTPUT_DIR = ROOT / "marts" / "hourly_airport_demand"
OUTPUT_PATH = OUTPUT_DIR / "forecast_training_features.csv"
QUALITY_PATH = OUTPUT_DIR / "forecast_feature_quality.json"
LOG_PATH = ROOT / "run_history" / "forecast_feature_build.log"

AIRPORTS = ("ORD", "MDW")
FLIGHT_DATA_LAG_MONTHS = 3

TAXI_FEATURE_COLUMNS = [
    "taxi_lag_1h",
    "taxi_lag_24h",
    "taxi_lag_168h",
    "taxi_rolling_mean_24h",
    "taxi_rolling_mean_168h",
]

BTS_FEATURE_COLUMNS = [
    "scheduled_arrivals_lag_3m",
    "actual_arrivals_lag_3m",
    "cancelled_flights_lag_3m",
    "diverted_flights_lag_3m",
    "mean_arrival_delay_lag_3m",
    "bts_lag_3m_available",
    "mean_arrival_delay_lag_3m_missing",
]

CALENDAR_FEATURE_COLUMNS = [
    "airport_ord",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "is_weekend",
    "is_us_federal_holiday",
]

MODEL_FEATURE_COLUMNS = (
    TAXI_FEATURE_COLUMNS
    + BTS_FEATURE_COLUMNS
    + CALENDAR_FEATURE_COLUMNS
)


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def read_json(path: Path, description: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден {description}: {path}"
        )

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Не удалось прочитать {description}: {path}"
        ) from exc


def load_panel() -> pd.DataFrame:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Не найдена объединённая почасовая панель: {PANEL_PATH}"
        )

    panel = pd.read_csv(
        PANEL_PATH,
        dtype={
            "timestamp_local": "string",
            "airport": "string",
        },
    )

    required_columns = {
        "timestamp_local",
        "airport",
        "trip_count",
        "scheduled_arrival_count",
        "actual_arrival_count",
        "cancelled_flight_count",
        "diverted_flight_count",
        "mean_arrival_delay_minutes",
    }

    missing_columns = required_columns - set(panel.columns)
    if missing_columns:
        raise ValueError(
            "В объединённой панели отсутствуют обязательные поля: "
            f"{sorted(missing_columns)}"
        )

    panel["timestamp_local"] = pd.to_datetime(
        panel["timestamp_local"],
        errors="coerce",
    )

    if panel["timestamp_local"].isna().any():
        bad_count = int(panel["timestamp_local"].isna().sum())
        raise ValueError(
            f"В панели найдено некорректных временных меток: {bad_count}"
        )

    if not panel["airport"].isin(AIRPORTS).all():
        raise ValueError(
            "В панели обнаружен неизвестный аэропорт."
        )

    numeric_columns = [
        "trip_count",
        "scheduled_arrival_count",
        "actual_arrival_count",
        "cancelled_flight_count",
        "diverted_flight_count",
        "mean_arrival_delay_minutes",
    ]

    for column in numeric_columns:
        panel[column] = pd.to_numeric(
            panel[column],
            errors="coerce",
        )

    required_numeric = [
        "trip_count",
        "scheduled_arrival_count",
        "actual_arrival_count",
        "cancelled_flight_count",
        "diverted_flight_count",
    ]

    if panel[required_numeric].isna().any().any():
        raise ValueError(
            "В обязательных числовых полях панели есть пропуски."
        )

    if (panel["trip_count"] < 0).any():
        raise ValueError(
            "В целевой переменной обнаружены отрицательные значения."
        )

    duplicates = panel.duplicated(
        subset=["timestamp_local", "airport"]
    )
    if duplicates.any():
        raise ValueError(
            "В панели есть повторные пары «время — аэропорт»."
        )

    return panel.sort_values(
        ["airport", "timestamp_local"]
    ).reset_index(drop=True)


def make_calendar_features(
    timestamps: pd.DatetimeIndex,
    airport: str,
    holiday_dates: set,
) -> pd.DataFrame:
    hour = timestamps.hour
    weekday = timestamps.day_of_week
    month = timestamps.month
    day_of_year = timestamps.day_of_year

    return pd.DataFrame(
        {
            "airport_ord": int(airport == "ORD"),
            "hour_sin": [
                math.sin(2 * math.pi * value / 24)
                for value in hour
            ],
            "hour_cos": [
                math.cos(2 * math.pi * value / 24)
                for value in hour
            ],
            "weekday_sin": [
                math.sin(2 * math.pi * value / 7)
                for value in weekday
            ],
            "weekday_cos": [
                math.cos(2 * math.pi * value / 7)
                for value in weekday
            ],
            "month_sin": [
                math.sin(2 * math.pi * (value - 1) / 12)
                for value in month
            ],
            "month_cos": [
                math.cos(2 * math.pi * (value - 1) / 12)
                for value in month
            ],
            "day_of_year_sin": [
                math.sin(2 * math.pi * (value - 1) / 365.25)
                for value in day_of_year
            ],
            "day_of_year_cos": [
                math.cos(2 * math.pi * (value - 1) / 365.25)
                for value in day_of_year
            ],
            "is_weekend": (weekday >= 5).astype("int8"),
            "is_us_federal_holiday": [
                int(timestamp.date() in holiday_dates)
                for timestamp in timestamps
            ],
        },
        index=timestamps,
    )


def build_airport_features(
    airport_panel: pd.DataFrame,
    holiday_dates: set,
    quality: dict,
) -> pd.DataFrame:
    airport_panel = airport_panel.sort_values(
        "timestamp_local"
    ).copy()

    airport = str(airport_panel["airport"].iloc[0])

    airport_panel = airport_panel.set_index("timestamp_local")

    if airport_panel.index.has_duplicates:
        raise ValueError(
            f"{airport}: обнаружены повторные временные метки."
        )

    target = airport_panel["trip_count"].astype("float64")
    timestamps = airport_panel.index

    taxi_lag_1h = target.shift(1)

    taxi_lag_24h = pd.Series(
        target.reindex(
            timestamps - pd.Timedelta(hours=24)
        ).to_numpy(),
        index=timestamps,
    )

    taxi_lag_168h = pd.Series(
        target.reindex(
            timestamps - pd.Timedelta(hours=168)
        ).to_numpy(),
        index=timestamps,
    )

    taxi_rolling_mean_24h = (
        target.shift(1)
        .rolling(window=24, min_periods=24)
        .mean()
    )

    taxi_rolling_mean_168h = (
        target.shift(1)
        .rolling(window=168, min_periods=168)
        .mean()
    )

    flight_reference_timestamps = (
        timestamps - pd.DateOffset(months=FLIGHT_DATA_LAG_MONTHS)
    )

    bts_sources = {
        "scheduled_arrivals_lag_3m": "scheduled_arrival_count",
        "actual_arrivals_lag_3m": "actual_arrival_count",
        "cancelled_flights_lag_3m": "cancelled_flight_count",
        "diverted_flights_lag_3m": "diverted_flight_count",
        "mean_arrival_delay_lag_3m": "mean_arrival_delay_minutes",
    }

    bts_features = {}

    for feature_name, source_column in bts_sources.items():
        bts_features[feature_name] = pd.Series(
            airport_panel[source_column]
            .reindex(flight_reference_timestamps)
            .to_numpy(),
            index=timestamps,
        )

    bts_reference_available = pd.Series(
        flight_reference_timestamps
        .isin(airport_panel.index)
        .astype("int8"),
        index=timestamps,
    )

    delay_missing = bts_features[
        "mean_arrival_delay_lag_3m"
    ].isna().astype("int8")

    calendar = make_calendar_features(
        timestamps=timestamps,
        airport=airport,
        holiday_dates=holiday_dates,
    )

    result = pd.DataFrame(
        {
            "forecast_origin_timestamp": (
                timestamps - pd.Timedelta(hours=1)
            ),
            "target_timestamp_local": timestamps,
            "airport": airport,
            "trip_count": target.to_numpy(),
            "flight_reference_timestamp": flight_reference_timestamps,
            "taxi_lag_1h": taxi_lag_1h.to_numpy(),
            "taxi_lag_24h": taxi_lag_24h.to_numpy(),
            "taxi_lag_168h": taxi_lag_168h.to_numpy(),
            "taxi_rolling_mean_24h": (
                taxi_rolling_mean_24h.to_numpy()
            ),
            "taxi_rolling_mean_168h": (
                taxi_rolling_mean_168h.to_numpy()
            ),
            **{
                name: values.to_numpy()
                for name, values in bts_features.items()
            },
            "bts_lag_3m_available": (
                bts_reference_available.to_numpy()
            ),
            "mean_arrival_delay_lag_3m_missing": (
                delay_missing.to_numpy()
            ),
        },
        index=timestamps,
    )

    for column in calendar.columns:
        result[column] = calendar[column].to_numpy()

    quality["airport_rows_before_warmup"][airport] = len(result)

    # До трёх месяцев BTS-истории нет нужного лага.
    flight_history_start = (
        timestamps.min()
        + pd.DateOffset(months=FLIGHT_DATA_LAG_MONTHS)
    )

    result = result[
        result["target_timestamp_local"] >= flight_history_start
    ].copy()

    quality["rows_removed_before_bts_history"][airport] = (
        quality["airport_rows_before_warmup"][airport] - len(result)
    )

    required_taxi_features = [
        "taxi_lag_1h",
        "taxi_lag_24h",
        "taxi_lag_168h",
        "taxi_rolling_mean_24h",
        "taxi_rolling_mean_168h",
    ]

    valid_taxi_history = result[
        required_taxi_features
    ].notna().all(axis=1)

    removed_taxi_warmup = int((~valid_taxi_history).sum())
    result = result[valid_taxi_history].copy()

    quality["rows_removed_for_taxi_warmup"][airport] = (
        removed_taxi_warmup
    )
    quality["rows_with_missing_bts_lag"][airport] = int(
        (result["bts_lag_3m_available"] == 0).sum()
    )
    quality["rows_with_missing_bts_delay"][airport] = int(
        result["mean_arrival_delay_lag_3m"].isna().sum()
    )

    return result.reset_index(drop=True)


def main() -> None:
    setup_logging()

    panel_quality = read_json(
        PANEL_QUALITY_PATH,
        "отчёт качества объединённой панели",
    )
    panel = load_panel()

    min_timestamp = panel["timestamp_local"].min()
    max_timestamp = panel["timestamp_local"].max()

    holiday_calendar = USFederalHolidayCalendar()
    holiday_index = holiday_calendar.holidays(
        start=min_timestamp.normalize(),
        end=max_timestamp.normalize(),
    )
    holiday_dates = set(holiday_index.date)

    quality = {
        "source_panel": str(PANEL_PATH.relative_to(ROOT)),
        "period_start": min_timestamp.isoformat(),
        "period_end": max_timestamp.isoformat(),
        "panel_rows": len(panel),
        "flight_data_lag_months": FLIGHT_DATA_LAG_MONTHS,
        "forecast_design": (
            "One-step-ahead hourly model. The 24-hour forecast is "
            "recursive: previous model predictions are fed back as "
            "Taxi history for later forecast hours."
        ),
        "bts_availability_policy": (
            "BTS features are taken from the same local hour three "
            "calendar months before the target timestamp. Same-hour "
            "BTS outcomes are not used as future predictors."
        ),
        "airport_rows_before_warmup": {},
        "rows_removed_before_bts_history": {},
        "rows_removed_for_taxi_warmup": {},
        "rows_with_missing_bts_lag": {},
        "rows_with_missing_bts_delay": {},
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "target_column": "trip_count",
    }

    if panel_quality.get("common_months"):
        quality["common_source_months"] = (
            panel_quality["common_months"]
        )

    feature_parts = []

    for airport in AIRPORTS:
        airport_panel = panel[
            panel["airport"] == airport
        ].copy()

        if airport_panel.empty:
            raise ValueError(
                f"В панели нет данных для аэропорта {airport}."
            )

        features = build_airport_features(
            airport_panel=airport_panel,
            holiday_dates=holiday_dates,
            quality=quality,
        )

        if features.empty:
            raise ValueError(
                f"После подготовки признаков для {airport} "
                "не осталось обучающих строк."
            )

        logging.info(
            "%s: сформировано %s строк признаков",
            airport,
            len(features),
        )
        feature_parts.append(features)

    result = pd.concat(feature_parts, ignore_index=True)
    result = result.sort_values(
        ["target_timestamp_local", "airport"]
    ).reset_index(drop=True)

    missing_model_columns = (
        set(MODEL_FEATURE_COLUMNS) - set(result.columns)
    )
    if missing_model_columns:
        raise ValueError(
            "Не сформированы признаки модели: "
            f"{sorted(missing_model_columns)}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    temporary_csv = OUTPUT_PATH.with_suffix(".csv.part")
    result.to_csv(
        temporary_csv,
        index=False,
        encoding="utf-8",
    )
    os.replace(temporary_csv, OUTPUT_PATH)

    quality["output_file"] = str(OUTPUT_PATH.relative_to(ROOT))
    quality["output_rows"] = len(result)
    quality["output_rows_by_airport"] = {
        airport: int((result["airport"] == airport).sum())
        for airport in AIRPORTS
    }
    quality["feature_period_start"] = (
        pd.to_datetime(result["target_timestamp_local"])
        .min()
        .isoformat()
    )
    quality["feature_period_end"] = (
        pd.to_datetime(result["target_timestamp_local"])
        .max()
        .isoformat()
    )
    quality["missing_values_by_column"] = {
        column: int(result[column].isna().sum())
        for column in MODEL_FEATURE_COLUMNS
    }
    quality["missing_delay_handling"] = (
        "Missing lagged average arrival delay is retained as null and "
        "flagged by mean_arrival_delay_lag_3m_missing. It must be "
        "imputed inside each training fold, using training data only."
    )

    temporary_json = QUALITY_PATH.with_suffix(".json.part")
    temporary_json.write_text(
        json.dumps(quality, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_json, QUALITY_PATH)

    logging.info("Подготовка признаков завершена")
    logging.info("Таблица признаков: %s", OUTPUT_PATH)
    logging.info("Строк: %s", len(result))
    logging.info("Отчёт качества: %s", QUALITY_PATH)


if __name__ == "__main__":
    main()