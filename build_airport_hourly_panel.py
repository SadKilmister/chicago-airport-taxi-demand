import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent

TAXI_QUALITY_PATH = (
    ROOT
    / "marts"
    / "hourly_airport_demand"
    / "taxi_data_quality.json"
)
TAXI_HOURLY_PATH = (
    ROOT
    / "marts"
    / "hourly_airport_demand"
    / "taxi_hourly_demand.csv"
)
BTS_MANIFEST_PATH = ROOT / "run_history" / "bts_collection_manifest.json"

OUTPUT_DIR = ROOT / "marts" / "hourly_airport_demand"
OUTPUT_PATH = OUTPUT_DIR / "airport_hourly_panel.csv"
QUALITY_PATH = OUTPUT_DIR / "airport_panel_quality.json"
LOG_PATH = ROOT / "run_history" / "airport_panel.log"

AIRPORTS = ("ORD", "MDW")

# Для внутренних перелётов в США разница между часовым поясом аэропорта
# назначения (Chicago) и аэропорта отправления находится в этом диапазоне.
# Это нужно, чтобы восстановить переход даты из FlightDate в дату прилёта.
MIN_OFFSET_DIFFERENCE_MINUTES = -240
MAX_OFFSET_DIFFERENCE_MINUTES = 360


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


def read_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден {label}: {path}"
        )

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Не удалось прочитать {label}: {path}"
        ) from exc


def parse_month(month_key: str) -> tuple[int, int]:
    try:
        parsed = datetime.strptime(month_key, "%Y-%m")
        return parsed.year, parsed.month
    except ValueError as exc:
        raise ValueError(
            f"Некорректный месяц в источнике: {month_key}"
        ) from exc


def month_key_from_timestamp(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m")


def parse_hhmm(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Преобразует BTS-время HHMM в число минут от начала местных суток.

    Поддерживает значение 2400, трактуя его как конец дня.
    Возвращает (минуты, маска корректных значений).
    """
    numeric = pd.to_numeric(series, errors="coerce")

    hours = numeric // 100
    minutes = numeric % 100

    valid = (
        numeric.notna()
        & numeric.ge(0)
        & numeric.le(2400)
        & minutes.ge(0)
        & minutes.lt(60)
        & (
            hours.lt(24)
            | (hours.eq(24) & minutes.eq(0))
        )
    )

    total_minutes = (hours * 60 + minutes).where(valid)

    return total_minutes, valid


def get_common_months(
    taxi_quality: dict,
    bts_manifest: dict,
) -> tuple[list[str], list[str], list[str]]:
    taxi_months = sorted(taxi_quality.get("months", {}).keys())
    bts_months = sorted(bts_manifest.keys())

    if not taxi_months:
        raise ValueError(
            "В отчёте качества Taxi нет обработанных месяцев."
        )

    if not bts_months:
        raise ValueError(
            "Манифест BTS пуст."
        )

    common_months = sorted(set(taxi_months) & set(bts_months))

    if not common_months:
        raise ValueError(
            "У Taxi и BTS нет общих месяцев."
        )

    for month in common_months:
        parse_month(month)

    taxi_only = sorted(set(taxi_months) - set(common_months))
    bts_only = sorted(set(bts_months) - set(common_months))

    return common_months, taxi_only, bts_only


def load_taxi_target(common_months: list[str]) -> pd.DataFrame:
    if not TAXI_HOURLY_PATH.exists():
        raise FileNotFoundError(
            f"Не найден почасовой ряд Taxi: {TAXI_HOURLY_PATH}"
        )

    taxi = pd.read_csv(
        TAXI_HOURLY_PATH,
        dtype={
            "timestamp_local": "string",
            "airport": "string",
        },
    )

    required = {
        "timestamp_local",
        "airport",
        "trip_count",
    }
    missing = required - set(taxi.columns)

    if missing:
        raise ValueError(
            f"В почасовом ряду Taxi отсутствуют поля: {sorted(missing)}"
        )

    taxi["timestamp_local"] = pd.to_datetime(
        taxi["timestamp_local"],
        errors="coerce",
    )
    taxi["trip_count"] = pd.to_numeric(
        taxi["trip_count"],
        errors="coerce",
    )

    invalid = (
        taxi["timestamp_local"].isna()
        | taxi["trip_count"].isna()
        | ~taxi["airport"].isin(AIRPORTS)
    )

    if invalid.any():
        raise ValueError(
            f"В почасовом ряду Taxi найдено некорректных строк: "
            f"{int(invalid.sum())}"
        )

    taxi["month"] = taxi["timestamp_local"].dt.strftime("%Y-%m")
    taxi = taxi[taxi["month"].isin(common_months)].copy()

    taxi["trip_count"] = taxi["trip_count"].astype("int64")

    duplicates = taxi.duplicated(
        subset=["timestamp_local", "airport"]
    )

    if duplicates.any():
        raise ValueError(
            "В почасовом ряду Taxi есть повторные пары "
            "«время — аэропорт»."
        )

    expected_airports = set(AIRPORTS)
    airport_count = taxi.groupby("timestamp_local")["airport"].agg(
        lambda values: set(values)
    )

    incomplete_hours = airport_count[
        airport_count != expected_airports
    ]

    if not incomplete_hours.empty:
        raise ValueError(
            "В Taxi-таблице есть часы без одной из аэропортовых зон. "
            f"Количество неполных часов: {len(incomplete_hours)}"
        )

    return taxi[
        ["timestamp_local", "airport", "trip_count", "month"]
    ]


def process_bts_month(
    month_key: str,
    manifest_entry: dict,
    quality: dict,
) -> pd.DataFrame:
    relative_path = Path(manifest_entry["filtered_file"])
    file_path = (
        relative_path
        if relative_path.is_absolute()
        else ROOT / relative_path
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Файл BTS из манифеста не найден: {file_path}"
        )

    bts = pd.read_csv(
        file_path,
        dtype="string",
        keep_default_na=False,
    )

    required_fields = {
        "FlightDate",
        "Origin",
        "Dest",
        "CRSDepTime",
        "CRSArrTime",
        "CRSElapsedTime",
        "ArrTime",
        "ArrDelay",
        "Cancelled",
        "Diverted",
    }
    missing = required_fields - set(bts.columns)

    if missing:
        raise ValueError(
            f"{month_key}: в файле BTS отсутствуют поля "
            f"{sorted(missing)}"
        )

    source_rows = len(bts)
    bts["Dest"] = bts["Dest"].str.strip().str.upper()
    bts["airport"] = bts["Dest"]

    flight_date = pd.to_datetime(
        bts["FlightDate"],
        errors="coerce",
    )

    dep_minutes, valid_dep_time = parse_hhmm(bts["CRSDepTime"])
    arr_minutes, valid_arr_time = parse_hhmm(bts["CRSArrTime"])

    elapsed = pd.to_numeric(
        bts["CRSElapsedTime"],
        errors="coerce",
    )

    cancelled = (
        pd.to_numeric(bts["Cancelled"], errors="coerce")
        .fillna(0)
        .eq(1)
    )
    diverted = (
        pd.to_numeric(bts["Diverted"], errors="coerce")
        .fillna(0)
        .eq(1)
    )
    arrival_delay = pd.to_numeric(
        bts["ArrDelay"],
        errors="coerce",
    )

    valid_destination = bts["airport"].isin(AIRPORTS)
    valid_base = (
        valid_destination
        & flight_date.notna()
        & valid_dep_time
        & valid_arr_time
        & elapsed.notna()
        & elapsed.gt(0)
    )

    rollover_count = pd.Series(0, index=bts.index, dtype="int64")
    rollover_value = pd.Series(-1, index=bts.index, dtype="int64")

    # FlightDate — дата начала операции; BTS сообщает времена локально.
    # Проверяем возможный переход суток по расписанию и длительности рейса.
    for day_offset in range(3):
        inferred_offset_difference = (
            arr_minutes
            - dep_minutes
            + 1440 * day_offset
            - elapsed
        )

        valid_offset = (
            valid_base
            & inferred_offset_difference.ge(
                MIN_OFFSET_DIFFERENCE_MINUTES
            )
            & inferred_offset_difference.le(
                MAX_OFFSET_DIFFERENCE_MINUTES
            )
        )

        rollover_count += valid_offset.astype("int64")
        rollover_value = rollover_value.where(
            ~valid_offset,
            day_offset,
        )

    valid_rollover = rollover_count.eq(1)
    scheduled_valid = valid_base & valid_rollover

    scheduled_timestamp = (
        flight_date
        + pd.to_timedelta(
            rollover_value.where(scheduled_valid, 0),
            unit="D",
        )
        + pd.to_timedelta(
            arr_minutes.fillna(0),
            unit="m",
        )
    )

    scheduled_hour = scheduled_timestamp.dt.floor("h")

    scheduled_events = pd.DataFrame(
        {
            "timestamp_local": scheduled_hour.loc[scheduled_valid],
            "airport": bts.loc[scheduled_valid, "airport"],
            "scheduled_arrival_count": 1,
            "cancelled_flight_count": cancelled.loc[scheduled_valid]
            .astype("int64"),
            "diverted_flight_count": diverted.loc[scheduled_valid]
            .astype("int64"),
        }
    )

    scheduled_hourly = (
        scheduled_events.groupby(
            ["timestamp_local", "airport"],
            as_index=False,
        )
        .agg(
            scheduled_arrival_count=(
                "scheduled_arrival_count",
                "sum",
            ),
            cancelled_flight_count=(
                "cancelled_flight_count",
                "sum",
            ),
            diverted_flight_count=(
                "diverted_flight_count",
                "sum",
            ),
        )
    )

    actual_time_minutes, valid_actual_time = parse_hhmm(bts["ArrTime"])

    actual_valid = (
        scheduled_valid
        & ~cancelled
        & ~diverted
        & valid_actual_time
        & arrival_delay.notna()
    )

    # ArrDelay задаётся относительно расписания в минутах. Прибавляя его
    # к плановому локальному времени, получаем фактический час прилёта.
    actual_timestamp = (
        scheduled_timestamp
        + pd.to_timedelta(
            arrival_delay.fillna(0),
            unit="m",
        )
    )
    actual_hour = actual_timestamp.dt.floor("h")

    actual_events = pd.DataFrame(
        {
            "timestamp_local": actual_hour.loc[actual_valid],
            "airport": bts.loc[actual_valid, "airport"],
            "actual_arrival_count": 1,
            "arrival_delay_sum": arrival_delay.loc[actual_valid],
            "delay_observation_count": 1,
        }
    )

    if actual_events.empty:
        actual_hourly = pd.DataFrame(
            columns=[
                "timestamp_local",
                "airport",
                "actual_arrival_count",
                "arrival_delay_sum",
                "delay_observation_count",
            ]
        )
    else:
        actual_hourly = (
            actual_events.groupby(
                ["timestamp_local", "airport"],
                as_index=False,
            )
            .agg(
                actual_arrival_count=(
                    "actual_arrival_count",
                    "sum",
                ),
                arrival_delay_sum=(
                    "arrival_delay_sum",
                    "sum",
                ),
                delay_observation_count=(
                    "delay_observation_count",
                    "sum",
                ),
            )
        )

    monthly = scheduled_hourly.merge(
        actual_hourly,
        on=["timestamp_local", "airport"],
        how="outer",
    )

    quality["bts_rows_read"] += source_rows
    quality["scheduled_rows_aggregated"] += int(scheduled_valid.sum())
    quality["rows_without_valid_scheduled_time"] += int(
        (valid_destination & ~valid_base).sum()
    )
    quality["rows_with_unresolved_arrival_date"] += int(
        (valid_base & ~valid_rollover).sum()
    )
    quality["completed_arrivals_aggregated"] += int(actual_valid.sum())
    quality["cancelled_flights_in_source"] += int(
        (valid_destination & cancelled).sum()
    )
    quality["diverted_flights_in_source"] += int(
        (valid_destination & diverted).sum()
    )

    quality["months"][month_key] = {
        "file": str(file_path.relative_to(ROOT)),
        "source_rows": source_rows,
        "scheduled_rows_aggregated": int(scheduled_valid.sum()),
        "rows_without_valid_scheduled_time": int(
            (valid_destination & ~valid_base).sum()
        ),
        "rows_with_unresolved_arrival_date": int(
            (valid_base & ~valid_rollover).sum()
        ),
        "completed_arrivals_aggregated": int(actual_valid.sum()),
        "cancelled_flights_in_source": int(
            (valid_destination & cancelled).sum()
        ),
        "diverted_flights_in_source": int(
            (valid_destination & diverted).sum()
        ),
    }

    logging.info(
        "%s: прочитано %s записей, расписаний агрегировано %s, "
        "фактических прилётов %s",
        month_key,
        source_rows,
        int(scheduled_valid.sum()),
        int(actual_valid.sum()),
    )

    return monthly


def build_panel(
    taxi: pd.DataFrame,
    bts_hourly: pd.DataFrame,
) -> pd.DataFrame:
    if bts_hourly.empty:
        raise ValueError(
            "После обработки BTS не осталось почасовых данных."
        )

    bts_hourly = (
        bts_hourly.groupby(
            ["timestamp_local", "airport"],
            as_index=False,
        )
        .sum(numeric_only=True)
    )

    panel = taxi.merge(
        bts_hourly,
        on=["timestamp_local", "airport"],
        how="left",
        validate="one_to_one",
    )

    count_columns = [
        "scheduled_arrival_count",
        "cancelled_flight_count",
        "diverted_flight_count",
        "actual_arrival_count",
        "arrival_delay_sum",
        "delay_observation_count",
    ]

    for column in count_columns:
        if column not in panel.columns:
            panel[column] = 0

        panel[column] = panel[column].fillna(0)

    panel["scheduled_arrival_count"] = panel[
        "scheduled_arrival_count"
    ].astype("int64")
    panel["cancelled_flight_count"] = panel[
        "cancelled_flight_count"
    ].astype("int64")
    panel["diverted_flight_count"] = panel[
        "diverted_flight_count"
    ].astype("int64")
    panel["actual_arrival_count"] = panel[
        "actual_arrival_count"
    ].astype("int64")
    panel["delay_observation_count"] = panel[
        "delay_observation_count"
    ].astype("int64")

    panel["mean_arrival_delay_minutes"] = (
        panel["arrival_delay_sum"]
        / panel["delay_observation_count"].replace(0, pd.NA)
    )

    panel = panel.drop(columns=["arrival_delay_sum"])
    panel = panel.sort_values(
        ["timestamp_local", "airport"]
    ).reset_index(drop=True)

    return panel


def main() -> None:
    setup_logging()

    taxi_quality = read_json(
        TAXI_QUALITY_PATH,
        "отчёт качества Taxi",
    )
    bts_manifest = read_json(
        BTS_MANIFEST_PATH,
        "манифест BTS",
    )

    common_months, taxi_only, bts_only = get_common_months(
        taxi_quality,
        bts_manifest,
    )

    logging.info(
        "Общие месяцы: %s — %s (%s месяцев)",
        common_months[0],
        common_months[-1],
        len(common_months),
    )

    if taxi_only:
        logging.info(
            "Месяцы Taxi без данных BTS исключены: %s",
            ", ".join(taxi_only),
        )

    if bts_only:
        logging.info(
            "Месяцы BTS без целевого ряда Taxi исключены: %s",
            ", ".join(bts_only),
        )

    taxi = load_taxi_target(common_months)

    quality = {
        "sources": {
            "taxi": "Chicago Taxi Trips (2024-)",
            "bts": "BTS Reporting Carrier On-Time Performance",
        },
        "airport_codes": list(AIRPORTS),
        "common_months": common_months,
        "taxi_months_without_bts": taxi_only,
        "bts_months_without_taxi": bts_only,
        "taxi_rows_in_common_period": len(taxi),
        "bts_rows_read": 0,
        "scheduled_rows_aggregated": 0,
        "rows_without_valid_scheduled_time": 0,
        "rows_with_unresolved_arrival_date": 0,
        "completed_arrivals_aggregated": 0,
        "cancelled_flights_in_source": 0,
        "diverted_flights_in_source": 0,
        "months": {},
        "arrival_date_method": (
            "BTS FlightDate is treated as the operation date at the "
            "origin. The scheduled arrival date is inferred using CRS "
            "departure time, CRS arrival time, CRSElapsedTime, and a "
            "plausible Chicago-to-US-origin UTC-offset difference range "
            f"of {MIN_OFFSET_DIFFERENCE_MINUTES} to "
            f"{MAX_OFFSET_DIFFERENCE_MINUTES} minutes. Rows without "
            "exactly one valid day offset are excluded from flight "
            "aggregation and counted in the quality report."
        ),
        "actual_arrival_method": (
            "Actual arrival time is derived from scheduled local arrival "
            "timestamp plus BTS ArrDelay. Cancelled and diverted flights "
            "are excluded from completed-arrival counts."
        ),
        "feature_availability_note": (
            "Same-hour flight values in this panel are historical "
            "descriptive values. Forecast models must use only lagged "
            "flight values that would be known at forecast time."
        ),
    }

    monthly_aggregates = []

    for month in common_months:
        manifest_entry = bts_manifest.get(month)

        if not manifest_entry:
            raise ValueError(
                f"В манифесте BTS отсутствует месяц {month}."
            )

        monthly_aggregates.append(
            process_bts_month(
                month_key=month,
                manifest_entry=manifest_entry,
                quality=quality,
            )
        )

    bts_hourly = pd.concat(
        monthly_aggregates,
        ignore_index=True,
    )

    panel = build_panel(taxi, bts_hourly)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    temporary_csv = OUTPUT_PATH.with_suffix(".csv.part")
    panel.to_csv(temporary_csv, index=False, encoding="utf-8")
    os.replace(temporary_csv, OUTPUT_PATH)

    quality["output_rows"] = len(panel)
    quality["output_file"] = str(OUTPUT_PATH.relative_to(ROOT))
    quality["period_start"] = panel["timestamp_local"].min().isoformat()
    quality["period_end"] = panel["timestamp_local"].max().isoformat()
    quality["airport_rows"] = {
        airport: int((panel["airport"] == airport).sum())
        for airport in AIRPORTS
    }
    quality["zero_flight_count_note"] = (
        "A zero flight count means no matching BTS flight record was "
        "present for that airport and local hour in the overlapping "
        "source period."
    )
    quality["delay_missing_note"] = (
        "Mean arrival delay is blank when no completed arrival with a "
        "valid delay observation exists in that hour."
    )

    temporary_json = QUALITY_PATH.with_suffix(".json.part")
    temporary_json.write_text(
        json.dumps(quality, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_json, QUALITY_PATH)

    logging.info("Объединённая таблица сохранена: %s", OUTPUT_PATH)
    logging.info("Строк в панели: %s", len(panel))
    logging.info(
        "Такси поездок в общем периоде: %s",
        int(panel["trip_count"].sum()),
    )
    logging.info(
        "Плановых прилётов агрегировано: %s",
        quality["scheduled_rows_aggregated"],
    )
    logging.info(
        "Фактических прилётов агрегировано: %s",
        quality["completed_arrivals_aggregated"],
    )
    logging.info("Отчёт качества: %s", QUALITY_PATH)


if __name__ == "__main__":
    main()