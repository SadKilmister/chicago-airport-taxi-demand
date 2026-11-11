import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd


AIRPORT_BY_AREA = {
    56: "MDW",
    76: "ORD",
}

AIRPORTS = ("ORD", "MDW")
TIMEZONE = "America/Chicago"

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "run_history" / "taxi_collection_manifest.json"
OUTPUT_DIR = ROOT / "marts" / "hourly_airport_demand"
OUTPUT_PATH = OUTPUT_DIR / "taxi_hourly_demand.csv"
QUALITY_PATH = OUTPUT_DIR / "taxi_data_quality.json"
LOG_PATH = ROOT / "run_history" / "taxi_preparation.log"


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


def parse_manifest_month(month_key: str) -> tuple[int, int]:
    try:
        parsed = datetime.strptime(month_key, "%Y-%m")
        return parsed.year, parsed.month
    except ValueError as exc:
        raise ValueError(
            f"Некорректный месяц в манифесте: {month_key}"
        ) from exc


def next_month(year: int, month: int) -> tuple[int, int]:
    index = year * 12 + month
    return index // 12, index % 12 + 1


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Не найден манифест загрузки: {MANIFEST_PATH}\n"
            "Сначала завершите загрузку Chicago Taxi Trips."
        )

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    if not manifest:
        raise ValueError("Манифест загрузки пуст.")

    return manifest


def get_month_entries(manifest: dict) -> list[tuple[str, Path]]:
    entries = []

    for month_key in sorted(manifest):
        parse_manifest_month(month_key)

        relative_path = Path(manifest[month_key]["file"])
        file_path = (
            relative_path
            if relative_path.is_absolute()
            else ROOT / relative_path
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Файл, указанный в манифесте, не найден: {file_path}"
            )

        entries.append((month_key, file_path))

    months = [parse_manifest_month(key) for key, _ in entries]

    for previous, current in zip(months, months[1:]):
        if next_month(*previous) != current:
            raise ValueError(
                "В манифесте есть пропуск месяца между "
                f"{previous[0]:04d}-{previous[1]:02d} и "
                f"{current[0]:04d}-{current[1]:02d}. "
                "Повторно запустите загрузчик Taxi Trips."
            )

    if len(entries) < 2:
        raise ValueError(
            "Для обработки нужно не менее двух месяцев в манифесте: "
            "последний месяц будет исключён как потенциально неполный."
        )

    return entries


def load_and_aggregate(
    month_key: str,
    file_path: Path,
    quality: dict,
) -> pd.DataFrame:
    year, month = parse_manifest_month(month_key)

    raw = pd.read_csv(
        file_path,
        usecols=[
            "trip_start_timestamp",
            "pickup_community_area",
        ],
        dtype={
            "trip_start_timestamp": "string",
            "pickup_community_area": "string",
        },
    )

    input_rows = len(raw)
    quality["input_rows"] += input_rows
    quality["months"][month_key] = {
        "file": str(file_path.relative_to(ROOT)),
        "input_rows": input_rows,
    }

    timestamps = pd.to_datetime(
        raw["trip_start_timestamp"],
        errors="coerce",
    )

    areas = pd.to_numeric(
        raw["pickup_community_area"],
        errors="coerce",
    )

    invalid_timestamp = timestamps.isna()
    valid_area = areas.isin(AIRPORT_BY_AREA.keys())
    valid_timestamp = timestamps.notna()

    month_start = pd.Timestamp(year=year, month=month, day=1)
    next_year, next_month_number = next_month(year, month)
    month_end = pd.Timestamp(
        year=next_year,
        month=next_month_number,
        day=1,
    )

    in_source_month = (
        valid_timestamp
        & timestamps.ge(month_start)
        & timestamps.lt(month_end)
    )

    outside_month = valid_timestamp & ~in_source_month

    quality["invalid_timestamps"] += int(invalid_timestamp.sum())
    quality["invalid_airport_areas"] += int((~valid_area).sum())
    quality["timestamps_outside_source_month"] += int(outside_month.sum())

    valid_rows = valid_timestamp & valid_area & in_source_month

    quality["retained_rows"] += int(valid_rows.sum())
    quality["dropped_rows"] += int((~valid_rows).sum())
    quality["months"][month_key].update(
        {
            "retained_rows": int(valid_rows.sum()),
            "dropped_rows": int((~valid_rows).sum()),
        }
    )

    if not valid_rows.any():
        logging.warning(
            "%s: после проверки качества не осталось строк",
            month_key,
        )
        return pd.DataFrame(columns=["hour", "airport", "trip_count"])

    work = pd.DataFrame(
        {
            "timestamp": timestamps.loc[valid_rows],
            "area": areas.loc[valid_rows].astype(int),
        }
    )

    work["airport"] = work["area"].map(AIRPORT_BY_AREA)
    work["hour"] = work["timestamp"].dt.floor("h")

    aggregated = (
        work.groupby(["hour", "airport"])
        .size()
        .rename("trip_count")
        .reset_index()
    )

    logging.info(
        "%s: строк на входе %s, оставлено %s, агрегировано в %s "
        "часовых групп",
        month_key,
        input_rows,
        int(valid_rows.sum()),
        len(aggregated),
    )

    return aggregated


def build_complete_hourly_series(
    grouped_counts: pd.DataFrame,
    first_month: tuple[int, int],
    last_month: tuple[int, int],
) -> pd.DataFrame:
    start = pd.Timestamp(
        year=first_month[0],
        month=first_month[1],
        day=1,
    )

    end_year, end_month_number = next_month(*last_month)
    end = pd.Timestamp(
        year=end_year,
        month=end_month_number,
        day=1,
    )

    hours = pd.date_range(
        start=start,
        end=end,
        freq="h",
        inclusive="left",
    )

    localized = hours.tz_localize(
        TIMEZONE,
        ambiguous=True,
        nonexistent="NaT",
    )
    nonexistent_hour = localized.isna()
    hours = hours[~nonexistent_hour]

    if grouped_counts.empty:
        grouped_series = pd.Series(dtype="int64")
    else:
        grouped_series = grouped_counts.set_index(
            ["hour", "airport"]
        )["trip_count"]

    result_parts = []

    for airport in AIRPORTS:
        if grouped_series.empty:
            airport_counts = pd.Series(dtype="int64")
        else:
            try:
                airport_counts = grouped_series.xs(
                    airport,
                    level="airport",
                )
            except KeyError:
                airport_counts = pd.Series(dtype="int64")

        counts = airport_counts.reindex(hours, fill_value=0).astype("int64")

        result_parts.append(
            pd.DataFrame(
                {
                    "timestamp_local": hours.strftime(
                        "%Y-%m-%dT%H:%M:%S"
                    ),
                    "airport": airport,
                    "trip_count": counts.to_numpy(),
                }
            )
        )

    result = pd.concat(result_parts, ignore_index=True)
    result = result.sort_values(
        ["timestamp_local", "airport"]
    ).reset_index(drop=True)

    return result


def save_outputs(
    result: pd.DataFrame,
    quality: dict,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    temporary_csv = OUTPUT_PATH.with_suffix(".csv.part")
    result.to_csv(temporary_csv, index=False, encoding="utf-8")
    temporary_csv.replace(OUTPUT_PATH)

    quality["output_rows"] = len(result)
    quality["output_file"] = str(OUTPUT_PATH.relative_to(ROOT))
    quality["time_handling"] = (
        "Timestamps are treated as Chicago local wall time. "
        "The nonexistent spring-forward hour is excluded. "
        "The repeated fall-back hour is aggregated into one local hour."
    )
    quality["deduplication"] = (
        "No rows were removed as duplicates. The downloaded extract has "
        "no trip identifier, and multiple real trips can share the same "
        "rounded timestamp and airport area."
    )
    quality["zero_count_note"] = (
        "Zero means no matching trip record was present in the processed "
        "source files for that airport and hour."
    )

    temporary_json = QUALITY_PATH.with_suffix(".json.part")
    temporary_json.write_text(
        json.dumps(quality, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_json.replace(QUALITY_PATH)


def main() -> None:
    setup_logging()

    manifest = load_manifest()
    all_entries = get_month_entries(manifest)

    # Последний месяц в выгрузке может быть неполным. Сохраняем его в raw,
    # но исключаем из целевого ряда и дальнейшего обучения.
    excluded_month_key, excluded_month_path = all_entries[-1]
    included_entries = all_entries[:-1]
    excluded_month_rows = manifest[excluded_month_key].get("rows")

    logging.info(
        "Исключаю последний месяц %s как потенциально неполный "
        "(в манифесте %s строк): %s",
        excluded_month_key,
        excluded_month_rows,
        excluded_month_path,
    )

    quality = {
        "source": "Chicago Taxi Trips (2024-)",
        "source_url": "https://data.cityofchicago.org/d/ajtu-isnz",
        "airport_areas": {
            "MDW": 56,
            "ORD": 76,
        },
        "source_months_in_manifest": len(all_entries),
        "processed_months": len(included_entries),
        "excluded_latest_month": {
            "month": excluded_month_key,
            "file": str(excluded_month_path.relative_to(ROOT)),
            "source_rows": excluded_month_rows,
            "reason": (
                "Последний месяц выгрузки исключён как потенциально "
                "неполный, чтобы не трактовать отсутствующие будущие "
                "часы как нулевой спрос."
            ),
        },
        "input_rows": 0,
        "retained_rows": 0,
        "dropped_rows": 0,
        "invalid_timestamps": 0,
        "invalid_airport_areas": 0,
        "timestamps_outside_source_month": 0,
        "months": {},
    }

    monthly_aggregates = []

    logging.info(
        "Начинаю подготовку %s полных помесячных файлов",
        len(included_entries),
    )

    for month_key, file_path in included_entries:
        monthly_aggregates.append(
            load_and_aggregate(
                month_key=month_key,
                file_path=file_path,
                quality=quality,
            )
        )

    grouped_counts = pd.concat(
        monthly_aggregates,
        ignore_index=True,
    )

    first_month = parse_manifest_month(included_entries[0][0])
    last_month = parse_manifest_month(included_entries[-1][0])

    result = build_complete_hourly_series(
        grouped_counts=grouped_counts,
        first_month=first_month,
        last_month=last_month,
    )

    if result.empty:
        raise ValueError("Почасовой набор данных оказался пустым.")

    save_outputs(result, quality)

    logging.info("Подготовка завершена")
    logging.info("Целевая таблица: %s", OUTPUT_PATH)
    logging.info("Часовых строк: %s", len(result))
    logging.info(
        "Строк поездок сохранено после очистки: %s",
        quality["retained_rows"],
    )
    logging.info("Строк отброшено: %s", quality["dropped_rows"])
    logging.info(
        "Исключён неполный месяц: %s",
        excluded_month_key,
    )
    logging.info("Отчёт качества: %s", QUALITY_PATH)


if __name__ == "__main__":
    main()