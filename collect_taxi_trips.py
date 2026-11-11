import argparse
import csv
import json
import logging
import os
import time
from datetime import date
from pathlib import Path

import requests


API_URL = "https://data.cityofchicago.org/resource/ajtu-isnz.json"
SOURCE_URL = "https://data.cityofchicago.org/d/ajtu-isnz"
AIRPORT_AREAS = (56, 76)
PAGE_SIZE = 10_000
MAX_ATTEMPTS = 8


def shift_month(year: int, month: int, offset: int) -> tuple[int, int]:
    month_index = year * 12 + (month - 1) + offset
    return month_index // 12, month_index % 12 + 1


def parse_month(value: str) -> tuple[int, int]:
    try:
        year_text, month_text = value.split("-")
        year, month = int(year_text), int(month_text)

        if not 1 <= month <= 12:
            raise ValueError

        return year, month

    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Ожидается месяц в формате YYYY-MM, получено: {value}"
        ) from exc


def month_range(start: tuple[int, int], end: tuple[int, int]):
    year, month = start

    while (year, month) < end:
        yield year, month
        year, month = shift_month(year, month, 1)


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def fetch_page(session: requests.Session, params: dict) -> list[dict]:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = session.get(
                API_URL,
                params=params,
                timeout=(15, 180),
            )

            if response.status_code == 429 or response.status_code >= 500:
                if attempt == MAX_ATTEMPTS:
                    response.raise_for_status()

                wait_seconds = min(attempt * 10, 60)
                logging.warning(
                    "Источник вернул HTTP %s; повтор через %s сек. "
                    "(попытка %s из %s)",
                    response.status_code,
                    wait_seconds,
                    attempt,
                    MAX_ATTEMPTS,
                )
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            return response.json()

        except (requests.RequestException, ValueError) as exc:
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Не удалось получить страницу после "
                    f"{MAX_ATTEMPTS} попыток"
                ) from exc

            wait_seconds = min(attempt * 10, 60)
            logging.warning(
                "Ошибка запроса: %s; повтор через %s сек. "
                "(попытка %s из %s)",
                exc,
                wait_seconds,
                attempt,
                MAX_ATTEMPTS,
            )
            time.sleep(wait_seconds)

    raise RuntimeError("Не удалось получить данные из источника")


def download_month(
    session: requests.Session,
    year: int,
    month: int,
    output_path: Path,
) -> int:
    next_year, next_month = shift_month(year, month, 1)

    period_start = f"{year:04d}-{month:02d}-01T00:00:00"
    period_end = f"{next_year:04d}-{next_month:02d}-01T00:00:00"

    where_clause = (
        f"pickup_community_area in ({','.join(map(str, AIRPORT_AREAS))}) "
        f"AND trip_start_timestamp >= '{period_start}' "
        f"AND trip_start_timestamp < '{period_end}'"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")

    total_rows = 0
    offset = 0

    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "trip_start_timestamp",
                "pickup_community_area",
            ],
        )
        writer.writeheader()

        while True:
            params = {
                "$select": "trip_start_timestamp,pickup_community_area",
                "$where": where_clause,
                "$order": ":id",
                "$limit": PAGE_SIZE,
                "$offset": offset,
            }

            rows = fetch_page(session, params)

            if not rows:
                break

            for row in rows:
                writer.writerow(
                    {
                        "trip_start_timestamp": row.get(
                            "trip_start_timestamp", ""
                        ),
                        "pickup_community_area": row.get(
                            "pickup_community_area", ""
                        ),
                    }
                )

            total_rows += len(rows)
            offset += len(rows)

            logging.info(
                "%04d-%02d: получено %s записей",
                year,
                month,
                total_rows,
            )

            if len(rows) < PAGE_SIZE:
                break

    if total_rows == 0:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"За {year:04d}-{month:02d} не получено ни одной записи. "
            "Проверьте доступность источника и формат полей."
        )

    os.replace(temporary_path, output_path)
    return total_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Загрузка поездок такси в аэропортовых зонах Чикаго."
    )

    parser.add_argument(
        "--start-month",
        type=parse_month,
        default=(2024, 1),
        help="Первый месяц выгрузки, формат YYYY-MM (по умолчанию 2024-01).",
    )

    parser.add_argument(
        "--refresh-months",
        type=int,
        default=2,
        help=(
            "Сколько последних месяцев загружать заново "
            "(по умолчанию 2)."
        ),
    )

    args = parser.parse_args()

    if args.refresh_months < 0:
        parser.error("--refresh-months не может быть отрицательным")

    root = Path(__file__).resolve().parent
    data_dir = root / "input_store" / "taxi_trips"
    log_path = root / "run_history" / "taxi_collection.log"
    manifest_path = root / "run_history" / "taxi_collection_manifest.json"

    setup_logging(log_path)

    # Не загружаем текущий месяц: данные за него ещё неполные.
    today = date.today()
    end_exclusive = (today.year, today.month)

    months = list(month_range(args.start_month, end_exclusive))

    if not months:
        raise SystemExit(
            "Нет месяцев для загрузки. Проверьте значение --start-month."
        )

    refresh_from = max(0, len(months) - args.refresh_months)
    refreshed_months = set(months[refresh_from:])

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "ChicagoAirportTaxiDemand/1.0 (student project)"
            )
        }
    )

    logging.info("Начало загрузки. Источник: %s", SOURCE_URL)
    logging.info(
        "Период: %04d-%02d — %04d-%02d (правая граница не включается)",
        args.start_month[0],
        args.start_month[1],
        end_exclusive[0],
        end_exclusive[1],
    )
    logging.info(
        "Аэропортовые community areas: %s",
        AIRPORT_AREAS,
    )

    for year, month in months:
        month_key = f"{year:04d}-{month:02d}"
        output_path = data_dir / f"taxi_trips_{month_key}.csv"

        if output_path.exists() and month_key not in refreshed_months:
            logging.info("%s уже загружен; пропускаю", month_key)
            continue

        logging.info("Загружаю %s", month_key)

        row_count = download_month(
            session=session,
            year=year,
            month=month,
            output_path=output_path,
        )

        manifest[month_key] = {
            "file": str(output_path.relative_to(root)),
            "rows": row_count,
            "source": SOURCE_URL,
            "airport_community_areas": list(AIRPORT_AREAS),
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logging.info(
            "%s завершён: %s записей",
            month_key,
            row_count,
        )

    logging.info("Загрузка завершена")


if __name__ == "__main__":
    main()