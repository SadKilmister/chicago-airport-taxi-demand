import argparse
import csv
import json
import logging
import os
import re
import time
import zipfile
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path

import requests


BASE_URL = "https://www.transtats.bts.gov/PREZIP/"
SOURCE_URL = "https://www.transtats.bts.gov/ontime/"
ARCHIVE_PATTERN = re.compile(
    r"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_"
    r"(\d{4})_(\d{1,2})\.zip"
)

OUTPUT_FIELDS = [
    "FlightDate",
    "Origin",
    "Dest",
    "CRSDepTime",
    "DepTime",
    "CRSArrTime",
    "ArrTime",
    "ArrDelay",
    "Cancelled",
    "Diverted",
    "CRSElapsedTime",
    "Distance",
]

AIRPORTS = ("ORD", "MDW")
MAX_ATTEMPTS = 8
CHUNK_SIZE = 1024 * 1024

ROOT = Path(__file__).resolve().parent
ARCHIVE_DIR = ROOT / "input_store" / "bts_archives"
ARRIVAL_DIR = ROOT / "input_store" / "bts_arrivals"
LOG_PATH = ROOT / "run_history" / "bts_collection.log"
MANIFEST_PATH = ROOT / "run_history" / "bts_collection_manifest.json"


def parse_month(value: str) -> tuple[int, int]:
    try:
        parsed = datetime.strptime(value, "%Y-%m")
        return parsed.year, parsed.month
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Ожидается месяц в формате YYYY-MM, получено: {value}"
        ) from exc


def shift_month(year: int, month: int, offset: int) -> tuple[int, int]:
    index = year * 12 + (month - 1) + offset
    return index // 12, index % 12 + 1


def month_range(
    start: tuple[int, int],
    end_inclusive: tuple[int, int],
) -> list[tuple[int, int]]:
    months = []
    current = start

    while current <= end_inclusive:
        months.append(current)
        current = shift_month(*current, 1)

    return months


def month_key(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


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


def request_with_retries(
    session: requests.Session,
    url: str,
    *,
    stream: bool = False,
) -> requests.Response:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = session.get(
                url,
                stream=stream,
                timeout=(30, 180),
            )

            if response.status_code == 429 or response.status_code >= 500:
                status_code = response.status_code
                response.close()

                if attempt == MAX_ATTEMPTS:
                    raise RuntimeError(
                        f"Источник вернул HTTP {status_code} "
                        f"после {MAX_ATTEMPTS} попыток."
                    )

                wait_seconds = min(attempt * 10, 60)
                logging.warning(
                    "Источник вернул HTTP %s; повтор через %s сек. "
                    "(попытка %s из %s)",
                    status_code,
                    wait_seconds,
                    attempt,
                    MAX_ATTEMPTS,
                )
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            return response

        except requests.RequestException as exc:
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Не удалось получить данные после "
                    f"{MAX_ATTEMPTS} попыток."
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

    raise RuntimeError("Не удалось выполнить запрос к BTS.")


def discover_available_months(
    session: requests.Session,
) -> list[tuple[int, int]]:
    logging.info("Проверяю список доступных месячных архивов BTS")

    response = request_with_retries(session, BASE_URL)
    listing = response.text
    response.close()

    available = {
        (int(year), int(month))
        for year, month in ARCHIVE_PATTERN.findall(listing)
    }

    available_months = sorted(
        month for month in available if month >= (2024, 1)
    )

    if not available_months:
        raise RuntimeError(
            "В официальном каталоге BTS не найдены архивы On-Time "
            "Performance за 2024 год и позднее."
        )

    logging.info(
        "Каталог BTS содержит данные с %s по %s",
        month_key(*available_months[0]),
        month_key(*available_months[-1]),
    )

    return available_months


def read_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}

    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Не удалось прочитать манифест: {MANIFEST_PATH}"
        ) from exc


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = MANIFEST_PATH.with_suffix(".json.part")

    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_path, MANIFEST_PATH)


def archive_url(year: int, month: int) -> str:
    filename = (
        "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_"
        f"{year}_{month}.zip"
    )
    return BASE_URL + filename


def download_archive(
    session: requests.Session,
    year: int,
    month: int,
    archive_path: Path,
    *,
    refresh: bool,
) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    if archive_path.exists() and not refresh:
        if zipfile.is_zipfile(archive_path):
            logging.info(
                "%s: архив уже скачан; использую его",
                month_key(year, month),
            )
            return

        logging.warning(
            "%s: существующий архив повреждён; скачиваю заново",
            month_key(year, month),
        )

    temporary_path = archive_path.with_suffix(".zip.part")
    url = archive_url(year, month)

    logging.info("%s: скачиваю архив BTS", month_key(year, month))

    try:
        response = request_with_retries(session, url, stream=True)

        with response, temporary_path.open("wb") as output_file:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    output_file.write(chunk)

        if not zipfile.is_zipfile(temporary_path):
            raise RuntimeError(
                f"Скачанный файл за {month_key(year, month)} "
                "не является корректным ZIP-архивом."
            )

        with zipfile.ZipFile(temporary_path) as archive:
            bad_member = archive.testzip()

        if bad_member is not None:
            raise RuntimeError(
                f"Архив за {month_key(year, month)} повреждён: "
                f"{bad_member}"
            )

        os.replace(temporary_path, archive_path)

        logging.info(
            "%s: архив сохранён, размер %.1f МБ",
            month_key(year, month),
            archive_path.stat().st_size / (1024 * 1024),
        )

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def extract_airport_arrivals(
    archive_path: Path,
    output_path: Path,
    year: int,
    month: int,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".csv.part")

    scanned_rows = 0
    airport_counts = {airport: 0 for airport in AIRPORTS}

    with zipfile.ZipFile(archive_path) as archive:
        csv_members = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_members:
            raise RuntimeError(
                f"В архиве за {month_key(year, month)} не найден CSV-файл."
            )

        csv_member = csv_members[0]
        logging.info(
            "%s: обрабатываю файл %s",
            month_key(year, month),
            csv_member,
        )

        with archive.open(csv_member) as binary_file:
            with TextIOWrapper(
                binary_file,
                encoding="utf-8-sig",
                errors="replace",
                newline="",
            ) as text_file:
                reader = csv.DictReader(text_file)

                if not reader.fieldnames:
                    raise RuntimeError(
                        f"В CSV за {month_key(year, month)} "
                        "не найдены заголовки столбцов."
                    )

                reader.fieldnames = [
                    field.strip() if field else ""
                    for field in reader.fieldnames
                ]

                missing_fields = set(OUTPUT_FIELDS) - set(reader.fieldnames)
                if missing_fields:
                    raise RuntimeError(
                        f"В CSV за {month_key(year, month)} "
                        "отсутствуют обязательные поля: "
                        f"{sorted(missing_fields)}"
                    )

                with temporary_path.open(
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as output_file:
                    writer = csv.DictWriter(
                        output_file,
                        fieldnames=OUTPUT_FIELDS,
                        extrasaction="ignore",
                    )
                    writer.writeheader()

                    for row in reader:
                        scanned_rows += 1
                        destination = (
                            row.get("Dest") or ""
                        ).strip().upper()

                        if destination not in AIRPORTS:
                            continue

                        writer.writerow(
                            {
                                field: row.get(field, "")
                                for field in OUTPUT_FIELDS
                            }
                        )
                        airport_counts[destination] += 1

                        if scanned_rows % 1_000_000 == 0:
                            logging.info(
                                "%s: просмотрено %s строк",
                                month_key(year, month),
                                scanned_rows,
                            )

    if sum(airport_counts.values()) == 0:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"В архиве за {month_key(year, month)} "
            "не найдено прилётов в ORD или MDW."
        )

    os.replace(temporary_path, output_path)

    return {
        "scanned_rows": scanned_rows,
        "airport_rows": airport_counts,
        "filtered_rows": sum(airport_counts.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Загрузка BTS On-Time Performance и фильтрация прилётов "
            "в ORD и MDW."
        )
    )
    parser.add_argument(
        "--start-month",
        type=parse_month,
        default=(2024, 1),
        help="Первый месяц загрузки, формат YYYY-MM (по умолчанию 2024-01).",
    )
    parser.add_argument(
        "--refresh-months",
        type=int,
        default=2,
        help=(
            "Сколько последних доступных месяцев скачивать заново "
            "(по умолчанию 2)."
        ),
    )

    args = parser.parse_args()

    if args.refresh_months < 0:
        parser.error("--refresh-months не может быть отрицательным")

    setup_logging()

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "ChicagoAirportTaxiDemand/1.0 (student project)"}
    )

    available_months = discover_available_months(session)
    latest_month = available_months[-1]
    requested_months = month_range(args.start_month, latest_month)

    if not requested_months:
        raise SystemExit(
            "Нет доступных месяцев для загрузки. Проверьте --start-month."
        )

    available_set = set(available_months)
    missing_months = [
        month for month in requested_months if month not in available_set
    ]

    if missing_months:
        formatted = ", ".join(
            month_key(*month) for month in missing_months
        )
        raise RuntimeError(
            "В каталоге BTS отсутствуют месяцы внутри запрошенного "
            f"периода: {formatted}"
        )

    refresh_from = max(0, len(requested_months) - args.refresh_months)
    months_to_refresh = set(requested_months[refresh_from:])

    manifest = read_manifest()

    logging.info("Источник: %s", SOURCE_URL)
    logging.info(
        "Период загрузки: %s — %s",
        month_key(*requested_months[0]),
        month_key(*requested_months[-1]),
    )
    logging.info("Аэропорты назначения: %s", ", ".join(AIRPORTS))
    logging.info("Месяцев к обработке: %s", len(requested_months))

    for year, month in requested_months:
        key = month_key(year, month)
        archive_path = ARCHIVE_DIR / (
            "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_"
            f"{year}_{month}.zip"
        )
        arrivals_path = ARRIVAL_DIR / f"bts_arrivals_{key}.csv"
        refresh = (year, month) in months_to_refresh

        if (
            arrivals_path.exists()
            and archive_path.exists()
            and not refresh
        ):
            logging.info(
                "%s: архив и фильтрованный файл уже есть",
                key,
            )
            continue

        download_archive(
            session=session,
            year=year,
            month=month,
            archive_path=archive_path,
            refresh=refresh,
        )

        result = extract_airport_arrivals(
            archive_path=archive_path,
            output_path=arrivals_path,
            year=year,
            month=month,
        )

        manifest[key] = {
            "source": SOURCE_URL,
            "archive_url": archive_url(year, month),
            "archive_file": str(archive_path.relative_to(ROOT)),
            "filtered_file": str(arrivals_path.relative_to(ROOT)),
            "archive_size_bytes": archive_path.stat().st_size,
            "scanned_rows": result["scanned_rows"],
            "filtered_rows": result["filtered_rows"],
            "airport_rows": result["airport_rows"],
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

        save_manifest(manifest)

        logging.info(
            "%s завершён: просмотрено %s строк; "
            "ORD — %s, MDW — %s",
            key,
            result["scanned_rows"],
            result["airport_rows"]["ORD"],
            result["airport_rows"]["MDW"],
        )

    logging.info("Загрузка BTS завершена")
    logging.info("Манифест: %s", MANIFEST_PATH)


if __name__ == "__main__":
    main()