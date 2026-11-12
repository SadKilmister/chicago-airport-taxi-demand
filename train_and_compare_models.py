from __future__ import annotations

import hashlib
import itertools
import json
import logging
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost
    from xgboost import XGBRegressor
except ImportError as exc:
    raise SystemExit("Установите XGBoost: pip install xgboost") from exc

try:
    import lightgbm
    from lightgbm import LGBMRegressor
except ImportError as exc:
    raise SystemExit("Установите LightGBM: pip install lightgbm") from exc

try:
    import catboost
    from catboost import CatBoostRegressor
except ImportError as exc:
    raise SystemExit("Установите CatBoost: pip install catboost") from exc


ROOT = Path(__file__).resolve().parent
FEATURES_PATH = (
    ROOT
    / "marts"
    / "hourly_airport_demand"
    / "forecast_training_features.csv"
)
RUNS_ROOT = ROOT / "model_runs"
LOG_PATH = ROOT / "run_history" / "model_training.log"

RANDOM_STATE = 42
HORIZON_HOURS = 24
TUNING_WINDOWS_PER_AIRPORT = 12

TARGET_COLUMN = "trip_count"
TIMESTAMP_COLUMN = "target_timestamp_local"
AIRPORT_COLUMN = "airport"

LAG_COLUMNS = {
    "taxi_lag_1h": 1,
    "taxi_lag_24h": 24,
    "taxi_lag_168h": 168,
}
ROLLING_COLUMNS = {
    "taxi_rolling_mean_24h": 24,
    "taxi_rolling_mean_168h": 168,
}


def configure_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_columns_from(data: pd.DataFrame) -> list[str]:
    excluded = {
        TARGET_COLUMN,
        TIMESTAMP_COLUMN,
        AIRPORT_COLUMN,
        "forecast_origin_timestamp",
        "flight_reference_timestamp",
    }
    columns = [
        name
        for name in data.columns
        if name not in excluded
        and pd.api.types.is_numeric_dtype(data[name])
    ]

    expected = set(LAG_COLUMNS) | set(ROLLING_COLUMNS)
    absent = expected.difference(columns)
    if absent:
        raise ValueError(
            "В таблице отсутствуют признаки спроса: "
            + ", ".join(sorted(absent))
        )
    return columns


def candidate_parameters() -> dict[str, list[dict[str, Any]]]:
    ridge = [
        {"alpha": alpha}
        for alpha in [0.1, 1.0, 10.0, 100.0, 500.0, 1000.0]
    ]

    rf = []
    for depth, leaf, features in itertools.product(
        [12, 20, None],
        [2, 5],
        [0.8, 1.0],
    ):
        rf.append(
            {
                "n_estimators": 300,
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "max_features": features,
            }
        )
    rf = rf[:8]

    xgb = [
        {
            "n_estimators": trees,
            "max_depth": depth,
            "learning_rate": rate,
            "min_child_weight": child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample,
            "reg_lambda": reg_lambda,
        }
        for trees, depth, rate, child_weight, subsample, colsample, reg_lambda in [
            (300, 3, 0.05, 1, 0.8, 0.8, 1.0),
            (500, 3, 0.03, 1, 0.8, 0.8, 5.0),
            (700, 3, 0.03, 3, 0.9, 0.8, 5.0),
            (300, 5, 0.05, 1, 0.8, 0.8, 1.0),
            (500, 5, 0.03, 3, 0.8, 0.9, 5.0),
            (700, 5, 0.03, 5, 0.9, 0.9, 10.0),
            (400, 7, 0.05, 3, 0.8, 0.8, 5.0),
            (600, 7, 0.03, 5, 0.9, 0.8, 10.0),
        ]
    ]

    lgbm = [
        {
            "n_estimators": trees,
            "learning_rate": rate,
            "num_leaves": leaves,
            "max_depth": depth,
            "min_child_samples": child,
            "subsample": subsample,
            "subsample_freq": 1,
            "colsample_bytree": colsample,
            "reg_lambda": reg_lambda,
            "verbosity": -1,
        }
        for trees, rate, leaves, depth, child, subsample, colsample, reg_lambda in [
            (300, 0.05, 15, 6, 20, 0.8, 0.8, 1.0),
            (500, 0.03, 15, 6, 30, 0.9, 0.8, 5.0),
            (700, 0.03, 31, 8, 30, 0.8, 0.9, 5.0),
            (300, 0.05, 31, 8, 50, 0.9, 0.8, 10.0),
            (500, 0.03, 31, -1, 50, 0.8, 0.9, 10.0),
            (700, 0.02, 63, 10, 50, 0.9, 0.9, 10.0),
            (400, 0.05, 63, 10, 80, 0.8, 1.0, 5.0),
            (600, 0.03, 15, -1, 80, 0.9, 1.0, 20.0),
        ]
    ]

    cat = [
        {
            "iterations": trees,
            "depth": depth,
            "learning_rate": rate,
            "l2_leaf_reg": l2,
            "random_strength": strength,
            "bagging_temperature": bagging,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": -1,
        }
        for trees, depth, rate, l2, strength, bagging in [
            (300, 5, 0.05, 3.0, 1.0, 0.5),
            (500, 5, 0.03, 5.0, 1.0, 1.0),
            (700, 6, 0.03, 5.0, 0.5, 1.0),
            (300, 7, 0.05, 5.0, 1.0, 0.5),
            (500, 7, 0.03, 7.0, 1.0, 1.0),
            (700, 7, 0.02, 7.0, 0.5, 2.0),
            (400, 8, 0.05, 9.0, 1.0, 1.0),
            (600, 8, 0.03, 9.0, 0.5, 2.0),
        ]
    ]

    return {
        "ridge": ridge,
        "random_forest": rf,
        "xgboost": xgb,
        "lightgbm": lgbm,
        "catboost": cat,
    }


def make_model(model_name: str, params: dict[str, Any]) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True))
    ]

    if model_name == "ridge":
        steps.extend(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(**params)),
            ]
        )
    elif model_name == "random_forest":
        steps.append(
            (
                "model",
                RandomForestRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    **params,
                ),
            )
        )
    elif model_name == "xgboost":
        steps.append(
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=0,
                    **params,
                ),
            )
        )
    elif model_name == "lightgbm":
        steps.append(
            (
                "model",
                LGBMRegressor(
                    objective="regression",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    **params,
                ),
            )
        )
    elif model_name == "catboost":
        steps.append(
            (
                "model",
                CatBoostRegressor(
                    loss_function="RMSE",
                    random_seed=RANDOM_STATE,
                    **params,
                ),
            )
        )
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")

    return Pipeline(steps)


def choose_tuning_starts(row_count: int) -> list[int]:
    full_windows = row_count // HORIZON_HOURS
    if full_windows == 0:
        return []

    number = min(TUNING_WINDOWS_PER_AIRPORT, full_windows)
    window_numbers = np.unique(
        np.linspace(0, full_windows - 1, num=number, dtype=int)
    )
    return [int(number * HORIZON_HOURS) for number in window_numbers]


def all_evaluation_starts(row_count: int) -> list[int]:
    full_windows = row_count // HORIZON_HOURS
    return [i * HORIZON_HOURS for i in range(full_windows)]


def simulate(
    model: Any | None,
    rows: pd.DataFrame,
    context_rows: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    split_name: str,
    starts_by_airport: dict[str, list[int]],
) -> pd.DataFrame:
    result: list[dict[str, Any]] = []

    context_by_airport = {
        str(airport): group.sort_values(TIMESTAMP_COLUMN)[TARGET_COLUMN]
        .astype(float)
        .to_numpy()
        for airport, group in context_rows.groupby(AIRPORT_COLUMN)
    }

    for airport, airport_rows in rows.groupby(AIRPORT_COLUMN, sort=True):
        airport = str(airport)
        airport_rows = airport_rows.sort_values(TIMESTAMP_COLUMN).reset_index(
            drop=True
        )
        actual = airport_rows[TARGET_COLUMN].astype(float).to_numpy()
        timestamps = airport_rows[TIMESTAMP_COLUMN].to_numpy()
        context = context_by_airport.get(airport, np.array([], dtype=float))

        starts = starts_by_airport.get(airport, [])
        for start in starts:
            end = start + HORIZON_HOURS
            if end > len(airport_rows):
                continue

            history = np.concatenate([context, actual[:start]]).tolist()
            if len(history) < 168:
                continue

            for position in range(start, end):
                feature_row = airport_rows.loc[
                    position, feature_columns
                ].copy()

                for column, lag in LAG_COLUMNS.items():
                    feature_row[column] = history[-lag]

                for column, window in ROLLING_COLUMNS.items():
                    feature_row[column] = float(np.mean(history[-window:]))

                if model is None:
                    # Сезонный наивный прогноз по тому же часу предыдущей недели.
                    prediction = history[-168]
                else:
                    prediction = float(
                        model.predict(
                            pd.DataFrame(
                                [feature_row.to_dict()],
                                columns=feature_columns,
                            )
                        )[0]
                    )
                prediction = max(0.0, float(prediction))

                result.append(
                    {
                        "model": model_name,
                        "airport": airport,
                        "split": split_name,
                        "timestamp": timestamps[position],
                        "actual": actual[position],
                        "prediction": prediction,
                        "forecast_step": position - start + 1,
                        "window_start": timestamps[start],
                    }
                )
                history.append(prediction)

    return pd.DataFrame(result)


def score(predictions: pd.DataFrame, training_rows: pd.DataFrame) -> dict[str, float]:
    if predictions.empty:
        raise ValueError("Нет прогнозов для расчёта метрик.")

    actual = predictions["actual"].to_numpy(dtype=float)
    predicted = predictions["prediction"].to_numpy(dtype=float)
    errors = np.abs(actual - predicted)

    values = training_rows[TARGET_COLUMN].astype(float).to_numpy()
    denominator = (
        np.abs(values[168:] - values[:-168]).mean()
        if len(values) > 168
        else float("nan")
    )

    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "WAPE_percent": float(errors.sum() / max(np.abs(actual).sum(), 1) * 100),
        "MASE": (
            float(errors.mean() / denominator)
            if denominator and np.isfinite(denominator)
            else float("nan")
        ),
        "forecast_rows": int(len(predictions)),
    }


def fit_and_tune(
    model_name: str,
    candidates: list[dict[str, Any]],
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    feature_columns: list[str],
    tuning_starts: dict[str, list[int]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    results: list[dict[str, Any]] = []

    for index, params in enumerate(candidates, start=1):
        started = time.perf_counter()
        logging.info(
            "%s: конфигурация %d/%d: %s",
            model_name,
            index,
            len(candidates),
            params,
        )

        model = make_model(model_name, params)
        model.fit(
            train_rows[feature_columns],
            train_rows[TARGET_COLUMN].astype(float),
        )

        predictions = simulate(
            model=model,
            rows=validation_rows,
            context_rows=train_rows,
            feature_columns=feature_columns,
            model_name=model_name,
            split_name="validation_tuning",
            starts_by_airport=tuning_starts,
        )
        metrics = score(predictions, train_rows)

        results.append(
            {
                "model": model_name,
                "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                **metrics,
                "duration_seconds": round(time.perf_counter() - started, 2),
            }
        )
        logging.info(
            "%s конфигурация %d: MAE=%.3f, окон=%d, %.1f сек.",
            model_name,
            index,
            metrics["MAE"],
            len(tuning_starts),
            time.perf_counter() - started,
        )

    table = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
    best_params = json.loads(table.loc[0, "params"])
    return best_params, table


def main() -> None:
    configure_logging()

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Не найдена таблица признаков: {FEATURES_PATH}"
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    logging.info("Читаю таблицу признаков: %s", FEATURES_PATH)
    data = pd.read_csv(FEATURES_PATH)
    data[TIMESTAMP_COLUMN] = pd.to_datetime(
        data[TIMESTAMP_COLUMN], errors="coerce"
    )
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    data = data.dropna(
        subset=[TIMESTAMP_COLUMN, TARGET_COLUMN, AIRPORT_COLUMN]
    ).copy()
    data[AIRPORT_COLUMN] = data[AIRPORT_COLUMN].astype(str)
    data = data.sort_values([TIMESTAMP_COLUMN, AIRPORT_COLUMN]).reset_index(
        drop=True
    )

    duplicate_count = data.duplicated([TIMESTAMP_COLUMN, AIRPORT_COLUMN]).sum()
    if duplicate_count:
        raise ValueError(
            f"Обнаружено повторов аэропорт + время: {duplicate_count}"
        )

    feature_columns = feature_columns_from(data)
    unique_times = np.array(sorted(data[TIMESTAMP_COLUMN].unique()))
    if len(unique_times) < 500:
        raise ValueError("Для временного разбиения слишком мало часов.")

    validation_start = pd.Timestamp(unique_times[int(len(unique_times) * 0.70)])
    test_start = pd.Timestamp(unique_times[int(len(unique_times) * 0.85)])

    train_rows = data[data[TIMESTAMP_COLUMN] < validation_start].copy()
    validation_rows = data[
        (data[TIMESTAMP_COLUMN] >= validation_start)
        & (data[TIMESTAMP_COLUMN] < test_start)
    ].copy()
    train_validation_rows = data[data[TIMESTAMP_COLUMN] < test_start].copy()
    test_rows = data[data[TIMESTAMP_COLUMN] >= test_start].copy()

    for name, frame in [
        ("Train", train_rows),
        ("Validation", validation_rows),
        ("Test", test_rows),
    ]:
        if frame.empty:
            raise ValueError(f"Часть {name} оказалась пустой.")
        logging.info(
            "%s: %s — %s (%d строк)",
            name,
            frame[TIMESTAMP_COLUMN].min(),
            frame[TIMESTAMP_COLUMN].max(),
            len(frame),
        )

    tuning_starts = {
        str(airport): choose_tuning_starts(len(group))
        for airport, group in validation_rows.groupby(AIRPORT_COLUMN)
    }
    evaluation_starts = {
        str(airport): all_evaluation_starts(len(group))
        for airport, group in validation_rows.groupby(AIRPORT_COLUMN)
    }
    test_starts = {
        str(airport): all_evaluation_starts(len(group))
        for airport, group in test_rows.groupby(AIRPORT_COLUMN)
    }

    logging.info(
        "Подбор будет оценивать до %d окон по 24 часа на аэропорт.",
        TUNING_WINDOWS_PER_AIRPORT,
    )

    candidate_tables: list[pd.DataFrame] = []
    final_predictions: list[pd.DataFrame] = []
    summary_records: list[dict[str, Any]] = []
    chosen_parameters: dict[str, dict[str, Any]] = {}

    # Сезонный наивный baseline не имеет подбираемых параметров.
    naive_validation = simulate(
        model=None,
        rows=validation_rows,
        context_rows=train_rows,
        feature_columns=feature_columns,
        model_name="naive_168h",
        split_name="validation",
        starts_by_airport=evaluation_starts,
    )
    naive_test = simulate(
        model=None,
        rows=test_rows,
        context_rows=train_validation_rows,
        feature_columns=feature_columns,
        model_name="naive_168h",
        split_name="test",
        starts_by_airport=test_starts,
    )
    naive_val_metrics = score(naive_validation, train_rows)
    naive_test_metrics = score(naive_test, train_validation_rows)
    final_predictions.extend([naive_validation, naive_test])
    summary_records.append(
        {
            "model": "naive_168h",
            "best_params": '{"seasonal_lag": 168}',
            **{f"validation_{k}": v for k, v in naive_val_metrics.items()},
            **{f"test_{k}": v for k, v in naive_test_metrics.items()},
        }
    )

    for model_name, candidates in candidate_parameters().items():
        best_params, candidate_table = fit_and_tune(
            model_name=model_name,
            candidates=candidates,
            train_rows=train_rows,
            validation_rows=validation_rows,
            feature_columns=feature_columns,
            tuning_starts=tuning_starts,
        )
        candidate_tables.append(candidate_table)
        chosen_parameters[model_name] = best_params

        # Полная валидация выбранной конфигурации: обучение только на Train.
        validation_model = make_model(model_name, best_params)
        validation_model.fit(
            train_rows[feature_columns],
            train_rows[TARGET_COLUMN].astype(float),
        )
        validation_predictions = simulate(
            model=validation_model,
            rows=validation_rows,
            context_rows=train_rows,
            feature_columns=feature_columns,
            model_name=model_name,
            split_name="validation",
            starts_by_airport=evaluation_starts,
        )
        validation_metrics = score(validation_predictions, train_rows)

        # Итоговый test: выбранные параметры, обучение Train + Validation.
        final_model = make_model(model_name, best_params)
        final_model.fit(
            train_validation_rows[feature_columns],
            train_validation_rows[TARGET_COLUMN].astype(float),
        )
        test_predictions = simulate(
            model=final_model,
            rows=test_rows,
            context_rows=train_validation_rows,
            feature_columns=feature_columns,
            model_name=model_name,
            split_name="test",
            starts_by_airport=test_starts,
        )
        test_metrics = score(test_predictions, train_validation_rows)

        joblib.dump(final_model, run_dir / f"{model_name}.joblib")
        final_predictions.extend([validation_predictions, test_predictions])

        summary_records.append(
            {
                "model": model_name,
                "best_params": json.dumps(
                    best_params, ensure_ascii=False, sort_keys=True
                ),
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
                **{
                    f"test_{key}": value
                    for key, value in test_metrics.items()
                },
            }
        )

        logging.info(
            "%s: validation MAE=%.3f, test MAE=%.3f",
            model_name,
            validation_metrics["MAE"],
            test_metrics["MAE"],
        )

    pd.concat(candidate_tables, ignore_index=True).to_csv(
        run_dir / "validation_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(summary_records).to_csv(
        run_dir / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(final_predictions, ignore_index=True).to_csv(
        run_dir / "recursive_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(FEATURES_PATH),
        "data_sha256": sha256_file(FEATURES_PATH),
        "target": TARGET_COLUMN,
        "airports": sorted(data[AIRPORT_COLUMN].unique().tolist()),
        "features": feature_columns,
        "split": {
            "validation_start": validation_start.isoformat(),
            "test_start": test_start.isoformat(),
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "test_rows": len(test_rows),
            "horizon_hours": HORIZON_HOURS,
            "tuning_windows_per_airport": TUNING_WINDOWS_PER_AIRPORT,
        },
        "selected_parameters": chosen_parameters,
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
            "catboost": catboost.__version__,
        },
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logging.info("Обучение завершено. Результаты сохранены в: %s", run_dir)
    logging.info("Журнал обучения: %s", LOG_PATH)


if __name__ == "__main__":
    main()