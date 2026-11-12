from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.inspection import permutation_importance


ROOT = Path(__file__).resolve().parent
FEATURES_PATH = (
    ROOT
    / "marts"
    / "hourly_airport_demand"
    / "forecast_training_features.csv"
)
RUNS_ROOT = ROOT / "model_runs"
LOG_PATH = ROOT / "run_history" / "model_explainability.log"

TARGET_COLUMN = "trip_count"
TIMESTAMP_COLUMN = "target_timestamp_local"
AIRPORT_COLUMN = "airport"
RANDOM_STATE = 42
HORIZON_HOURS = 24

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


def find_latest_lightgbm_run() -> tuple[Path, Path, Path]:
    """Возвращает пути к модели, метаданным и папке актуального запуска."""
    if not RUNS_ROOT.exists():
        raise FileNotFoundError(f"Не найдена папка запусков: {RUNS_ROOT}")

    run_dirs = sorted(
        [path for path in RUNS_ROOT.iterdir() if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )

    for run_dir in run_dirs:
        model_path = run_dir / "lightgbm.joblib"
        metadata_path = run_dir / "run_metadata.json"
        if model_path.exists() and metadata_path.exists():
            return model_path, metadata_path, run_dir

    raise FileNotFoundError(
        "Не найдены lightgbm.joblib и run_metadata.json. "
        "Сначала запустите train_and_compare_models.py."
    )


def read_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Не найдена таблица признаков: {FEATURES_PATH}"
        )

    data = pd.read_csv(FEATURES_PATH)
    data[TIMESTAMP_COLUMN] = pd.to_datetime(
        data[TIMESTAMP_COLUMN], errors="coerce"
    )
    data[TARGET_COLUMN] = pd.to_numeric(
        data[TARGET_COLUMN], errors="coerce"
    )
    data = data.dropna(
        subset=[TIMESTAMP_COLUMN, TARGET_COLUMN, AIRPORT_COLUMN]
    ).copy()
    data[AIRPORT_COLUMN] = data[AIRPORT_COLUMN].astype(str)
    return data.sort_values(
        [AIRPORT_COLUMN, TIMESTAMP_COLUMN]
    ).reset_index(drop=True)


def sample_rows(
    data: pd.DataFrame,
    limit: int,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    if len(data) <= limit:
        return data.copy()
    return data.sample(n=limit, random_state=seed).sort_index()


def transform_features(
    model: Any,
    data: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Применяет к исходным признакам тот же imputer, что был при обучении."""
    preprocessor = model[:-1]
    transformed = preprocessor.transform(data[feature_columns])

    imputer = model.named_steps["imputer"]
    transformed_names = imputer.get_feature_names_out(feature_columns).tolist()

    if transformed.shape[1] != len(transformed_names):
        raise ValueError(
            "Размер преобразованной матрицы не совпал с количеством "
            "сформированных имён признаков."
        )

    return np.asarray(transformed), transformed_names


def tree_shap_values(
    model: Any,
    data: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Вычисляет TreeSHAP через встроенный интерфейс LightGBM."""
    transformed, transformed_names = transform_features(
        model,
        data,
        feature_columns,
    )
    booster = model.named_steps["model"].booster_
    contributions = booster.predict(transformed, pred_contrib=True)

    shap_values = contributions[:, :-1]
    base_values = contributions[:, -1]

    return (
        pd.DataFrame(
            shap_values,
            columns=transformed_names,
            index=data.index,
        ),
        base_values,
    )


def save_html_figure(figure: Any, path: Path) -> None:
    figure.write_html(
        path,
        include_plotlyjs="cdn",
        full_html=True,
    )


def save_importance_outputs(
    model: Any,
    data: pd.DataFrame,
    train_validation_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
    permutation_sample_size: int,
    shap_sample_size: int,
    dependence_feature: str,
) -> None:
    logging.info("Считаю permutation importance на отложенной выборке.")
    permutation_data = sample_rows(
        test_rows,
        limit=permutation_sample_size,
    )

    permutation_result = permutation_importance(
        estimator=model,
        X=permutation_data[feature_columns],
        y=permutation_data[TARGET_COLUMN].astype(float),
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    permutation_table = pd.DataFrame(
        {
            "feature": feature_columns,
            "permutation_importance_mean": permutation_result.importances_mean,
            "permutation_importance_std": permutation_result.importances_std,
        }
    )

    logging.info("Считаю TreeSHAP для сравнения временных периодов.")
    train_validation_sample = sample_rows(
        train_validation_rows,
        limit=shap_sample_size,
        seed=RANDOM_STATE,
    )
    test_sample = sample_rows(
        test_rows,
        limit=shap_sample_size,
        seed=RANDOM_STATE + 1,
    )

    trainval_shap, _ = tree_shap_values(
        model,
        train_validation_sample,
        feature_columns,
    )
    test_shap, _ = tree_shap_values(
        model,
        test_sample,
        feature_columns,
    )

    trainval_mean_abs = trainval_shap.abs().mean()
    test_mean_abs = test_shap.abs().mean()

    transformed_names = test_shap.columns.tolist()
    booster = model.named_steps["model"].booster_
    gain_values = booster.feature_importance(importance_type="gain")
    split_values = booster.feature_importance(importance_type="split")

    gain_table = pd.DataFrame(
        {
            "feature": transformed_names,
            "lightgbm_gain": gain_values,
            "lightgbm_split_count": split_values,
            "mean_abs_treeshap_train_validation": [
                float(trainval_mean_abs[name])
                for name in transformed_names
            ],
            "mean_abs_treeshap_test": [
                float(test_mean_abs[name])
                for name in transformed_names
            ],
        }
    )

    importance = gain_table.merge(
        permutation_table,
        on="feature",
        how="left",
    )
    importance = importance.sort_values(
        "permutation_importance_mean",
        ascending=False,
    )
    importance.to_csv(
        output_dir / "feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    period_comparison = gain_table[
        [
            "feature",
            "mean_abs_treeshap_train_validation",
            "mean_abs_treeshap_test",
        ]
    ].copy()
    period_comparison.to_csv(
        output_dir / "importance_by_period.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top_gain = (
        gain_table.sort_values("lightgbm_gain", ascending=False)
        .head(20)
        .sort_values("lightgbm_gain")
    )
    gain_figure = px.bar(
        top_gain,
        x="lightgbm_gain",
        y="feature",
        orientation="h",
        title="Важность признаков LightGBM по gain",
        labels={
            "lightgbm_gain": "Суммарное уменьшение ошибки",
            "feature": "Признак",
        },
    )
    save_html_figure(
        gain_figure,
        output_dir / "importance_lightgbm_gain.html",
    )

    top_permutation = (
        importance.sort_values(
            "permutation_importance_mean",
            ascending=False,
        )
        .head(20)
        .sort_values("permutation_importance_mean")
    )
    permutation_figure = px.bar(
        top_permutation,
        x="permutation_importance_mean",
        y="feature",
        orientation="h",
        error_x="permutation_importance_std",
        title="Permutation importance на тестовом периоде",
        labels={
            "permutation_importance_mean": "Рост MAE после перемешивания",
            "feature": "Признак",
        },
    )
    save_html_figure(
        permutation_figure,
        output_dir / "importance_permutation.html",
    )

    period_long = period_comparison.melt(
        id_vars="feature",
        var_name="period",
        value_name="mean_absolute_treeshap",
    )
    period_long["period"] = period_long["period"].map(
        {
            "mean_abs_treeshap_train_validation": "Train + Validation",
            "mean_abs_treeshap_test": "Test",
        }
    )
    top_period_features = (
        period_comparison.assign(
            rank_value=period_comparison[
                "mean_abs_treeshap_test"
            ]
        )
        .nlargest(20, "rank_value")["feature"]
        .tolist()
    )
    period_long = period_long[
        period_long["feature"].isin(top_period_features)
    ]

    period_figure = px.bar(
        period_long,
        x="mean_absolute_treeshap",
        y="feature",
        color="period",
        barmode="group",
        orientation="h",
        title="Изменение вклада признаков между периодами",
        labels={
            "mean_absolute_treeshap": "Средний абсолютный вклад TreeSHAP",
            "feature": "Признак",
            "period": "Период",
        },
    )
    save_html_figure(
        period_figure,
        output_dir / "importance_period_comparison.html",
    )

    if dependence_feature not in feature_columns:
        raise ValueError(
            f"Признак для графика зависимости не найден: "
            f"{dependence_feature}"
        )
    if dependence_feature not in test_shap.columns:
        raise ValueError(
            f"Для признака {dependence_feature} не рассчитан вклад TreeSHAP."
        )

    dependence_values = test_sample[dependence_feature].to_numpy()
    dependence_contributions = test_shap[dependence_feature].to_numpy()

    dependence_table = pd.DataFrame(
        {
            "feature_value": dependence_values,
            "shap_contribution": dependence_contributions,
            "hour": test_sample[TIMESTAMP_COLUMN].dt.hour.to_numpy(),
        }
    )
    dependence_figure = px.scatter(
        dependence_table,
        x="feature_value",
        y="shap_contribution",
        color="hour",
        opacity=0.65,
        title=f"Зависимость вклада TreeSHAP от {dependence_feature}",
        labels={
            "feature_value": dependence_feature,
            "shap_contribution": "Вклад признака в прогноз",
            "hour": "Час суток",
        },
        color_continuous_scale="Turbo",
    )
    save_html_figure(
        dependence_figure,
        output_dir / "feature_dependence.html",
    )

    logging.info(
        "Сохранены общие оценки важности и сравнение периодов."
    )


def explain_recursive_forecast(
    model: Any,
    data: pd.DataFrame,
    feature_columns: list[str],
    airport: str,
    forecast_start: pd.Timestamp,
    explain_step: int,
    output_dir: Path,
) -> None:
    """Строит 24-часовой рекурсивный прогноз и сохраняет TreeSHAP-вклады."""
    airport_data = (
        data[data[AIRPORT_COLUMN] == airport]
        .sort_values(TIMESTAMP_COLUMN)
        .reset_index(drop=True)
    )

    positions = airport_data.index[
        airport_data[TIMESTAMP_COLUMN] == forecast_start
    ].tolist()
    if not positions:
        raise ValueError(
            f"Время {forecast_start} не найдено для аэропорта {airport}."
        )

    start_index = positions[0]
    future_rows = airport_data.iloc[
        start_index : start_index + HORIZON_HOURS
    ].copy()
    if len(future_rows) != HORIZON_HOURS:
        raise ValueError(
            "Для выбранного времени нет полного 24-часового окна."
        )

    timestamps = future_rows[TIMESTAMP_COLUMN].to_numpy()
    if not np.all(np.diff(timestamps) == np.timedelta64(1, "h")):
        raise ValueError(
            "Выбранное 24-часовое окно содержит пропуск или скачок времени."
        )

    history = (
        airport_data.iloc[:start_index][TARGET_COLUMN]
        .astype(float)
        .tolist()
    )
    if len(history) < 168:
        raise ValueError(
            "Недостаточно предшествующей истории для лага 168 часов."
        )

    if not 1 <= explain_step <= HORIZON_HOURS:
        raise ValueError(
            f"Номер часа объяснения должен быть от 1 до {HORIZON_HOURS}."
        )

    booster = model.named_steps["model"].booster_
    output_records: list[dict[str, Any]] = []
    selected_step_values: dict[str, Any] | None = None

    for step, (_, row) in enumerate(future_rows.iterrows(), start=1):
        feature_row = row[feature_columns].copy()

        for column, lag in LAG_COLUMNS.items():
            feature_row[column] = history[-lag]

        for column, window in ROLLING_COLUMNS.items():
            feature_row[column] = float(np.mean(history[-window:]))

        model_input = pd.DataFrame(
            [feature_row.to_dict()],
            columns=feature_columns,
        )
        transformed, transformed_names = transform_features(
            model,
            model_input,
            feature_columns,
        )

        prediction = float(model.predict(model_input)[0])
        contribution_row = booster.predict(
            transformed,
            pred_contrib=True,
        )[0]

        base_value = float(contribution_row[-1])
        shap_values = contribution_row[:-1]

        for feature, value, contribution in zip(
            transformed_names,
            transformed[0],
            shap_values,
        ):
            output_records.append(
                {
                    "forecast_step": step,
                    "timestamp": row[TIMESTAMP_COLUMN],
                    "feature": feature,
                    "feature_value_after_imputation": float(value),
                    "tree_shap_contribution": float(contribution),
                    "base_value": base_value,
                    "raw_model_prediction": prediction,
                    "actual": float(row[TARGET_COLUMN]),
                }
            )

        if step == explain_step:
            selected_step_values = {
                "timestamp": row[TIMESTAMP_COLUMN],
                "prediction": prediction,
                "actual": float(row[TARGET_COLUMN]),
                "base_value": base_value,
                "features": transformed_names,
                "contributions": shap_values,
                "feature_values": transformed[0],
            }

        # Внутри прогнозного окна используем собственный прогноз,
        # не фактическое значение целевого показателя.
        history.append(max(0.0, prediction))

    local_table = pd.DataFrame(output_records)
    local_table.to_csv(
        output_dir / "local_forecast_contributions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if selected_step_values is None:
        raise RuntimeError("Не удалось сформировать локальное объяснение.")

    local_explanation = pd.DataFrame(
        {
            "feature": selected_step_values["features"],
            "feature_value_after_imputation": selected_step_values[
                "feature_values"
            ],
            "tree_shap_contribution": selected_step_values[
                "contributions"
            ],
        }
    )
    local_explanation["absolute_contribution"] = (
        local_explanation["tree_shap_contribution"].abs()
    )
    local_explanation = local_explanation.sort_values(
        "absolute_contribution",
        ascending=False,
    )
    local_explanation.to_csv(
        output_dir / "selected_forecast_explanation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top_local = (
        local_explanation.head(20)
        .sort_values("tree_shap_contribution")
    )
    local_figure = px.bar(
        top_local,
        x="tree_shap_contribution",
        y="feature",
        orientation="h",
        color="tree_shap_contribution",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        title=(
            f"Объяснение прогноза для {airport}, "
            f"{selected_step_values['timestamp']:%Y-%m-%d %H:%M}"
        ),
        labels={
            "tree_shap_contribution": "Вклад в прогноз, поездки/час",
            "feature": "Признак",
        },
    )
    local_figure.add_vline(x=0, line_color="#333333", line_width=1)
    local_figure.add_annotation(
        text=(
            f"Прогноз модели: "
            f"{selected_step_values['prediction']:.2f}; "
            f"факт: {selected_step_values['actual']:.2f}; "
            f"базовое значение: "
            f"{selected_step_values['base_value']:.2f}"
        ),
        xref="paper",
        yref="paper",
        x=0,
        y=1.12,
        showarrow=False,
        align="left",
    )
    save_html_figure(
        local_figure,
        output_dir / "selected_forecast_explanation.html",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Интерпретация модели LightGBM: важность признаков, "
            "зависимости и объяснение выбранного прогноза."
        )
    )
    parser.add_argument(
        "--airport",
        choices=["ORD", "MDW"],
        default="ORD",
        help="Аэропорт для локального объяснения.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help=(
            "Начало 24-часового окна, например "
            "'2026-04-21 14:00'. По умолчанию используется начало Test."
        ),
    )
    parser.add_argument(
        "--explain-step",
        type=int,
        default=1,
        help="Какой час внутри окна объяснить: от 1 до 24.",
    )
    parser.add_argument(
        "--dependence-feature",
        type=str,
        default="taxi_lag_1h",
        help="Признак для графика зависимости.",
    )
    parser.add_argument(
        "--permutation-sample-size",
        type=int,
        default=2000,
        help="Размер выборки для permutation importance.",
    )
    parser.add_argument(
        "--shap-sample-size",
        type=int,
        default=5000,
        help="Максимум строк на период для расчёта средней важности TreeSHAP.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()

    model_path, metadata_path, run_dir = find_latest_lightgbm_run()
    metadata = read_metadata(metadata_path)
    data = read_features()

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Не найдена таблица признаков: {FEATURES_PATH}"
        )

    feature_columns = metadata["features"]
    missing_features = set(feature_columns).difference(data.columns)
    if missing_features:
        raise ValueError(
            "В таблице отсутствуют признаки модели: "
            + ", ".join(sorted(missing_features))
        )

    model = joblib.load(model_path)
    test_start = pd.Timestamp(metadata["split"]["test_start"])

    train_validation_rows = data[
        data[TIMESTAMP_COLUMN] < test_start
    ].copy()
    test_rows = data[data[TIMESTAMP_COLUMN] >= test_start].copy()
    if train_validation_rows.empty or test_rows.empty:
        raise ValueError(
            "Не удалось разделить данные на Train + Validation и Test."
        )

    forecast_start = (
        pd.Timestamp(args.start)
        if args.start
        else test_start
    )
    if forecast_start < test_start:
        raise ValueError(
            "Для локального объяснения выберите время из тестового периода."
        )

    output_dir = run_dir / "interpretability"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Модель: %s", model_path)
    logging.info("Test начинается: %s", test_start)
    logging.info("Считаю глобальную важность и сравнение периодов.")

    save_importance_outputs(
        model=model,
        data=data,
        train_validation_rows=train_validation_rows,
        test_rows=test_rows,
        feature_columns=feature_columns,
        output_dir=output_dir,
        permutation_sample_size=args.permutation_sample_size,
        shap_sample_size=args.shap_sample_size,
        dependence_feature=args.dependence_feature,
    )

    logging.info(
        "Объясняю 24-часовой прогноз: аэропорт %s, начало %s, шаг %d.",
        args.airport,
        forecast_start,
        args.explain_step,
    )
    explain_recursive_forecast(
        model=model,
        data=data,
        feature_columns=feature_columns,
        airport=args.airport,
        forecast_start=forecast_start,
        explain_step=args.explain_step,
        output_dir=output_dir,
    )

    explanation_metadata = {
        "model_run_id": run_dir.name,
        "model_path": str(model_path),
        "test_start": test_start.isoformat(),
        "permutation_scoring": (
            "Одношаговая оценка на тестовых строках с наблюдаемыми лагами; "
            "используется только для интерпретации, не для подбора модели."
        ),
        "period_comparison": (
            "Средние абсолютные TreeSHAP-вклады Train + Validation "
            "сравниваются с Test."
        ),
        "local_explanation": {
            "airport": args.airport,
            "forecast_start": forecast_start.isoformat(),
            "explained_step": args.explain_step,
            "horizon_hours": HORIZON_HOURS,
        },
        "random_state": RANDOM_STATE,
    }
    (output_dir / "explanation_metadata.json").write_text(
        json.dumps(explanation_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logging.info("Интерпретация завершена: %s", output_dir)
    logging.info("Журнал: %s", LOG_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Интерпретация завершилась ошибкой.")
        raise