from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
FEATURES_PATH = (
    ROOT
    / "marts"
    / "hourly_airport_demand"
    / "forecast_training_features.csv"
)
RUNS_ROOT = ROOT / "model_runs"
LOG_PATH = ROOT / "run_history" / "app_monitoring.log"

TARGET_COLUMN = "trip_count"
TIMESTAMP_COLUMN = "target_timestamp_local"
AIRPORT_COLUMN = "airport"

HORIZON_HOURS = 24
PSI_REVIEW_THRESHOLD = 0.25
PSI_HEATMAP_MAX = 0.5
MIN_COMPLETE_WEEK_ROWS = 144

LAG_COLUMNS = {
    "taxi_lag_1h": 1,
    "taxi_lag_24h": 24,
    "taxi_lag_168h": 168,
}
ROLLING_COLUMNS = {
    "taxi_rolling_mean_24h": 24,
    "taxi_rolling_mean_168h": 168,
}

DAY_NAMES = [
    "Понедельник",
    "Вторник",
    "Среда",
    "Четверг",
    "Пятница",
    "Суббота",
    "Воскресенье",
]
AIRPORT_NAMES = {
    "ORD": "O'Hare (ORD)",
    "MDW": "Midway (MDW)",
}
FEATURE_LABELS = {
    "taxi_lag_1h": "Спрос 1 час назад",
    "taxi_lag_24h": "Спрос 24 часа назад",
    "taxi_lag_168h": "Спрос 168 часов назад",
    "taxi_rolling_mean_24h": "Средний спрос за 24 часа",
    "taxi_rolling_mean_168h": "Средний спрос за 168 часов",
    "scheduled_arrivals_lag_3m": "Плановые прилёты, лаг 3 месяца",
    "actual_arrivals_lag_3m": "Фактические прилёты, лаг 3 месяца",
    "cancelled_flights_lag_3m": "Отмены рейсов, лаг 3 месяца",
    "diverted_flights_lag_3m": "Перенаправления рейсов, лаг 3 месяца",
    "mean_arrival_delay_lag_3m": "Средняя задержка, лаг 3 месяца",
    "hour_sin": "Час суток, sin",
    "hour_cos": "Час суток, cos",
    "weekday_sin": "День недели, sin",
    "weekday_cos": "День недели, cos",
    "month_sin": "Месяц, sin",
    "month_cos": "Месяц, cos",
    "day_of_year_sin": "День года, sin",
    "day_of_year_cos": "День года, cos",
    "is_weekend": "Выходной",
    "is_us_federal_holiday": "Праздник США",
    "bts_lag_3m_available": "Доступность BTS-лага",
    "mean_arrival_delay_lag_3m_missing": "Нет данных о задержке",
    "airport_ord": "Аэропорт ORD",
}


def configure_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
        ],
        force=True,
    )


def find_model_run() -> tuple[Path, Path, Path]:
    """Находит последнюю папку запуска с моделью, прогнозами и метаданными."""
    if not RUNS_ROOT.exists():
        raise FileNotFoundError(f"Не найдена папка запусков: {RUNS_ROOT}")

    run_directories = sorted(
        [path for path in RUNS_ROOT.iterdir() if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )

    for run_dir in run_directories:
        model_path = run_dir / "lightgbm.joblib"
        metadata_path = run_dir / "run_metadata.json"
        predictions_path = run_dir / "recursive_predictions.csv"

        if (
            model_path.exists()
            and metadata_path.exists()
            and predictions_path.exists()
        ):
            return model_path, metadata_path, predictions_path

    raise FileNotFoundError(
        "Не найдены LightGBM-модель, run_metadata.json и "
        "recursive_predictions.csv. Сначала выполните "
        "train_and_compare_models.py."
    )


@st.cache_resource(show_spinner="Загружаю LightGBM...")
def load_model(model_path: str) -> Any:
    return joblib.load(model_path)


@st.cache_data(show_spinner="Загружаю метаданные...")
def load_metadata(metadata_path: str) -> dict[str, Any]:
    with Path(metadata_path).open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner="Загружаю результаты...")
def load_predictions(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)

    required_columns = {
        "model",
        "airport",
        "split",
        "timestamp",
        "actual",
        "prediction",
        "forecast_step",
        "window_start",
    }
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(
            "В таблице прогнозов отсутствуют столбцы: "
            + ", ".join(sorted(missing))
        )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"], errors="coerce"
    )
    data["window_start"] = pd.to_datetime(
        data["window_start"], errors="coerce"
    )
    data["actual"] = pd.to_numeric(data["actual"], errors="coerce")
    data["prediction"] = pd.to_numeric(
        data["prediction"], errors="coerce"
    )
    data = data.dropna(
        subset=["timestamp", "window_start", "actual", "prediction"]
    )
    data[AIRPORT_COLUMN] = data[AIRPORT_COLUMN].astype(str)
    return data


@st.cache_data(show_spinner="Загружаю таблицу признаков...")
def load_features(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
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


def show_error(message: str) -> None:
    st.error(message)
    st.info(
        "Проверьте таблицу признаков и файлы последнего запуска обучения. "
        "Если модель отсутствует, выполните train_and_compare_models.py."
    )
    st.stop()


def available_forecast_starts(
    airport_data: pd.DataFrame,
    test_start: pd.Timestamp,
) -> list[pd.Timestamp]:
    """Возвращает все часы Test с полным непрерывным окном на 24 часа."""
    test_rows = (
        airport_data[airport_data[TIMESTAMP_COLUMN] >= test_start]
        .sort_values(TIMESTAMP_COLUMN)
        .reset_index(drop=True)
    )

    starts: list[pd.Timestamp] = []
    for start_index in range(0, len(test_rows) - HORIZON_HOURS + 1):
        window = test_rows.iloc[
            start_index : start_index + HORIZON_HOURS
        ]
        if len(window) != HORIZON_HOURS:
            continue

        times = window[TIMESTAMP_COLUMN].to_numpy()
        if np.all(np.diff(times) == np.timedelta64(1, "h")):
            starts.append(pd.Timestamp(window.iloc[0][TIMESTAMP_COLUMN]))

    return starts


def forecast_24_hours(
    model: Any,
    airport_data: pd.DataFrame,
    forecast_start: pd.Timestamp,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Строит последовательный прогноз на 24 часа."""
    airport_data = airport_data.sort_values(TIMESTAMP_COLUMN).reset_index(
        drop=True
    )
    positions = airport_data.index[
        airport_data[TIMESTAMP_COLUMN] == forecast_start
    ].tolist()

    if not positions:
        raise ValueError("Выбранное время отсутствует в таблице признаков.")

    start_index = positions[0]
    future_rows = airport_data.iloc[
        start_index : start_index + HORIZON_HOURS
    ].copy()
    if len(future_rows) != HORIZON_HOURS:
        raise ValueError("Для выбранного времени нет полного окна на 24 часа.")

    timestamps = future_rows[TIMESTAMP_COLUMN].to_numpy()
    if not np.all(np.diff(timestamps) == np.timedelta64(1, "h")):
        raise ValueError("В 24-часовом окне есть пропуск или скачок времени.")

    history = (
        airport_data.iloc[:start_index][TARGET_COLUMN]
        .astype(float)
        .tolist()
    )
    if len(history) < 168:
        raise ValueError(
            "Для выбранного времени недостаточно истории на 168 часов."
        )

    records: list[dict[str, Any]] = []
    for _, row in future_rows.iterrows():
        feature_row = row[feature_columns].copy()

        for column, lag in LAG_COLUMNS.items():
            feature_row[column] = history[-lag]

        for column, window in ROLLING_COLUMNS.items():
            feature_row[column] = float(np.mean(history[-window:]))

        model_input = pd.DataFrame(
            [feature_row.to_dict()],
            columns=feature_columns,
        )
        prediction = max(0.0, float(model.predict(model_input)[0]))

        records.append(
            {
                "timestamp": row[TIMESTAMP_COLUMN],
                "actual": float(row[TARGET_COLUMN]),
                "prediction": prediction,
            }
        )
        history.append(prediction)

    return pd.DataFrame(records)


def week_start(series: pd.Series) -> pd.Series:
    """Преобразует временные метки в даты начала календарной недели."""
    return series.dt.to_period("W-SUN").dt.start_time


def weekly_error_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Считает недельные MAE и WAPE по сохранённым прогнозам."""
    data = predictions.copy()
    data["week_start"] = week_start(data["timestamp"])
    data["absolute_error"] = (data["actual"] - data["prediction"]).abs()

    result = (
        data.groupby("week_start")
        .agg(
            MAE=("absolute_error", "mean"),
            actual_total=("actual", "sum"),
            absolute_error_total=("absolute_error", "sum"),
            forecast_rows=("actual", "size"),
        )
        .reset_index()
    )
    result["WAPE_percent"] = (
        result["absolute_error_total"]
        / result["actual_total"].abs().clip(lower=1)
        * 100
    )
    result["complete_week"] = (
        result["forecast_rows"] >= MIN_COMPLETE_WEEK_ROWS
    )
    return result.sort_values("week_start").reset_index(drop=True)


def calculate_psi(
    reference: pd.Series,
    current: pd.Series,
    max_bins: int = 10,
) -> float:
    """
    Рассчитывает PSI для категориальных или числовых значений.
    Пропуски учитываются как отдельная категория.
    """
    reference = pd.to_numeric(reference, errors="coerce")
    current = pd.to_numeric(current, errors="coerce")

    reference_non_missing = reference.dropna()
    if reference_non_missing.empty:
        return float("nan")

    unique_reference = reference_non_missing.nunique()

    if unique_reference <= 10:
        reference_labels = reference.astype("string").fillna("__MISSING__")
        current_labels = current.astype("string").fillna("__MISSING__")
        labels = sorted(
            set(reference_labels.tolist())
            | set(current_labels.tolist())
        )
        reference_counts = reference_labels.value_counts().reindex(
            labels, fill_value=0
        )
        current_counts = current_labels.value_counts().reindex(
            labels, fill_value=0
        )
    else:
        quantiles = np.linspace(0, 1, max_bins + 1)
        boundaries = np.unique(
            np.quantile(reference_non_missing.to_numpy(), quantiles)
        )

        if len(boundaries) <= 2:
            edges = np.array([-np.inf, np.inf])
        else:
            edges = np.concatenate(
                ([-np.inf], boundaries[1:-1], [np.inf])
            )

        reference_bins = pd.cut(
            reference,
            bins=edges,
            labels=False,
            include_lowest=True,
        ).astype("Int64")
        current_bins = pd.cut(
            current,
            bins=edges,
            labels=False,
            include_lowest=True,
        ).astype("Int64")

        reference_labels = reference_bins.astype("string").fillna(
            "__MISSING__"
        )
        current_labels = current_bins.astype("string").fillna(
            "__MISSING__"
        )
        labels = sorted(
            set(reference_labels.tolist())
            | set(current_labels.tolist())
        )
        reference_counts = reference_labels.value_counts().reindex(
            labels, fill_value=0
        )
        current_counts = current_labels.value_counts().reindex(
            labels, fill_value=0
        )

    bin_count = max(len(labels), 1)
    reference_share = (
        (reference_counts.astype(float) + 0.5)
        / (reference_counts.sum() + 0.5 * bin_count)
    )
    current_share = (
        (current_counts.astype(float) + 0.5)
        / (current_counts.sum() + 0.5 * bin_count)
    )

    psi = (
        (current_share - reference_share)
        * np.log(current_share / reference_share)
    ).sum()
    return float(psi)


def choose_monitored_features(
    feature_columns: list[str],
    run_dir: Path,
) -> list[str]:
    """Берёт важные признаки и добавляет показатели авиапотока."""
    selected: list[str] = []
    importance_path = run_dir / "interpretability" / "feature_importance.csv"

    if importance_path.exists():
        importance = pd.read_csv(importance_path)
        score_column = "mean_abs_treeshap_test"

        if {"feature", score_column}.issubset(importance.columns):
            top_features = (
                importance[importance["feature"].isin(feature_columns)]
                .sort_values(score_column, ascending=False)["feature"]
                .head(8)
                .tolist()
            )
            selected.extend(top_features)

    for feature in [
        "scheduled_arrivals_lag_3m",
        "actual_arrivals_lag_3m",
        "hour_cos",
        "taxi_lag_1h",
    ]:
        if feature in feature_columns:
            selected.append(feature)

    return list(dict.fromkeys(selected))[:10]


def weekly_feature_drift(
    airport_features: pd.DataFrame,
    reference_end: pd.Timestamp,
    monitored_features: list[str],
) -> pd.DataFrame:
    """Считает PSI каждой полной недели Test относительно Train + Validation."""
    reference_rows = airport_features[
        airport_features[TIMESTAMP_COLUMN] < reference_end
    ].copy()
    current_rows = airport_features[
        airport_features[TIMESTAMP_COLUMN] >= reference_end
    ].copy()

    if reference_rows.empty or current_rows.empty:
        return pd.DataFrame()

    current_rows["week_start"] = week_start(current_rows[TIMESTAMP_COLUMN])
    records: list[dict[str, Any]] = []

    for week, current_week in current_rows.groupby("week_start"):
        if len(current_week) < MIN_COMPLETE_WEEK_ROWS:
            continue

        for feature in monitored_features:
            if feature not in airport_features.columns:
                continue

            reference_values = reference_rows[feature]
            current_values = current_week[feature]

            records.append(
                {
                    "week_start": week,
                    "feature": feature,
                    "psi": calculate_psi(reference_values, current_values),
                    "reference_missing_percent": (
                        pd.to_numeric(
                            reference_values, errors="coerce"
                        ).isna().mean()
                        * 100
                    ),
                    "current_missing_percent": (
                        pd.to_numeric(
                            current_values, errors="coerce"
                        ).isna().mean()
                        * 100
                    ),
                    "feature_rows": len(current_week),
                }
            )

    return pd.DataFrame(records)


def has_two_consecutive_exceedances(
    weekly_test: pd.DataFrame,
    threshold: float,
) -> bool:
    complete = (
        weekly_test[weekly_test["complete_week"]]
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    if len(complete) < 2:
        return False

    for index in range(1, len(complete)):
        previous_week = complete.loc[index - 1, "week_start"]
        current_week = complete.loc[index, "week_start"]
        are_adjacent = (
            current_week - previous_week == pd.Timedelta(days=7)
        )
        both_above = (
            complete.loc[index - 1, "MAE"] > threshold
            and complete.loc[index, "MAE"] > threshold
        )
        if are_adjacent and both_above:
            return True

    return False


def render_forecast_tab(
    model: Any,
    features: pd.DataFrame,
    feature_columns: list[str],
    test_start: pd.Timestamp,
    selected_airport: str,
    airport_labels: dict[str, str],
) -> None:
    airport_data = (
        features[features[AIRPORT_COLUMN] == selected_airport]
        .sort_values(TIMESTAMP_COLUMN)
        .reset_index(drop=True)
    )
    starts = available_forecast_starts(airport_data, test_start)

    if not starts:
        st.warning(
            "Для аэропорта нет полного окна прогноза в тестовом периоде."
        )
        return

    available_dates = sorted({timestamp.date() for timestamp in starts})
    date_column, hour_column = st.columns([1, 1])

    with date_column:
        selected_date = st.selectbox(
            "Дата начала прогноза",
            options=available_dates,
            format_func=lambda value: value.strftime("%d.%m.%Y"),
            key="forecast_date",
        )

    hours_for_date = [
        timestamp
        for timestamp in starts
        if timestamp.date() == selected_date
    ]

    with hour_column:
        selected_start = st.selectbox(
            "Час",
            options=hours_for_date,
            format_func=lambda value: value.strftime("%H:%M"),
            key="forecast_hour",
        )

    try:
        scenario = forecast_24_hours(
            model=model,
            airport_data=airport_data,
            forecast_start=selected_start,
            feature_columns=feature_columns,
        )
    except (ValueError, IndexError) as error:
        st.error(str(error))
        return

    profile_history = airport_data[
        airport_data[TIMESTAMP_COLUMN] < test_start
    ].copy()
    profile_history["weekday"] = (
        profile_history[TIMESTAMP_COLUMN].dt.dayofweek
    )
    profile_history["hour"] = profile_history[TIMESTAMP_COLUMN].dt.hour

    calendar_reference = profile_history.groupby(
        ["weekday", "hour"]
    )[TARGET_COLUMN].mean()

    scenario["weekday"] = scenario["timestamp"].dt.dayofweek
    scenario["hour"] = scenario["timestamp"].dt.hour
    scenario["calendar_average"] = [
        calendar_reference.get((weekday, hour), np.nan)
        for weekday, hour in zip(scenario["weekday"], scenario["hour"])
    ]
    scenario["forecast_cumulative"] = scenario["prediction"].cumsum()
    scenario["actual_cumulative"] = scenario["actual"].cumsum()

    forecast_total = float(scenario["prediction"].sum())
    actual_total = float(scenario["actual"].sum())
    scenario_mae = float(
        (scenario["actual"] - scenario["prediction"]).abs().mean()
    )
    peak_row = scenario.loc[scenario["prediction"].idxmax()]
    peak_hour = pd.Timestamp(peak_row["timestamp"]).strftime("%H:%M")

    forecast_metric, actual_metric, error_metric, peak_metric = st.columns(4)
    forecast_metric.metric(
        "Прогноз за 24 часа",
        f"{forecast_total:,.0f} поездок".replace(",", " "),
    )
    actual_metric.metric(
        "Фактически за 24 часа",
        f"{actual_total:,.0f} поездок".replace(",", " "),
    )
    error_metric.metric(
        "MAE выбранного окна",
        f"{scenario_mae:.1f} поездки/час",
    )
    peak_metric.metric(
        "Пиковый прогноз",
        f"{peak_hour} · {peak_row['prediction']:.0f}",
    )

    st.subheader("Почасовой профиль выбранного окна")
    st.caption(
        f"{airport_labels[selected_airport]} · "
        f"{selected_start:%d.%m.%Y, %H:%M} · 24 часа"
    )

    hourly_chart = go.Figure()
    hourly_chart.add_trace(
        go.Bar(
            x=scenario["timestamp"],
            y=scenario["prediction"],
            name="Прогноз",
            marker_color="#2878B5",
            opacity=0.9,
        )
    )
    hourly_chart.add_trace(
        go.Scatter(
            x=scenario["timestamp"],
            y=scenario["actual"],
            name="Фактический спрос",
            mode="lines+markers",
            line={"color": "#F28E2B", "width": 2},
            marker={"size": 5},
        )
    )
    hourly_chart.add_trace(
        go.Scatter(
            x=scenario["timestamp"],
            y=scenario["calendar_average"],
            name="Типичный календарный уровень",
            mode="lines",
            line={"color": "#2E8B57", "width": 2, "dash": "dash"},
        )
    )
    hourly_chart.update_layout(
        height=390,
        margin={"l": 10, "r": 10, "t": 25, "b": 10},
        xaxis_title="Час",
        yaxis_title="Поездки",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        hovermode="x unified",
    )
    hourly_chart.update_xaxes(tickformat="%d.%m %H:%M")
    st.plotly_chart(hourly_chart, use_container_width=True)

    st.subheader("Накопленный объём поездок")
    st.caption("Суммарный спрос, накопленный от начала выбранного окна.")

    cumulative_chart = go.Figure()
    cumulative_chart.add_trace(
        go.Scatter(
            x=scenario["timestamp"],
            y=scenario["forecast_cumulative"],
            name="Прогноз",
            mode="lines+markers",
            line={"color": "#2878B5", "width": 3},
            marker={"size": 5},
        )
    )
    cumulative_chart.add_trace(
        go.Scatter(
            x=scenario["timestamp"],
            y=scenario["actual_cumulative"],
            name="Фактически",
            mode="lines+markers",
            line={"color": "#F28E2B", "width": 2},
            marker={"size": 4},
        )
    )
    cumulative_chart.update_layout(
        height=310,
        margin={"l": 10, "r": 10, "t": 15, "b": 10},
        xaxis_title="Час",
        yaxis_title="Накоплено поездок",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        hovermode="x unified",
    )
    cumulative_chart.update_xaxes(tickformat="%d.%m %H:%M")
    st.plotly_chart(cumulative_chart, use_container_width=True)

    st.caption(
        "Результаты воспроизводятся на историческом тестовом периоде. "
        "Это не оперативный прогноз на текущую дату."
    )


def render_monitoring_tab(
    all_predictions: pd.DataFrame,
    features: pd.DataFrame,
    feature_columns: list[str],
    run_dir: Path,
    test_start: pd.Timestamp,
    selected_airport: str,
) -> None:
    selected_predictions = all_predictions[
        (all_predictions["model"] == "lightgbm")
        & (all_predictions[AIRPORT_COLUMN] == selected_airport)
    ].copy()

    validation_rows = selected_predictions[
        selected_predictions["split"] == "validation"
    ]
    test_rows = selected_predictions[
        selected_predictions["split"] == "test"
    ]

    if validation_rows.empty or test_rows.empty:
        st.warning(
            "В файле запуска нет прогнозов LightGBM одновременно "
            "для validation и test."
        )
        return

    validation_weekly = weekly_error_metrics(validation_rows)
    test_weekly = weekly_error_metrics(test_rows)

    validation_complete = validation_weekly[
        validation_weekly["complete_week"]
    ]
    test_complete = test_weekly[test_weekly["complete_week"]]

    if validation_complete.empty or test_complete.empty:
        st.warning("Недостаточно полных недель для мониторинга качества.")
        return

    validation_mean = float(validation_complete["MAE"].mean())
    validation_std = float(validation_complete["MAE"].std(ddof=1))
    if not np.isfinite(validation_std):
        validation_std = 0.0

    quality_threshold = validation_mean + 2 * validation_std
    latest_week = test_complete.iloc[-1]
    latest_max_psi = float("nan")
    latest_drift_feature = "—"

    airport_features = features[
        features[AIRPORT_COLUMN] == selected_airport
    ].copy()
    monitored_features = choose_monitored_features(
        feature_columns,
        run_dir,
    )

    drift_table = weekly_feature_drift(
        airport_features=airport_features,
        reference_end=test_start,
        monitored_features=monitored_features,
    )

    if not drift_table.empty:
        latest_week_start = latest_week["week_start"]
        latest_drift = drift_table[
            drift_table["week_start"] == latest_week_start
        ].sort_values("psi", ascending=False)

        if not latest_drift.empty:
            latest_max_psi = float(latest_drift.iloc[0]["psi"])
            latest_drift_feature = str(latest_drift.iloc[0]["feature"])

    quality_column, baseline_column, drift_column = st.columns(3)
    quality_column.metric(
        "MAE последней полной недели",
        f"{latest_week['MAE']:.2f} поездки/час",
    )
    baseline_column.metric(
        "Порог проверки качества",
        f"{quality_threshold:.2f} поездки/час",
        help=(
            "Средняя недельная MAE validation + 2 стандартных "
            "отклонения по полным неделям."
        ),
    )
    drift_column.metric(
        "Максимальный PSI за последнюю неделю",
        f"{latest_max_psi:.3f}" if np.isfinite(latest_max_psi) else "н/д",
        help=f"Наибольший сдвиг: {latest_drift_feature}",
    )

    retraining_signal = has_two_consecutive_exceedances(
        test_weekly,
        quality_threshold,
    )

    if retraining_signal:
        st.error(
            "Сигнал: MAE выше порога две полные недели подряд. "
            "Следует проверить данные и рассмотреть переобучение."
        )
    elif latest_week["MAE"] > quality_threshold:
        st.warning(
            "MAE последней полной недели выше порога. "
            "Нужно наблюдать за следующими неделями."
        )
    else:
        st.success(
            "MAE последней полной недели находится в базовом коридоре."
        )

    if np.isfinite(latest_max_psi):
        if latest_max_psi >= PSI_REVIEW_THRESHOLD:
            st.warning(
                "Обнаружен заметный сдвиг распределения входных признаков. "
                "Это сигнал для проверки данных; сам по себе он не запускает "
                "переобучение."
            )
        else:
            st.info(
                "Для отслеживаемых признаков PSI последней полной недели "
                "ниже установленного порога проверки."
            )

    st.subheader("Качество прогноза по неделям")
    st.caption(
        "Недельная MAE на тесте сопоставляется с порогом, рассчитанным "
        "по полным неделям validation."
    )

    quality_chart = go.Figure()
    quality_chart.add_trace(
        go.Scatter(
            x=test_complete["week_start"],
            y=test_complete["MAE"],
            name="MAE Test",
            mode="lines+markers",
            line={"color": "#2878B5", "width": 3},
            marker={"size": 7},
        )
    )
    quality_chart.add_hline(
        y=quality_threshold,
        line_dash="dash",
        line_color="#C43D3D",
        annotation_text="Порог проверки",
    )
    quality_chart.update_layout(
        height=330,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        xaxis_title="Начало недели",
        yaxis_title="MAE, поездки/час",
        hovermode="x unified",
    )
    st.plotly_chart(quality_chart, use_container_width=True)

    wape_chart = px.line(
        test_complete,
        x="week_start",
        y="WAPE_percent",
        markers=True,
        title="Относительная ошибка по неделям",
        labels={
            "week_start": "Начало недели",
            "WAPE_percent": "WAPE, %",
        },
    )
    wape_chart.update_layout(
        height=300,
        margin={"l": 10, "r": 10, "t": 35, "b": 10},
    )
    st.plotly_chart(wape_chart, use_container_width=True)

    st.subheader("Сдвиг входных признаков")
    st.caption(
        "PSI сравнивает распределение признаков каждой полной недели Test "
        "с Train + Validation. Цветовая шкала ограничена значением 0,5; "
        "более высокие значения отображаются максимальным цветом. "
        "Точные значения приведены в таблице ниже."
    )

    if drift_table.empty:
        st.info("Недостаточно данных для расчёта PSI.")
        return

    drift_pivot = drift_table.pivot(
        index="feature",
        columns="week_start",
        values="psi",
    ).sort_index(axis=1)

    drift_pivot = drift_pivot.reindex(index=monitored_features)
    feature_labels = [
        FEATURE_LABELS.get(name, name)
        for name in drift_pivot.index
    ]
    week_labels = [
        pd.Timestamp(value).strftime("%d.%m.%Y")
        for value in drift_pivot.columns
    ]

    drift_figure = go.Figure(
        data=go.Heatmap(
            z=drift_pivot.to_numpy(dtype=float),
            x=week_labels,
            y=feature_labels,
            zmin=0,
            zmax=PSI_HEATMAP_MAX,
            colorscale="YlOrRd",
            colorbar={
                "title": "PSI",
                "tickmode": "array",
                "tickvals": [0, 0.1, PSI_REVIEW_THRESHOLD, PSI_HEATMAP_MAX],
                "ticktext": ["0", "0,10", "0,25", "0,50+"],
            },
            hovertemplate=(
                "Неделя: %{x}<br>"
                "Признак: %{y}<br>"
                "PSI: %{z:.3f}<extra></extra>"
            ),
        )
    )
    visible_tick_positions = list(range(0, len(week_labels), 2))
    if week_labels and visible_tick_positions[-1] != len(week_labels) - 1:
        visible_tick_positions.append(len(week_labels) - 1)

    drift_figure.update_layout(
        height=max(330, 34 * len(feature_labels)),
        margin={"l": 10, "r": 10, "t": 15, "b": 70},
        xaxis_title="Начало недели",
        yaxis_title="Признак",
    )
    drift_figure.update_xaxes(
        type="category",
        tickmode="array",
        tickvals=[week_labels[index] for index in visible_tick_positions],
        ticktext=[week_labels[index] for index in visible_tick_positions],
        tickangle=-45,
    )
    st.plotly_chart(drift_figure, use_container_width=True)

    drift_summary = (
        drift_table.groupby("feature", as_index=False)
        .agg(
            max_psi=("psi", "max"),
            mean_psi=("psi", "mean"),
            reference_missing_percent=(
                "reference_missing_percent",
                "mean",
            ),
            test_missing_percent=("current_missing_percent", "mean"),
        )
        .sort_values("max_psi", ascending=False)
    )
    drift_summary["feature"] = drift_summary["feature"].map(
        lambda name: FEATURE_LABELS.get(name, name)
    )
    st.dataframe(
        drift_summary.rename(
            columns={
                "feature": "Признак",
                "max_psi": "Максимальный PSI",
                "mean_psi": "Средний PSI",
                "reference_missing_percent": (
                    "Пропуски, Train + Validation, %"
                ),
                "test_missing_percent": "Пропуски, Test, %",
            }
        ).round(3),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Качество считается по сохранённым рекурсивным прогнозам. "
        "PSI рассчитан по наблюдаемым строкам таблицы признаков; "
        "в 24-часовом прогнозе внутри окна лаги заменяются предыдущими "
        "прогнозами."
    )


def render_calendar_tab(
    features: pd.DataFrame,
    test_start: pd.Timestamp,
    selected_airport: str,
    airport_labels: dict[str, str],
) -> None:
    airport_data = features[
        (features[AIRPORT_COLUMN] == selected_airport)
        & (features[TIMESTAMP_COLUMN] < test_start)
    ].copy()

    if airport_data.empty:
        st.warning("Нет данных для календарного профиля.")
        return

    airport_data["weekday"] = (
        airport_data[TIMESTAMP_COLUMN].dt.dayofweek
    )
    airport_data["hour"] = airport_data[TIMESTAMP_COLUMN].dt.hour

    profile = (
        airport_data.groupby(["weekday", "hour"])[TARGET_COLUMN]
        .mean()
        .unstack("hour")
        .reindex(index=range(7), columns=range(24))
    )

    st.subheader(f"Календарный профиль · {airport_labels[selected_airport]}")
    st.caption(
        "Среднее фактическое число поездок по часу суток и дню недели. "
        "Используются только Train + Validation."
    )

    heatmap = px.imshow(
        profile.to_numpy(),
        x=[f"{hour:02d}:00" for hour in range(24)],
        y=DAY_NAMES,
        aspect="auto",
        color_continuous_scale="Blues",
        labels={
            "x": "Час суток",
            "y": "День недели",
            "color": "Поездок/час",
        },
    )
    heatmap.update_layout(
        height=360,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
    )
    heatmap.update_xaxes(side="bottom")
    st.plotly_chart(heatmap, use_container_width=True)


def main() -> None:
    configure_logging()
    st.set_page_config(
        page_title="Airport Demand Brief",
        page_icon="✈️",
        layout="wide",
    )

    try:
        model_path, metadata_path, predictions_path = find_model_run()
        metadata = load_metadata(str(metadata_path))
        model = load_model(str(model_path))
        predictions = load_predictions(str(predictions_path))

        if not FEATURES_PATH.exists():
            raise FileNotFoundError(
                f"Не найдена таблица признаков: {FEATURES_PATH}"
            )
        features = load_features(str(FEATURES_PATH))

        feature_columns = metadata["features"]
        test_start = pd.Timestamp(metadata["split"]["test_start"])

    except (FileNotFoundError, ValueError, KeyError, OSError) as error:
        show_error(str(error))
        return

    missing_features = set(feature_columns).difference(features.columns)
    if missing_features:
        show_error(
            "В таблице признаков отсутствуют признаки модели: "
            + ", ".join(sorted(missing_features))
        )

    st.markdown(
        """
        <style>
          .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1440px;
          }
          .topline {
            color: #6b7280;
            font-size: 0.82rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
          }
          .model-chip {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: #e8f3ff;
            color: #145da0;
            font-weight: 700;
            font-size: 0.85rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="topline">Chicago · Airport demand</div>',
        unsafe_allow_html=True,
    )
    st.title("Сводка спроса на такси")
    st.markdown(
        '<span class="model-chip">Активная модель · LightGBM</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Историческая демонстрация на данных до августа 2026 года. "
        "Мониторинг не получает новые данные в реальном времени."
    )

    airport_options = sorted(
        features[AIRPORT_COLUMN].astype(str).unique().tolist()
    )
    airport_labels = {
        airport: AIRPORT_NAMES.get(airport, airport)
        for airport in airport_options
    }

    selected_airport = st.radio(
        "Аэропорт",
        options=airport_options,
        format_func=lambda value: airport_labels[value],
        horizontal=True,
    )

    forecast_tab, monitoring_tab, calendar_tab = st.tabs(
        ["Прогноз", "Мониторинг", "Календарный профиль"]
    )

    with forecast_tab:
        render_forecast_tab(
            model=model,
            features=features,
            feature_columns=feature_columns,
            test_start=test_start,
            selected_airport=selected_airport,
            airport_labels=airport_labels,
        )

    with monitoring_tab:
        st.subheader(
            f"Мониторинг модели · {airport_labels[selected_airport]}"
        )
        st.caption(
            f"Версия модели: {model_path.parent.name} · "
            f"Test начинается {test_start:%d.%m.%Y %H:%M}"
        )
        render_monitoring_tab(
            all_predictions=predictions,
            features=features,
            feature_columns=feature_columns,
            run_dir=model_path.parent,
            test_start=test_start,
            selected_airport=selected_airport,
        )

    with calendar_tab:
        render_calendar_tab(
            features=features,
            test_start=test_start,
            selected_airport=selected_airport,
            airport_labels=airport_labels,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Ошибка при запуске интерфейса.")
        st.error(
            "Не удалось отобразить приложение. Подробности записаны "
            "в run_history/app_monitoring.log."
        )
        raise