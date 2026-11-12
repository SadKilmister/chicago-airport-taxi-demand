from __future__ import annotations

import json
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


st.set_page_config(
    page_title="Airport Demand Brief",
    page_icon="✈️",
    layout="wide",
)


def find_model_run() -> tuple[Path, Path]:
    """Находит последний запуск с моделью LightGBM и метаданными."""
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
        if model_path.exists() and metadata_path.exists():
            return model_path, metadata_path

    raise FileNotFoundError(
        "Не найдены lightgbm.joblib и run_metadata.json. "
        "Сначала выполните train_and_compare_models.py."
    )


@st.cache_resource(show_spinner="Загружаю LightGBM...")
def load_model(model_path: str) -> Any:
    return joblib.load(model_path)


@st.cache_data(show_spinner="Загружаю метаданные...")
def load_metadata(metadata_path: str) -> dict[str, Any]:
    with Path(metadata_path).open("r", encoding="utf-8") as file:
        return json.load(file)


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
    data = data.sort_values([AIRPORT_COLUMN, TIMESTAMP_COLUMN]).reset_index(
        drop=True
    )
    return data


def show_error(message: str) -> None:
    st.error(message)
    st.info(
        "Проверьте наличие таблицы признаков и обученной LightGBM-модели."
    )
    st.stop()


def available_forecast_starts(
    airport_data: pd.DataFrame,
    test_start: pd.Timestamp,
) -> list[pd.Timestamp]:
    """Часы тестового периода, для которых доступно полное окно в 24 часа."""
    test_rows = (
        airport_data[airport_data[TIMESTAMP_COLUMN] >= test_start]
        .sort_values(TIMESTAMP_COLUMN)
        .reset_index(drop=True)
    )

    starts: list[pd.Timestamp] = []

    for start_index in range(0, len(test_rows) - 23):
        window = test_rows.iloc[start_index : start_index + 24]
        if len(window) != 24:
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
    """Строит рекурсивный прогноз на 24 часа."""
    airport_data = airport_data.sort_values(TIMESTAMP_COLUMN).reset_index(
        drop=True
    )

    start_positions = airport_data.index[
        airport_data[TIMESTAMP_COLUMN] == forecast_start
    ].tolist()
    if not start_positions:
        raise ValueError("Выбранное время отсутствует в таблице признаков.")

    start_index = start_positions[0]
    forecast_rows = airport_data.iloc[start_index : start_index + 24].copy()

    if len(forecast_rows) != 24:
        raise ValueError("Для выбранного времени нет полного окна из 24 часов.")

    history = (
        airport_data.iloc[:start_index][TARGET_COLUMN]
        .astype(float)
        .tolist()
    )
    if len(history) < 168:
        raise ValueError(
            "Перед выбранным временем недостаточно истории для лага 168 часов."
        )

    result: list[dict[str, Any]] = []

    for _, row in forecast_rows.iterrows():
        feature_row = row[feature_columns].copy()

        for column, lag in LAG_COLUMNS.items():
            feature_row[column] = history[-lag]

        for column, window in ROLLING_COLUMNS.items():
            feature_row[column] = float(np.mean(history[-window:]))

        prediction = float(
            model.predict(
                pd.DataFrame(
                    [feature_row.to_dict()],
                    columns=feature_columns,
                )
            )[0]
        )
        prediction = max(0.0, prediction)

        result.append(
            {
                "timestamp": row[TIMESTAMP_COLUMN],
                "actual": float(row[TARGET_COLUMN]),
                "prediction": prediction,
            }
        )

        # Для следующего часа используется прогноз, а не фактическое значение.
        history.append(prediction)

    return pd.DataFrame(result)


try:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Не найдена таблица признаков: {FEATURES_PATH}"
        )

    model_path, metadata_path = find_model_run()
    model = load_model(str(model_path))
    metadata = load_metadata(str(metadata_path))
    features = load_features(str(FEATURES_PATH))

    feature_columns = metadata["features"]
    test_start = pd.Timestamp(metadata["split"]["test_start"])

except (FileNotFoundError, KeyError, ValueError, OSError) as error:
    show_error(str(error))


missing_features = set(feature_columns).difference(features.columns)
if missing_features:
    show_error(
        "В таблице признаков отсутствуют столбцы: "
        + ", ".join(sorted(missing_features))
    )


st.markdown(
    """
    <style>
      .block-container {
        padding-top: 1.5rem;
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
      .subtle {
        color: #667085;
        font-size: 0.92rem;
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
    "Интерфейс воспроизводит прогнозы на отложенном историческом периоде. "
    "Это не оперативный прогноз на текущую дату."
)


airport_options = sorted(features[AIRPORT_COLUMN].unique().tolist())
airport_labels = {
    airport: AIRPORT_NAMES.get(airport, airport)
    for airport in airport_options
}

airport_column, date_column, hour_column = st.columns([1.2, 1, 0.8])

with airport_column:
    selected_airport = st.selectbox(
        "Аэропорт",
        options=airport_options,
        format_func=lambda value: airport_labels[value],
    )

airport_data = (
    features[features[AIRPORT_COLUMN] == selected_airport]
    .sort_values(TIMESTAMP_COLUMN)
    .reset_index(drop=True)
)

starts = available_forecast_starts(airport_data, test_start)
if not starts:
    show_error(
        "Для выбранного аэропорта нет полного 24-часового окна "
        "в тестовом периоде."
    )

available_dates = sorted({timestamp.date() for timestamp in starts})

with date_column:
    selected_date = st.selectbox(
        "Дата начала прогноза",
        options=available_dates,
        format_func=lambda value: value.strftime("%d.%m.%Y"),
    )

hours_for_date = [
    timestamp for timestamp in starts if timestamp.date() == selected_date
]

with hour_column:
    selected_start = st.selectbox(
        "Час",
        options=hours_for_date,
        format_func=lambda value: value.strftime("%H:%M"),
    )

try:
    scenario = forecast_24_hours(
        model=model,
        airport_data=airport_data,
        forecast_start=selected_start,
        feature_columns=feature_columns,
    )
except (ValueError, IndexError) as error:
    show_error(str(error))


# Типичный календарный спрос рассчитывается только до начала теста.
profile_history = airport_data[
    airport_data[TIMESTAMP_COLUMN] < test_start
].copy()
profile_history["weekday"] = profile_history[TIMESTAMP_COLUMN].dt.dayofweek
profile_history["hour"] = profile_history[TIMESTAMP_COLUMN].dt.hour

calendar_reference = (
    profile_history.groupby(["weekday", "hour"])[TARGET_COLUMN]
    .mean()
)

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


st.divider()

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


forecast_column, scenario_column = st.columns([2.1, 1])

with forecast_column:
    st.subheader("Почасовой профиль выбранного окна")
    st.caption(
        f"{airport_labels[selected_airport]} · "
        f"старт {selected_start:%d.%m.%Y, %H:%M} · 24 часа"
    )

    chart = go.Figure()
    chart.add_trace(
        go.Bar(
            x=scenario["timestamp"],
            y=scenario["prediction"],
            name="Прогноз",
            marker_color="#2878B5",
            opacity=0.9,
        )
    )
    chart.add_trace(
        go.Scatter(
            x=scenario["timestamp"],
            y=scenario["actual"],
            name="Фактический спрос",
            mode="lines+markers",
            line={"color": "#F28E2B", "width": 2},
            marker={"size": 5},
        )
    )
    chart.add_trace(
        go.Scatter(
            x=scenario["timestamp"],
            y=scenario["calendar_average"],
            name="Типичный календарный уровень",
            mode="lines",
            line={"color": "#2E8B57", "width": 2, "dash": "dash"},
        )
    )
    chart.update_layout(
        height=390,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        xaxis_title="Час",
        yaxis_title="Поездки",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        hovermode="x unified",
    )
    chart.update_xaxes(tickformat="%d.%m %H:%M")
    st.plotly_chart(chart, use_container_width=True)

with scenario_column:
    st.subheader("Сценарий")
    st.markdown(
        f"""
        **Аэропорт:** {airport_labels[selected_airport]}  
        **Начало окна:** {selected_start:%d.%m.%Y, %H:%M}  
        **Модель:** LightGBM  
        **Режим:** тестовый период
        """
    )
    st.markdown(
        '<p class="subtle">Внутри окна каждый следующий прогноз '
        'использует прогнозы предыдущих часов. Зелёная линия — средний '
        'исторический спрос для того же аэропорта, дня недели и часа.</p>',
        unsafe_allow_html=True,
    )
    st.caption(f"Модель загружена из запуска {model_path.parent.name}")


st.subheader("Накопленный объём поездок")
st.caption(
    "Показывает, как суммарный спрос накапливается в течение выбранных суток."
)

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
    height=320,
    margin={"l": 10, "r": 10, "t": 15, "b": 10},
    xaxis_title="Час",
    yaxis_title="Накоплено поездок",
    legend={"orientation": "h", "y": 1.12, "x": 0},
    hovermode="x unified",
)
cumulative_chart.update_xaxes(tickformat="%d.%m %H:%M")
st.plotly_chart(cumulative_chart, use_container_width=True)


st.divider()
st.subheader("Календарный профиль спроса")
st.caption(
    "Среднее фактическое число поездок по часу суток и дню недели. "
    "Профиль рассчитан только по данным до тестового периода."
)

profile = (
    profile_history.groupby(["weekday", "hour"])[TARGET_COLUMN]
    .mean()
    .unstack("hour")
    .reindex(index=range(7), columns=range(24))
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

st.caption(
    "Календарный профиль показывает исторические закономерности "
    "и сам по себе не является прогнозом."
)