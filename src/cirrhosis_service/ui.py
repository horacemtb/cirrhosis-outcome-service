"""A demonstration interface for the prediction API."""

import json
import os
from pathlib import Path

import altair as alt
import httpx
import pandas as pd
import streamlit as st

from cirrhosis_service.preprocessing import CATEGORIES

EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "examples/predict_request.json"

CATEGORY_LABELS = {
    "Drug": {"Placebo": "Плацебо", "D-penicillamine": "D-пеницилламин"},
    "Sex": {"F": "Женский", "M": "Мужской"},
    "Ascites": {"N": "Нет", "Y": "Есть"},
    "Hepatomegaly": {"N": "Нет", "Y": "Есть"},
    "Spiders": {"N": "Нет", "Y": "Есть"},
    "Edema": {
        "N": "Нет",
        "S": "Есть без диуретиков или купируются ими",
        "Y": "Сохраняются на фоне диуретиков",
    },
}

LAB_FIELDS = {
    "Bilirubin": ("Билирубин, мг/дл", 0.1),
    "Cholesterol": ("Холестерин, мг/дл", 1.0),
    "Albumin": ("Альбумин, г/дл", 0.05),
    "Copper": ("Медь в моче, мкг/сут", 1.0),
    "Alk_Phos": ("Щелочная фосфатаза, Ед/л", 1.0),
    "SGOT": ("SGOT (АСТ), Ед/мл", 1.0),
    "Tryglicerides": ("Триглицериды, мг/дл", 1.0),
    "Platelets": ("Тромбоциты", 1.0),
    "Prothrombin": ("Протромбиновое время, с", 0.1),
}

# Broad input guards, not healthy reference intervals; see docs/input_limits.md.
INPUT_LIMITS = {
    "Age": (18, 100),
    "Bilirubin": (0.1, 100.0),
    "Cholesterol": (10.0, 2000.0),
    "Albumin": (0.2, 10.0),
    "Copper": (1.0, 1000.0),
    "Alk_Phos": (1.0, 20000.0),
    "SGOT": (1.0, 1000.0),
    "Tryglicerides": (10.0, 2000.0),
    "Platelets": (1.0, 2000.0),
    "Prothrombin": (5.0, 100.0),
}

LABELS = {
    "Age": "Возраст, лет",
    "Sex": "Пол",
    "Drug": "Препарат",
    "Stage": "Стадия заболевания",
    "Ascites": "Асцит",
    "Hepatomegaly": "Увеличение печени",
    "Spiders": "Сосудистые звёздочки",
    "Edema": "Отёки",
    **{name: label for name, (label, step) in LAB_FIELDS.items()},
}


def load_example():
    record = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
    for name, value in record.items():
        if name == "Age":
            value = int(round(value / 365.25))
        elif name in LAB_FIELDS:
            value = float(value)
        st.session_state[name] = value
    st.session_state.pop("result", None)


def category_input(name: str) -> str:
    return st.selectbox(
        LABELS[name],
        CATEGORIES[name],
        format_func=CATEGORY_LABELS[name].get,
        key=name,
    )


def patient_form() -> tuple[dict, bool]:
    with st.form("patient", border=False, enter_to_submit=False):
        st.markdown("#### Данные пациента")
        first, second = st.columns(2)
        with first:
            age = st.number_input(
                LABELS["Age"],
                min_value=INPUT_LIMITS["Age"][0],
                max_value=INPUT_LIMITS["Age"][1],
                value=None,
                step=1,
                format="%d",
                help="18–100 лет. В API передаётся возраст в днях: годы × 365,25.",
                key="Age",
            )
            drug = category_input("Drug")
        with second:
            sex = category_input("Sex")
            stage = st.selectbox(LABELS["Stage"], [1, 2, 3, 4], key="Stage")

        record = {
            "Age": int(round(age * 365.25)) if age is not None else None,
            "Drug": drug,
            "Sex": sex,
            "Stage": stage,
        }

        st.markdown("#### Клинические признаки")
        columns = st.columns(2)
        for index, name in enumerate(["Ascites", "Hepatomegaly", "Spiders", "Edema"]):
            with columns[index % 2]:
                record[name] = category_input(name)

        st.markdown("#### Лабораторные показатели")
        st.caption("Введите значения в исходной шкале данных, без пересчёта единиц.")
        columns = st.columns(2)
        for index, (name, (label, step)) in enumerate(LAB_FIELDS.items()):
            with columns[index % 2]:
                minimum, maximum = INPUT_LIMITS[name]
                record[name] = st.number_input(
                    label,
                    min_value=minimum,
                    max_value=maximum,
                    value=None,
                    step=step,
                    format="%.3f" if name == "Albumin" else "%.2f",
                    help=(
                        f"Допустимый ввод: {minimum:g}–{maximum:g}. "
                        "Это не диапазон нормы."
                    ),
                    key=name,
                )

        submitted = st.form_submit_button(
            "Рассчитать прогноз", type="primary", width="stretch"
        )
    return record, submitted


def request_explanation(record: dict) -> dict:
    url = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
    response = httpx.post(f"{url}/explain", json=record, timeout=30.0)
    response.raise_for_status()
    return response.json()


def display_value(feature: str, value) -> str:
    if feature in CATEGORY_LABELS:
        return CATEGORY_LABELS[feature][value]
    if feature == "Age":
        return str(int(round(value / 365.25)))
    return f"{value:g}"


def show_result(bundle: dict):
    result = bundle["result"]
    st.caption("Результат последнего расчёта")
    first, second, third = st.columns(3)
    first.metric("Класс", result["predicted_status"])
    second.metric("Оценка класса D", f"{result['probability_d']:.3f}")
    third.metric("Порог", f"{result['threshold']:.2f}")

    if result["predicted_status"] == "D":
        st.write("**Модель прогнозирует неблагоприятный исход (D).**")
    else:
        st.write("**Модель не прогнозирует неблагоприятный исход (C).**")
    st.caption(
        "Модель обучена различать смерть за время наблюдения (D) и наблюдения "
        "без зарегистрированной смерти (C). Прогноз C не означает выздоровление."
    )
    st.caption(
        "Оценка классификатора не задаёт индивидуальный риск за конкретный срок."
    )

    st.divider()
    st.markdown("#### Вклад признаков")
    st.caption(
        f"Базовое значение модели: {result['base_value']:.3f}. "
        "Положительный вклад повышает оценку D, отрицательный — снижает."
    )
    table = pd.DataFrame(result["contributions"])
    table["label"] = table["feature"].map(LABELS)
    table["display_value"] = [
        display_value(row["feature"], row["value"]) for row in result["contributions"]
    ]
    order = table["shap_value"].abs().sort_values(ascending=False).index
    top = table.loc[order, ["label", "display_value", "shap_value"]].head(10)
    chart = (
        alt.Chart(top)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("shap_value:Q", title="Вклад в оценку D"),
            y=alt.Y(
                "label:N",
                title=None,
                sort=top["label"].tolist(),
                axis=alt.Axis(labelLimit=240),
            ),
            color=alt.condition(
                alt.datum.shap_value >= 0,
                alt.value("#B95D61"),
                alt.value("#347D98"),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Признак"),
                alt.Tooltip("display_value:N", title="Значение"),
                alt.Tooltip("shap_value:Q", title="Вклад SHAP", format="+.4f"),
            ],
        )
        .properties(height=340)
        .configure_view(stroke=None)
    )
    st.altair_chart(chart, width="stretch")
    st.caption("Показаны 10 наибольших вкладов по абсолютному значению.")

    with st.expander("Все признаки и значения"):
        details = table[["label", "display_value", "shap_value"]].rename(
            columns={
                "label": "Признак",
                "display_value": "Значение",
                "shap_value": "Вклад SHAP",
            }
        )
        st.dataframe(details, hide_index=True, width="stretch")

    st.download_button(
        "Скачать результат",
        json.dumps(bundle, ensure_ascii=False, indent=2),
        file_name="prediction.json",
        mime="application/json",
    )


def main():
    st.set_page_config(page_title="Cirrhosis Outcome Service", layout="wide")
    st.html(
        "<style>.st-key-patient_card, .st-key-result_card "
        "{background-color: white;}</style>"
    )
    if "Age" not in st.session_state:
        load_example()
    elif isinstance(st.session_state.Age, float):
        st.session_state.Age = int(round(st.session_state.Age))

    st.title("Оценка исхода при циррозе печени")
    st.caption("Показатели пациента · прогноз модели · объяснение SHAP")
    inputs, results = st.columns([1.1, 1], gap="large")
    with inputs:
        st.button("Загрузить пример", on_click=load_example)
        with st.container(border=True, key="patient_card"):
            record, submitted = patient_form()

    with results:
        with st.container(border=True, key="result_card"):
            st.subheader("Результат")
            if submitted:
                st.session_state.pop("result", None)
                try:
                    with st.spinner("Расчёт прогноза и вклада признаков…"):
                        result = request_explanation(record)
                    st.session_state.result = {"input": record, "result": result}
                except httpx.TimeoutException:
                    st.error("Сервис не ответил вовремя. Повторите расчёт.")
                except httpx.HTTPStatusError as error:
                    if error.response.status_code == 422:
                        st.error("Проверьте значения полей: входные данные отклонены.")
                    else:
                        st.error("Сервис не смог выполнить расчёт. Повторите попытку.")
                except httpx.RequestError:
                    st.error("Не удалось подключиться к сервису. Проверьте запуск API.")

            if "result" in st.session_state:
                show_result(st.session_state.result)
            else:
                st.info("Заполните показатели и нажмите «Рассчитать прогноз».")

    st.caption(
        "Модель дополняет врачебную оценку. Клиническая валидация не проводилась; "
        "SHAP объясняет расчёт модели и не оценивает влияние лечения."
    )


if __name__ == "__main__":
    main()
