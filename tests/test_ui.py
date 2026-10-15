"""Check the form and API responses without launching web servers."""

import math
from pathlib import Path

import httpx
import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP_PATH = Path(__file__).resolve().parents[1] / "src/cirrhosis_service/ui.py"


@pytest.fixture
def ui(monkeypatch):
    calls = []

    def response(url, *, json, timeout):
        calls.append({"url": url, "record": json, "timeout": timeout})
        result = {
            "predicted_class": 1,
            "predicted_status": "D",
            "probability_d": 0.6,
            "threshold": 0.35,
            "base_value": 0.5,
            "contributions": [
                {
                    "feature": name,
                    "value": value,
                    "shap_value": 0.1 if name == "Bilirubin" else 0.0,
                }
                for name, value in json.items()
            ],
        }
        return httpx.Response(200, json=result, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", response)
    return AppTest.from_file(str(APP_PATH), default_timeout=20).run(), calls


def test_submit_converts_age_and_displays_result(ui, monkeypatch):
    app, calls = ui
    monkeypatch.setenv("API_URL", "http://localhost:9001/")
    assert isinstance(app.number_input(key="Age").value, int)
    app.number_input(key="Age").set_value(61)
    assert calls == []

    app.button[1].click().run()

    assert not app.exception
    assert len(calls) == 1
    assert calls[0]["url"] == "http://localhost:9001/explain"
    assert calls[0]["timeout"] == 30.0
    assert calls[0]["record"]["Age"] == int(round(61 * 365.25))
    assert len(calls[0]["record"]) == 17
    assert app.metric[0].value == "D"
    assert len(app.dataframe[0].value) == 17
    assert app.session_state.result["input"] == calls[0]["record"]
    assert any("Модель прогнозирует" in item.value for item in app.markdown)


def test_numeric_inputs_are_bounded(ui):
    app, calls = ui
    assert len(app.number_input) == 10
    for field in app.number_input:
        assert 0 < field.min <= field.value <= field.max
        assert math.isfinite(field.max)
    assert app.number_input(key="Age").min == 18
    assert app.number_input(key="Age").max == 100
    assert calls == []


def test_existing_session_uses_integer_age(record):
    app = AppTest.from_file(str(APP_PATH), default_timeout=20)
    for name, value in record.items():
        app.session_state[name] = 59.87 if name == "Age" else value
    app.run()

    assert not app.exception
    assert app.number_input(key="Age").value == 60
    assert isinstance(app.number_input(key="Age").value, int)


def test_class_c_is_presented_as_a_prediction(ui, monkeypatch):
    app, _ = ui
    original_post = httpx.post

    def response(url, **kwargs):
        result = original_post(url, **kwargs).json()
        result.update(predicted_class=0, predicted_status="C", probability_d=0.2)
        return httpx.Response(200, json=result, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", response)
    app.button[1].click().run()

    assert not app.exception
    assert app.metric[0].value == "C"
    assert any("Модель не прогнозирует" in item.value for item in app.markdown)
    assert any("не означает выздоровление" in item.value for item in app.caption)


def test_loading_example_clears_result(ui, record):
    app, calls = ui
    app.number_input(key="Age").set_value(70)
    app.button[1].click().run()
    app.button[0].click().run()

    assert not app.exception
    assert app.number_input(key="Age").value == int(round(record["Age"] / 365.25))
    assert "result" not in app.session_state
    assert not app.metric
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["connection", "timeout", "invalid"])
def test_service_failure_clears_previous_result(ui, monkeypatch, failure):
    app, _ = ui
    app.button[1].click().run()
    assert app.metric

    def fail(url, **kwargs):
        if failure == "connection":
            raise httpx.ConnectError("Service unavailable")
        if failure == "timeout":
            raise httpx.ReadTimeout("Request timed out")
        return httpx.Response(422, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fail)
    app.button[1].click().run()

    assert not app.exception
    assert app.error
    assert "result" not in app.session_state
    assert not app.metric
