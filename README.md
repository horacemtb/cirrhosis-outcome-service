# Cirrhosis Outcome Service

Сервис прогнозирования исхода заболевания по клинико-лабораторным данным.
Random Forest классифицирует `Status`: `C` — цензурированное наблюдение, `D` — смерть. FastAPI предоставляет HTTP-интерфейс для предсказаний. Модель не оценивает время до события.

[Карточка модели](model_card.md) · [Модель в техническом процессе](Model_BPMN.md)

## Установка

Python 3.10–3.12 и [Poetry 2](https://python-poetry.org/docs/#installation).
Команды выполняются из корня проекта.

```powershell
poetry config virtualenvs.in-project true --local
poetry env use 3.12
poetry install
```

В Git сохраняются `pyproject.toml`, `poetry.lock` и `poetry.toml`.
Окружение `.venv/` и результаты обучения исключены из Git.
Обучающие данные находятся в `data/train.csv`.

## Данные и обучение

Выполните команду:

```powershell
poetry run python -m cirrhosis_service.train --data data/train.csv --output artifacts
```

Модель обучается на 80% данных, остальные 20% используются для оценки. Результаты: `artifacts/model.joblib` и `artifacts/metrics.json`. Повторный запуск перезаписывает эти файлы.

## Запуск API

После обучения запустите сервер:

```powershell
poetry run uvicorn cirrhosis_service.api:app --host 127.0.0.1 --port 8000
```

Откройте [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
Для проверки `POST /predict` используйте [пример запроса](examples/predict_request.json).
`GET /health` проверяет доступность сервиса.

Ответ содержит класс, статус, вероятность класса `D` и порог `0.35`.
Неверный вход возвращает HTTP 422.
Путь к модели можно изменить переменной `MODEL_PATH`; по умолчанию используется `artifacts/model.joblib`.

## Docker

Требуется Docker Desktop в режиме Linux containers.
Из корня проекта соберите образ и запустите контейнер:

```powershell
docker build -t cirrhosis-outcome-service .
docker run --rm --name cirrhosis-api -p 127.0.0.1:8000:8000 cirrhosis-outcome-service
```

При сборке устанавливаются зависимости из `poetry.lock` и обучается модель
на `data/train.csv`. Модель и метрики сохраняются внутри образа;
локальный каталог `artifacts/` не изменяется.
При запуске контейнера повторное обучение не выполняется.
После изменения кода, зависимостей или данных пересоберите образ.

Откройте [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
В другом терминале PowerShell, из корня проекта, проверьте запросы:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
$body = Get-Content examples/predict_request.json -Raw
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body $body
```

Ожидается `status: ok` и JSON с предсказанием.
Остановка: `docker stop cirrhosis-api` из другого терминала.

## Предсказание без API

```powershell
poetry run python -m cirrhosis_service.predict --input examples/predict_request.json
```

## Тесты

```powershell
poetry run pytest -q
```

Тесты не требуют обучающего CSV или готовой модели.

## Проверка качества кода

```powershell
poetry run black src tests
poetry run flake8 src tests
poetry run pre-commit install
poetry run pre-commit run --all-files
```

Black форматирует код, Flake8 проверяет стиль и ошибки.
После установки pre-commit проверки выполняются перед каждым коммитом.

## CI

[GitHub Actions](https://github.com/horacemtb/cirrhosis-outcome-service/actions)
проверяет каждый push и pull request: конфигурацию Poetry и pre-commit,
форматирование Black, Flake8, тесты, сборку Docker и запросы к API контейнера.

Результаты доступны во вкладке **Actions**, процесс **CI**, задание **Checks**.
Проверку можно запустить вручную кнопкой **Run workflow**.
Автоматическое развёртывание на сервер не настроено.
