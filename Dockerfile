FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

WORKDIR /app

RUN pip install --no-cache-dir poetry==2.5.1

COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --only main --no-root --no-ansi

COPY src/ ./src/
RUN poetry install --only-root --no-ansi

COPY data/train.csv ./data/train.csv
RUN python -m cirrhosis_service.train --data data/train.csv --output artifacts

EXPOSE 8000

CMD ["uvicorn", "cirrhosis_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
