# Train API

Мини-сервис на FastAPI для инференса Iris-модели, обучаемой через `train.py`.

## Что внутри

- `train.py` — обучение модели и сохранение артефакта `model.pkl`
- `app.py` — API с эндпоинтами:
  - `GET /health` — liveness (`ok` когда модель загружена, `degraded` иначе)
  - `GET /ready` — readiness (модель загружена)
  - `GET /info` — источник, версия и статус загруженной модели
  - `GET /metrics` — Prometheus metrics
  - `POST /predict` — предсказание по 4 признакам

## Как запустить локально

```bash
cd train
python3 -m pip install -r requirements.txt
python3 train.py
uvicorn app:app --reload
```

Проверка:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/ready
curl -s http://127.0.0.1:8000/info
curl -s http://127.0.0.1:8000/metrics
curl -s -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features":[5.1,3.5,1.4,0.2]}'
```

Примеры ответов:

```json
{"status":"ok"}
```

```json
{"prediction":[0]}
```

## Конфигурация

Переменные читаются из `train/.env`:

- `MODEL_NAME`
- `MODEL_VERSION`
- `MODEL_URI` (MLflow model URI, по умолчанию `models:/iris-model@champion`)

Пример:

```env
MODEL_NAME=iris-random-forest
MODEL_VERSION=v1
MODEL_URI=models:/iris-model@champion
```

Поведение загрузки модели:

- сначала API пытается загрузить модель из MLflow по `MODEL_URI`
- если загрузка из MLflow не удалась, модель считается недоступной (`/ready` вернёт `503`)
- `model_name`/`model_version` в `/health` и `/info` отражают реально загруженную версию из Registry
- текущий источник модели можно посмотреть в `GET /info` (`model_source`)

## Как тестировать

Запуск тестов:

```bash
cd train
python3 -m pytest -q
```

Что покрыто:

- unit test обучения (`tests/test_train.py`)
- API tests (`tests/test_api.py`):
  - `/health`
  - `/ready`
  - валидный и невалидный `/predict`
  - `500` и `503` ветки ошибок
  - формат и рост метрик в `/metrics`

## Local MLflow usage

Запусти обучение и открой UI:

```bash
cd train
python3 train.py
mlflow ui
```

Дальше открой `http://127.0.0.1:5000` и проверь:

- experiment `iris`
- run с метрикой `accuracy`
- модель `iris-model` в Model Registry

Локальные артефакты MLflow (`mlflow.db`, `mlruns/`) исключены из git через `.gitignore`.

## Как собрать Docker

Сборка и запуск:

```bash
cd train
docker build -t train-api:local .
docker run --rm -p 8000:8000 train-api:local
```

Smoke check контейнера:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features":[5.1,3.5,1.4,0.2]}'
```

## Как работает CI

Workflow: `.github/workflows/pipeline.yml`

Workflow запускается на `push`/`pull_request`, если изменены:

- `train/**`
- `.github/workflows/pipeline.yml`

Также поддерживается ручной запуск через `workflow_dispatch`.

В pipeline выполняется:

1. установка Python и зависимостей
2. запуск `train.py` и проверка `model.pkl`
3. запуск `pytest`
4. smoke test локального API
5. `docker build` с тегом `${{ github.sha }}`
6. security scan образа через Trivy (fail на `HIGH/CRITICAL`)
7. smoke test контейнера
8. на `main`:
   - login в GHCR
   - push образов:
     - `ghcr.io/<repo>/train-api:${{ github.sha }}`
     - `ghcr.io/<repo>/train-api:latest`
