# Сервис распознавания объектов (CPU)

REST API сервис для распознавания объектов на изображениях с использованием FastAPI и YOLOv5.

## Возможности

- Работа только на CPU
- Локальное хранение моделей
- Замена моделей через API
- Поддержка кастомных моделей YOLOv5
- Настройка размера изображения для инференса (320-1280)
- Регулируемая толщина рамок
- Пакетная обработка изображений

## Быстрый старт

### 1. Подготовка модели

Скачайте модель YOLOv5 и поместите в папку `models/`:

```bash
mkdir -p models
# Скачать стандартную модель
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt -O models/yolov5s.pt
```

Доступные модели:
- `yolov5n.pt` - nano (самая быстрая, наименее точная)
- `yolov5s.pt` - small
- `yolov5m.pt` - medium
- `yolov5l.pt` - large
- `yolov5x.pt` - extra large (самая точная, самая медленная)

### 2. Запуск

**Локально:**
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Docker:**
```bash
docker-compose up --build
```

## API Эндпоинты

### Детекция

| Эндпоинт | Метод | Описание |
|----------|-------|----------|
| `/detect` | POST | JSON с результатами детекции |
| `/detect/image` | POST | Изображение с рамками |
| `/detect/batch` | POST | Пакетная обработка |
| `/classes` | GET | Список классов модели |
| `/sizes` | GET | Доступные размеры инференса |

### Управление моделями

| Эндпоинт | Метод | Описание |
|----------|-------|----------|
| `/models` | GET | Список всех моделей |
| `/models/current` | GET | Информация об активной модели |
| `/models/upload` | POST | Загрузить новую модель |
| `/models/activate/{name}` | POST | Активировать модель |
| `/models/{name}` | DELETE | Удалить модель |

## Параметры

### Параметры детекции

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `confidence_threshold` | float | 0.25 | Порог уверенности (0-1) |
| `imgsz` | string | "640" | Размер изображения для инференса |
| `line_thickness` | int | 2 | Толщина рамки (1-10) |

### Доступные размеры изображения

| Размер | Описание |
|--------|----------|
| 320 | Быстрый, низкое качество |
| 416 | Быстрый |
| 512 | Средний |
| 640 | Стандартный (по умолчанию) |
| 768 | Высокое качество |
| 1024 | Очень высокое качество |
| 1280 | Максимальное качество, медленный |

## Примеры использования

### Загрузка новой модели

```bash
# Загрузить и сразу активировать
curl -X POST "http://localhost:8000/models/upload?activate=true" \
  -F "file=@yolov5m.pt"

# Только загрузить (без активации)
curl -X POST "http://localhost:8000/models/upload?activate=false" \
  -F "file=@yolov5l.pt"
```

### Переключение между моделями

```bash
# Список моделей
curl http://localhost:8000/models

# Активировать другую модель
curl -X POST "http://localhost:8000/models/activate/yolov5m.pt"

# Проверить текущую модель
curl http://localhost:8000/models/current
```

### Детекция объектов

```bash
# Получить JSON результат
curl -X POST "http://localhost:8000/detect?imgsz=640&confidence_threshold=0.5" \
  -F "file=@image.jpg"

# Получить изображение с рамками
curl -X POST "http://localhost:8000/detect/image?imgsz=768&line_thickness=3" \
  -F "file=@image.jpg" \
  --output result.jpg
```

### Python клиент

```python
import requests

API = "http://localhost:8000"

# Загрузка модели
with open("my_custom_model.pt", "rb") as f:
    resp = requests.post(
        f"{API}/models/upload",
        files={"file": f},
        params={"activate": "true"}
    )
    print(resp.json())

# Детекция
with open("image.jpg", "rb") as f:
    resp = requests.post(
        f"{API}/detect",
        files={"file": f},
        params={
            "confidence_threshold": 0.5,
            "imgsz": "640"
        }
    )
    print(resp.json())

# Получить изображение с рамками
with open("image.jpg", "rb") as f:
    resp = requests.post(
        f"{API}/detect/image",
        files={"file": f},
        params={
            "confidence_threshold": 0.5,
            "imgsz": "1024",
            "line_thickness": 3
        }
    )
    with open("result.jpg", "wb") as out:
        out.write(resp.content)
```

## Структура проекта

```
object_detection_service/
├── main.py              # FastAPI приложение
├── requirements.txt     # Зависимости
├── Dockerfile
├── docker-compose.yml
├── client.py            # Пример клиента
├── models/              # Директория для моделей
│   ├── yolov5s.pt       # Модель YOLOv5
│   └── active_model.pt  # Копия активной модели
├── README.md            # Документация (English)
├── README_UA.md         # Документация (Українська)
└── README_RU.md         # Документация (Русский)
```

## Кастомные модели

Сервис поддерживает кастомные модели YOLOv5, обученные на ваших данных.

**Требования к модели:**
- Формат PyTorch (.pt)
- Архитектура YOLOv5
- Совместимость с `torch.hub.load("ultralytics/yolov5", "custom", ...)`

**Загрузка кастомной модели:**
```bash
curl -X POST "http://localhost:8000/models/upload" \
  -F "file=@my_custom_model.pt" \
  -F "activate=true"
```

## Производительность (CPU)

| Модель | Размер | Время инференса* |
|--------|--------|------------------|
| yolov5n | 4 MB | ~50мс |
| yolov5s | 14 MB | ~100мс |
| yolov5m | 42 MB | ~200мс |
| yolov5l | 92 MB | ~350мс |
| yolov5x | 174 MB | ~500мс |

*Intel Core i7, изображение 640x640

## Документация API

После запуска сервера доступна интерактивная документация:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Лицензия

MIT
