# CarAI: Сервис для расчета покраски деталей автомобиля

Добро пожаловать в **CarAI**! Этот проект представляет собой инструмент для автоматического обнаружения повреждений деталей автомобиля (двери, капот, крылья и т.п.) и расчета стоимости их покраски с помощью искусственного интеллекта.

## Основные возможности
- Обнаружение и классификация автомобильных деталей (дверь, капот, крыло и т.д.) с использованием модели YOLO.
- Выделение поврежденных участков деталей с помощью модели SAM (Segment Anything Model).
- Расчет общей площади, подлежащей обработке, и стоимости покраски на основе заданных параметров (стоимость краски, ширина покрытия и т.д.).
- Удобное сохранение аннотированных изображений с визуализацией обработанных областей.

---

## Установка

### 1. Клонирование репозитория
Склонируйте проект на локальный компьютер:
```bash
git clone https://github.com/your-username/CarAI.git
cd CarAI
```

### 2. Установка зависимостей
Убедитесь, что у вас установлен Python 3.8 или выше. Затем выполните команду:
```bash
pip install -r requirements.txt
```

### 3. Установка весов моделей
Весовые файлы для моделей YOLO и SAM уже включены в проект. Убедитесь, что они находятся в папке models:

* train14/weights/best.pt — YOLO-модель.
* models/sam_vit_h_4b8939.pth — SAM-модель.
## Использование

## Обработка изображений

### 1. Поместите изображения для обработки в папку:
```bash
./assets/unedited/jpgs
```
### 2. Запустите скрипт:
```bash
python main.py
```
### 3. Результаты:
Аннотированные изображения сохраняются в папке:
```bash
./assets/edited
```
В терминале выводятся площади повреждений и стоимость их покраски для каждой детали.

## Конвертация аннотаций из LabelMe в YOLO
Если у вас есть аннотации в формате LabelMe, вы можете конвертировать их в формат YOLO:

### 1. Поместите JSON-файлы аннотаций в папку:
```bash
./assets/unedited/annotations
```
### 2. Запустите скрипт:
```bash
python labelme_to_yolo.py
```
### 3. Конвертированные аннотации сохранятся в папке:
```bash
./assets/yolo_annotations
```
## Конфигурация
Параметры обучения и тестирования указаны в файле dataset.yaml:

```bash
train: "D:/Algoth/CarAI/CarAI/assets/dataset/train_images"
val: "D:/Algoth/CarAI/CarAI/assets/dataset/train_images"
test: "D:/Algoth/CarAI/CarAI/assets/dataset/test_images"

names:
  0: "hood"
  1: "door"
  2: "wing"
```
### Параметры покраски указаны в main.py:
1. door_area: Примерная площадь дверей (в м²).
2. spray_width: Ширина краскопульта (в м).
3. nozzle_drift: Допустимое смещение сопла (в м).
4. LKM_cost_per_liter: Стоимость краски за литр (в руб.).
5. coverage_per_liter: Покрытие краски в м² на литр.
## Структура проекта
```bash
CarAI/
├── assets/
│   ├── dataset/               # Датасеты для обучения/валидации
│   │   ├── train_images/      # Изображения для обучения
│   │   ├── val_images/        # Изображения для валидации
│   │   └── test_images/       # Изображения для тестирования
│   │   └── dataset.yaml       # Конфигурация для обучения
│   ├── unedited/              # Необработанные изображения и аннотации
│   │   ├── annotations/       # Аннотации в формате LabelMe
│   │   └── jpgs/              # Изображения
│   ├── yolo_annotations/      # Конвертированные аннотации в формате YOLO
│   └── edited/                # Обработанные изображения
├── models/                    # Модели и веса
│   ├── downloader.py          # Скрипт для загрузки моделей (опционально)
│   ├── sam_vit_h_4b8939.pth   # SAM модель
│   └── yolo11n.pt             # YOLO модель
├── train14/                   # Обученная YOLO модель
│   └── weights/
│       └── best.pt
├── labelme_to_yolo.py         # Конвертер аннотаций
├── main.py                    # Главный скрипт обработки
├── requirements.txt           # Зависимости проекта
├── .gitignore                 # Файлы и папки для игнорирования
└── README.md                  # Документация
```
## Как работает проект?
### 1. Модель YOLO:

  * YOLO обнаруживает на изображении области, соответствующие частям автомобиля.
  * Классы объектов:
      * hood — капот.
      * door — дверь.
      * wing — крыло.
### 2. Модель SAM:
  * Выделяет точные контуры поврежденных участков деталей.
### 3. Расчет стоимости:

  * Рассчитывается площадь каждой детали.
  * Определяется количество краски, необходимое для покраски, с учетом покрытия на литр.
  * Считается итоговая стоимость.
## Лицензия
  Проект распространяется под MIT License.

Теперь у вас есть полная и связная документация, готовая к добавлению в репозиторий или публикации. Если нужно что-то доработать, дайте знать! 😊  
---