# ITSM / Customer Support Ticket Routing

Проект по multi-task NLP классификации тикетов службы поддержки.

Цель — построить модель, которая по тексту обращения предсказывает:

* `queue` — очередь обработки тикета, 52 класса;
* `priority` — приоритет тикета, 5 классов;
* `type` — тип тикета: `Incident`, `Request`, `Problem`, `Change`, `Unknown`.

Значение `Unknown` используется для пропущенной разметки в поле `type`.

## Данные

Используется датасет Hugging Face:

```text
Tobi-Bueck/customer-support-tickets
```

Входные признаки:

* `subject`
* `body`

Целевые переменные:

* `queue`
* `priority`
* `type`

Сам датасет в репозиторий не добавляется. Он загружается напрямую через библиотеку `datasets`.

## Split

Используется фиксированное разбиение с seed `42`:

| Split      | Размер |
| ---------- | -----: |
| Train      | 49 412 |
| Validation |  6 176 |
| Test       |  6 177 |

Индексы разбиения лежат в папке `data/`:

```text
data/train_idx.txt
data/val_idx.txt
data/test_idx.txt
```

После загрузки датасета строки выбираются по этим индексам.

## Структура проекта

```text
.
├── README.md
├── Homework.ipynb
├── eda_benchmark.py
├── requirements.txt
├── .gitignore
└── data/
    ├── train_idx.txt
    ├── val_idx.txt
    └── test_idx.txt
```

## Что сделано

В проекте есть:

* EDA датасета;
* анализ распределения классов `queue`, `priority`, `type`;
* анализ длины текстов `subject` и `body`;
* проверка дублей между train / validation / test;
* baseline-модель `TF-IDF + LinearSVC`;
* сравнение нескольких подходов к классификации тикетов;
* расчёт итогового score;
* анализ уверенности предсказаний.

## Метрики

Основная метрика:

```text
Macro-F1(queue)
```

Дополнительные метрики:

```text
Accuracy(queue)
Accuracy(priority)
Accuracy(type)
```

Итоговый score считается по формуле:

```text
Score = 0.70 * MacroF1(queue) + 0.15 * Acc(priority) + 0.15 * Acc(type)
```

#

## Основные файлы

| Файл               | Описание                                                  |
| ------------------ | --------------------------------------------------------- |
| `Homework.ipynb`   | Основной ноутбук с EDA, моделями, результатами и выводами |
| `eda_benchmark.py` | Скрипт для EDA, проверки split и baseline-модели          |
| `data/*.txt`       | Фиксированные индексы train / validation / test           |


## Автор

Учебный проект по NLP, классификации текстов и маршрутизации обращений в customer support.

