# Deep Learning — учебные проекты и соревнования

Репозиторий с учебными и соревновательными проектами по Deep Learning. Внутри собраны задачи по computer vision, NLP, multi-label classification, ticket routing и реализации базовых компонентов нейронных сетей на NumPy.

Фокус репозитория — показать не только запуск моделей, но и полный цикл работы: анализ данных, подготовка признаков, обучение, подбор архитектур, валидация, сравнение подходов и подготовка итоговых файлов для соревнований.

## Проекты

| Проект | Задача | Метрика / результат | Основные методы |
|---|---|---|---|
| [`competition_cv`](./competition_cv) | Классификация изображений на 100 классов | Public score: **90.625**, validation accuracy до **89.48%** | PyTorch, timm, ConvNeXt, transfer learning, Albumentations, MixUp, SWA |
| [`competition_nlp`](./competition_nlp) | Multi-label классификация русскоязычных текстов | Best validation F1-macro: **0.8376** | TF-IDF, Word2Vec, CNN/RNN-подходы, threshold tuning, PyTorch |
| [`itsm-ticket-routing-nlp`](./itsm-ticket-routing-nlp) | Multi-task классификация тикетов поддержки | Best test score: **0.8770** | TF-IDF, LinearSVC, SBERT, LightGBM, mBERT, LoRA, hybrid ensemble |
| [`hw_numpy`](./hw_numpy) | Реализация базовых DL-компонентов с нуля | Учебная реализация | NumPy, Linear, Softmax, BatchNorm, Dropout, Conv2D, Pooling, activations, losses |

## 1. Computer Vision: классификация изображений

**Папка:** [`competition_cv`](./competition_cv)

### Задача

Нужно построить модель классификации изображений на **100 классов**. Данные состоят из маленьких RGB-изображений размером **32×32**, из-за чего важную роль играют upscale, transfer learning и корректная аугментация.

### Что сделано

В ноутбуке реализованы:

- EDA тренировочных и тестовых изображений;
- проверка форматов, размеров и RGB-каналов;
- анализ дисбаланса классов;
- визуальная проверка представителей классов;
- пайплайн аугментаций через `Albumentations`;
- сравнение моделей с transfer learning и без него;
- сравнение простых и продвинутых аугментаций;
- обучение ConvNeXt через `timm`;
- использование `MixUp`, class weights, learning-rate scheduler, early stopping и SWA;
- генерация итогового `submission.csv`.

### Использованные подходы

- `ConvNeXt` как основная модель;
- transfer learning на предобученной модели;
- `Albumentations` для аугментаций;
- `MixUp` для регуляризации;
- `SWA` для стабилизации весов;
- стратифицированное разбиение train / validation;
- постобработка submission-файла.

### Результаты

| Модель / подход | Validation accuracy | Public score |
|---|---:|---:|
| ConvNeXt, baseline-аугментации | до **89.48%** | — |
| ConvNeXt, финальный соревновательный вариант | — | **90.625** |

Итоговый файл с предсказаниями:

```text
competition_cv/ConvNeXt_192_long (1).csv
```

## 2. NLP: multi-label классификация текстов

**Папка:** [`competition_nlp`](./competition_nlp)

### Задача

Нужно решить задачу multi-label классификации русскоязычных текстов. Для каждого текста модель предсказывает набор бинарных меток.

### Данные

Архив с данными лежит в папке проекта:

```text
competition_nlp/dl-2025-study-competition-2.zip
```

Состав данных:

| Файл | Размер | Описание |
|---|---:|---|
| `train.csv` | **29 568 строк** | тексты и multi-label разметка |
| `test.csv` | **7 392 строки** | тексты для предсказания |
| `sample_submission.csv` | **7 392 строки** | шаблон итогового файла |

### Метрика

Основная метрика соревнования:

```text
F1-macro
```

### Что сделано

В проекте реализованы:

- EDA текстов и меток;
- анализ дисбаланса классов;
- анализ количества меток на объект;
- проверка дублей;
- анализ длины текстов;
- пайплайны очистки текста под разные подходы;
- обучение моделей на TF-IDF, Word2Vec, CNN/RNN-подходах и embedding-based решениях;
- подбор threshold для multi-label классификации;
- early stopping и scheduler;
- генерация submission-файла.

### Результаты

Лучший зафиксированный результат на validation:

| Подход | Validation F1-macro |
|---|---:|
| TF-IDF + нейросетевая модель с threshold tuning | **0.8376** |

## 3. ITSM Ticket Routing: multi-task NLP классификация

**Папка:** [`itsm-ticket-routing-nlp`](./itsm-ticket-routing-nlp)

### Задача

Нужно построить модель маршрутизации обращений в службу поддержки. По тексту тикета модель предсказывает сразу несколько целевых переменных:

- `queue` — очередь обработки тикета, **52 класса**;
- `priority` — приоритет тикета, **5 классов**;
- `type` — тип тикета: `Incident`, `Request`, `Problem`, `Change`, `Unknown`.

### Данные

Используется датасет Hugging Face:

```text
Tobi-Bueck/customer-support-tickets
```

Фиксированное разбиение:

| Split | Размер |
|---|---:|
| Train | **49 412** |
| Validation | **6 176** |
| Test | **6 177** |

Индексы разбиения сохранены в папке:

```text
itsm-ticket-routing-nlp/data/
```

### Метрика

Итоговый score считается как взвешенная комбинация метрик:

```text
Score = 0.70 * MacroF1(queue) + 0.15 * Acc(priority) + 0.15 * Acc(type)
```

### Что сделано

В проекте реализованы:

- EDA датасета;
- анализ распределения `queue`, `priority`, `type`;
- анализ длины `subject` и `body`;
- проверка дублей между train / validation / test;
- baseline `TF-IDF + LinearSVC`;
- сравнение классических ML-подходов и transformer-based моделей;
- эксперименты с `mBERT`, LoRA и frozen encoder;
- hybrid-модель на TF-IDF и SBERT;
- финальное сравнение моделей на test set;
- анализ confidence предсказаний.

### Результаты

| Место | Модель | Test score |
|---:|---|---:|
| 1 | Hybrid TF-IDF + SBERT | **0.8770** |
| 2 | TF-IDF + LinearSVC | **0.8604** |
| 3 | TF-IDF + LogReg | **0.8317** |

Лучший результат показала гибридная модель `Hybrid TF-IDF + SBERT`.

## 4. NumPy: реализация компонентов нейронных сетей

**Папка:** [`hw_numpy`](./hw_numpy)

### Задача

Реализовать базовые компоненты нейронных сетей без использования готовых DL-фреймворков для самих слоёв.

### Что реализовано

В ноутбуке реализованы:

- sequential container;
- linear layer;
- softmax и log-softmax;
- batch normalization;
- dropout;
- Conv2D;
- MaxPool2D и AvgPool2D;
- GlobalMaxPool2D и GlobalAvgPool2D;
- flatten;
- функции активации: Leaky ReLU, ELU, SoftPlus, GELU;
- negative log-likelihood loss;
- базовая структура для обучения моделей.

Этот блок показывает понимание внутренних механизмов нейронных сетей: forward pass, backward pass, параметры слоёв, градиенты, регуляризацию и функции потерь.

## Структура репозитория

```text
Deep-learning/
├── competition_cv/
│   ├── README.md
│   ├── ДЗ_2.1 (1).ipynb
│   └── ConvNeXt_192_long (1).csv
│
├── competition_nlp/
│   ├── README.md
│   ├── ДЗ_3.1_recheck_1.ipynb
│   └── dl-2025-study-competition-2.zip
│
├── hw_numpy/
│   ├── README.md
│   ├── homework_module.ipynb
│   └── тестик.ipynb
│
├── itsm-ticket-routing-nlp/
│   ├── README.md
│   ├── Homework.ipynb
│   ├── eda_benchmark.py
│   └── data/
│
└── README.md
```

## Стек

- Python
- NumPy
- pandas
- scikit-learn
- PyTorch
- torchvision
- timm
- Albumentations
- LightGBM
- sentence-transformers
- Hugging Face Datasets
- Hugging Face Transformers
- matplotlib, seaborn

## Что показывает этот репозиторий

Репозиторий демонстрирует практические навыки, важные для ролей Data Scientist и Machine Learning Engineer:

- работу с CV и NLP задачами;
- обучение и сравнение deep learning моделей;
- transfer learning на предобученных архитектурах;
- построение пайплайнов аугментации изображений;
- работу с multi-label и multi-task классификацией;
- подбор метрик и threshold под конкретную задачу;
- построение сильных baseline-моделей;
- сравнение классического ML и transformer-based подходов;
- понимание базовых компонентов нейронных сетей на уровне NumPy-реализации;
- подготовку итоговых submission-файлов для соревнований.

## Как запустить

Для большинства проектов основной код находится в Jupyter Notebook.

Общий порядок запуска:

```bash
git clone https://github.com/DMalakhov1/Deep-learning.git
cd Deep-learning
```

Далее нужно перейти в нужную папку проекта и открыть соответствующий ноутбук.

Пример для ITSM-проекта:

```bash
cd itsm-ticket-routing-nlp
python eda_benchmark.py
```

Для соревнований `competition_cv` и `competition_nlp` основная логика находится в ноутбуках.

