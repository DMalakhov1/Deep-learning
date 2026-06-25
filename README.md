# Deep Learning — учебные проекты и соревнования

Репозиторий с учебными и соревновательными проектами по Deep Learning. Внутри собраны задачи по computer vision, NLP, multi-label classification, ticket routing и реализации базовых компонентов нейронных сетей на NumPy.

Фокус репозитория — показать не только запуск моделей, но и полный цикл работы: анализ данных, подготовка признаков, обучение, подбор архитектур, валидация, сравнение подходов и подготовка итоговых файлов для соревнований.

## Проекты

| Проект                                                 | Задача                                          | Метрика / результат                                         | Основные методы                                                                  |
| ------------------------------------------------------ | ----------------------------------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------------- |
| [`competition_cv`](./competition_cv)                   | Классификация изображений на 100 классов        | Public score: **90.625**, validation accuracy до **89.48%** | PyTorch, timm, ConvNeXt, transfer learning, Albumentations, MixUp, SWA           |
| [`competition_nlp`](./competition_nlp)                 | Multi-label классификация русскоязычных текстов | Best validation F1-macro: **0.8376**                        | TF-IDF, Word2Vec, CNN/RNN-подходы, threshold tuning, PyTorch                     |
| [`itsm-ticket-routing-nlp`](./itsm-ticket-routing-nlp) | Multi-task классификация тикетов поддержки      | Best test score: **0.8770**                                 | TF-IDF, LinearSVC, SBERT, LightGBM, mBERT, LoRA, hybrid ensemble                 |
| [`hw_numpy`](./hw_numpy)                               | Реализация базовых DL-компонентов с нуля        | Учебная реализация                                          | NumPy, Linear, Softmax, BatchNorm, Dropout, Conv2D, Pooling, activations, losses |

## Стек

* Python
* NumPy, pandas, scikit-learn
* PyTorch, torchvision, timm
* Albumentations
* LightGBM
* sentence-transformers
* Hugging Face Datasets / Transformers
* matplotlib, seaborn

## Что показывает этот репозиторий

Репозиторий демонстрирует практические навыки, важные для ролей Data Scientist и Machine Learning Engineer:

* работу с CV и NLP задачами;
* обучение и сравнение deep learning моделей;
* transfer learning на предобученных архитектурах;
* построение пайплайнов аугментации изображений;
* работу с multi-label и multi-task классификацией;
* подбор метрик и threshold под конкретную задачу;
* построение сильных baseline-моделей;
* сравнение классического ML и transformer-based подходов;
* понимание базовых компонентов нейронных сетей на уровне NumPy-реализации;
* подготовку итоговых submission-файлов для соревнований.
