# Оценка соответствия паттернам популярности по тексту

В рабочем модуле оставлена **только выбранная нейронная сеть** `NeuralNetworkModel`. Градиентный бустинг, случайный лес и CatBoost перенесены в `model_tests/text_candidates.py`. Старые эксперименты и их сохранённые модели доступны через `ExperimentalTextAssessment.load()` из этого файла.

## Вход и выход

Обучение: текст, тема и целевая оценка модуля 2. Прогноз: текст и тема. Социальные метрики и аудитория не входят в признаки.

```text
Полный текст → замороженная SBERT/E5 → 384 числа
Тема → E5 с префиксом «Тема статьи: …» → 384 числа
                         ↓
StandardScaler → MLP 768 → 256 → 64 → 1 → 10 × tanh
                         ↓
             оценка в диапазоне [-10, 10]
```

`intfloat/multilingual-e5-small` используется через общий `TextEmbedder` из модуля БД. Длинные тексты обрабатываются частями. Веса E5 заморожены; эмбеддинги тем кэшируются. Обучается только MLP: CPU, 100 эпох, AdamW, MSE, batch size 128, шаг 0.001, weight decay 0.01, Dropout 0.2. Scaler обучается только на обучающей части и сохраняется с весами сети. В прогнозе Dropout выключен.

## Запуск из корня проекта

```powershell
# Переобучение нейросети на всех размеченных статьях БД
.\.venv312\Scripts\python.exe -m src.assesment_conform_of_PP_text_based train

# Прогноз по текстовому файлу
.\.venv312\Scripts\python.exe -m src.assesment_conform_of_PP_text_based predict --model-file artifacts/text_models/neural_network.joblib --text-file model_tests/example_article.txt --topic 'Технологии'
```

`--model neural_network` поддерживается для совместимости. Другие модели доступны только в `model_tests`. `--output` меняет каталог артефактов; по умолчанию `artifacts/text_models`. `--refresh-scores` пересчитывает целевые оценки через модуль БД перед обучением. `--reembed` нужен, если задан другой `EMBEDDING_MODEL`.

```python
from src.assesment_conform_of_PP_text_based import AssesmentConformPPtextBased

assessor = AssesmentConformPPtextBased()
assessor.train(texts, topics, metric_scores)
assessor.save("artifacts/text_models/neural_network.joblib")
loaded = AssesmentConformPPtextBased.load("artifacts/text_models/neural_network.joblib")
score = loaded.assess("Текст новой статьи", "Игры")
```

Артефакт хранит scaler, веса MLP, кэш тем, версии признаков, E5 и формулы метрик. Веса E5 не дублируются. Перед использованием загрузите `.env`, чтобы применить проектный `HF_HOME`. Загружайте собственные доверенные joblib-файлы.

Данные читаются через `ArticleTextTable.list_training_data()`. Целевая оценка проверяется по снимку аудитории `article_metrics.audience`; неизвестные метрики, нулевые просмотры и нулевая аудитория исключают целевую оценку. Известные нули реакций и комментариев допустимы. Модуль не перезаписывает оценки в БД своими прогнозами. При загрузке проверяется версия целевой формулы: старый артефакт нужно переобучить после пересчёта БД.

## Связь с модулем 4

`src.social_metrics_prediction train` обучает согласованную пару: эту нейросеть на всех данных и выбранную модель социальных метрик на её прогнозах вне обучающих групп (cross-fitting). Для последовательного прогноза используйте эту команду после обновления БД. Она сохраняет SHA256 нейросети в метаданных модуля 4. Отдельное переобучение модуля 3 требует переобучить и модуль 4.

## Результаты выбора

На выборке из 2415 статей нейросеть получила средний MSE CV 12.9526 и MSE отложенной проверки 14.1842. Сравнение четырёх моделей находится в `model_tests/results/20261021_141831_UTC/`. Команда повторного сравнения: `python -m model_tests.compare_text_models`.

Оценка является прогнозом эвристической формулы модуля 2. Разделение по группам одинаковых текстов проверяет имеющиеся каналы; оно не измеряет качество будущих дат и новых ресурсов.
