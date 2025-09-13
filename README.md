## Финальный проект (Спринт 2): Языковая модель для автодополнения твитов

Проект посвящён обучению и сравнению двух подходов к автодополнению текста твитов:

- классическая LSTM-языковая модель, обучаемая с нуля;
- дообучение предобученной трансформерной модели (через `transformers`).

В качестве набора данных используется корпус твитов (1.6M, noemoticon). Основные эксперименты и результаты оформлены в ноутбуках в корне репозитория.

### Содержание репозитория

- `solution_lstm.ipynb` — обучение и оценка LSTM LM.
- `solution_pretrained.ipynb` — дообучение предобученной модели и сравнение.
- `compare.ipynb` — дополнительное сравнение/визуализации.
- `src/` — служебный код:
  - `lstm_model.py` — реализация LSTM LM и утилита загрузки сохранённой модели `load_lstm_lm`.
  - `next_token_dataset.py` — датасет и `collate_fn` для обучения предсказанию следующего токена c маскированием первых 75% токенов из функции потерь.
  - `data_utils.py` — очистка текста и единая логика разбиения текста на контекст и целевые токены (75%/25%).
- `data/` — данные: исходный CSV, ZIP и подготовленный `tweets_cleaned.csv`.
- `models/` — артефакты обученных моделей, например `lstm_lm-YYYYMMDD-HHMMSS/` с `model.pt`, `tokenizer.json`, `vocab.txt` и др.
- `vm/` — копии/варианты ноутбуков для запуска на удалённой ВМ.

## Установка окружения

Требуется Python 3.10+ (рекомендуется 3.10–3.12). Зависимости закреплены в `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

При необходимости распакуйте датасет твитов:

```bash
unzip -n data/training.1600000.processed.noemoticon.csv.zip -d data
```

## Данные

- `data/training.1600000.processed.noemoticon.csv(.zip)` — корпус твитов (Sentiment140 / noemoticon). Используется как источник сырых текстов.
- `data/tweets_cleaned.csv` — подготовленный текст после базовой очистки (`data_utils.clean_string`).

Очистка включает:
- приведение к нижнему регистру;
- удаление символов, кроме латиницы, цифр и пробелов;
- нормализацию пробелов.

Единая схема разбиения на вход/цель: первые 75% слов — контекст, последние 25% — целевая продолжение. Это правило последовательно применяется и для LSTM, и для предобученной модели (см. `data_utils.split_x_target_by_words`).

## Быстрый старт: обучение в ноутбуках

1) Откройте нужный ноутбук:
- `solution_lstm.ipynb` — обучение LSTM LM c нуля на подготовленных текстах;
- `solution_pretrained.ipynb` — дообучение предобученной модели из `transformers`.

2) Выполните ячейки по порядку:
- загрузка и подготовка данных;
- формирование датасета `NextTokenDataset` и `collate_batch` (маскирование первых 75% токенов в лоссе);
- обучение модели и сохранение чекпоинтов;
- базовая оценка (loss, при необходимости — perplexity) и сравнение подходов.

Артефакты модели сохраняются в `models/lstm_lm-<timestamp>/` вместе с токенизатором (`AutoTokenizer`-совместимая структура), что позволяет легко восстановить модель для инференса.

## Инференс LSTM-модели из кода

Ниже минимальный пример генерации продолжения твита на базе сохранённой модели LSTM.

```python
import torch
from src.lstm_model import load_lstm_lm

# Загрузите последнюю или нужную директорию запуска
run_dir = "models/lstm_lm-20250912-181525"

tokenizer, model, pad_id, eos_id = load_lstm_lm(run_dir, device="cpu")

prompt = "the weather today"
prefix_ids = torch.tensor([
    tokenizer.encode(prompt, add_special_tokens=False)
], dtype=torch.long)

with torch.no_grad():
    seq = model.generate(prefix_ids, max_new_tokens=20, eos_id=eos_id)

text = tokenizer.decode(seq[0].tolist(), skip_special_tokens=True)
print(text)
```

Примечания:
- генерация жадная (argmax по логитам); для разнообразия можно добавить температурное/топ-k/топ-p сэмплирование при необходимости (в текущей реализации не включено);
- ранняя остановка — при предсказании `eos_id` для всех элементов батча.

## Ключевые детали реализации

- `LSTMLM` (`src/lstm_model.py`):
  - эмбеддинги + LSTM (`batch_first=True`) + линейная «голова» до размера словаря;
  - `forward` принимает `input_ids` и реальные `lengths` (после паддинга), использует `pack_padded_sequence`/`pad_packed_sequence`;
  - `generate` инициализируется скрытым состоянием по префиксу и порождает токены итеративно (жадно).
- `NextTokenDataset` (`src/next_token_dataset.py`):
  - добавляет `eos_id`, формирует вход/цель со смещением на 1;
  - `collate_batch` паддит входы, а в `labels` маскирует первые 75% токенов значением `-100` (игнор в `CrossEntropyLoss`).
- Токенизатор — `AutoTokenizer` (совместимая структура сохранена в каталоге модели), поэтому модель можно восстановить без исходного кода ноутбуков.

## Структура каталогов

```text
data/
  training.1600000.processed.noemoticon.csv[.zip]
  tweets_cleaned.csv
models/
  lstm_lm-YYYYMMDD-HHMMSS/
    model.pt
    meta.json               # метаданные запуска (если сохранены)
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    vocab.txt
src/
  data_utils.py
  lstm_model.py
  next_token_dataset.py
```

## Репродуцируемость

- зависимости зафиксированы в `requirements.txt`;
- ноутбуки содержат все шаги подготовки/обучения;
- для повторения инференса достаточно каталога модели `models/lstm_lm-*/` и функции `load_lstm_lm`.

## Лицензия и данные

- Данные твитов принадлежат их авторам; используйте их с соблюдением условий источника (Sentiment140). Проверьте актуальные ограничения и лицензионные требования.
- Код предоставляется «как есть» в учебных целях.

## Контакты

Вопросы и предложения можно оформлять в Issues или передавать автору проекта.


