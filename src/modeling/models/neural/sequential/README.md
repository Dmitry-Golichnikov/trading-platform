# Sequential Models для временных рядов

Этот модуль содержит реализации нейросетевых моделей для sequence data.

## Реализованные модели

### 1. LSTM (Long Short-Term Memory)

Базовая рекуррентная модель с механизмом "памяти" для долгосрочных зависимостей.

**Варианты:**
- `LSTMModel` - базовая LSTM (uni/bidirectional)
- `StackedLSTMModel` - глубокая LSTM с residual connections
- `AttentionLSTMModel` - LSTM с attention mechanism

**Конфигурации:**
- `lstm_default.yaml` - базовая конфигурация
- `lstm_bidirectional.yaml` - bidirectional LSTM
- `lstm_deep.yaml` - глубокая сеть для сложных задач

**Пример использования:**
```python
from src.modeling.models.neural.sequential import LSTMModel

model = LSTMModel(
    input_size=10,
    hidden_size=128,
    num_layers=2,
    seq_length=50,
    output_size=1,
    dropout=0.2,
    bidirectional=False,
    task="classification",
)

model.fit(X_train, y_train, X_val, y_val)
predictions = model.predict(X_test)
```

### 2. GRU (Gated Recurrent Unit)

Упрощённая версия LSTM с меньшим количеством параметров и быстрым обучением.

**Варианты:**
- `GRUModel` - базовая GRU
- `AttentionGRUModel` - GRU с attention
- `MultiHeadAttentionGRUModel` - GRU с multi-head attention

**Конфигурации:**
- `gru_default.yaml` - базовая конфигурация
- `gru_regression.yaml` - для регрессионных задач

### 3. Seq2Seq with Attention

Encoder-Decoder архитектура с attention механизмом.

**Варианты:**
- `Seq2SeqAttentionModel` - базовая seq2seq с Bahdanau/Luong attention
- `MultiStepSeq2SeqModel` - для многошагового прогнозирования

**Конфигурация:**
- `seq2seq_default.yaml`

**Особенности:**
- Поддержка LSTM и GRU в encoder/decoder
- Bahdanau и Luong attention
- Визуализация attention weights

### 4. TCN (Temporal Convolutional Network)

Использует dilated causal convolutions для обработки последовательностей.

**Варианты:**
- `TCNModel` - базовая TCN
- `ResidualTCNModel` - TCN с дополнительными residual connections

**Конфигурация:**
- `tcn_default.yaml`

**Преимущества:**
- Параллельное обучение (быстрее RNN)
- Большой receptive field
- Эффективна для длинных последовательностей

### 5. CNN+LSTM Hybrid

Комбинация CNN для локальных паттернов и LSTM для долгосрочных зависимостей.

**Варианты:**
- `CNNLSTMModel` - базовая гибридная модель
- `CNNGRUModel` - CNN+GRU
- `ResidualCNNLSTMModel` - с residual connections
- `MultiScaleCNNLSTMModel` - multi-scale CNN с разными kernel sizes

**Конфигурация:**
- `cnn_lstm_default.yaml`

**Применение:**
- Когда важны и локальные, и глобальные паттерны
- Извлечение иерархических признаков

### 6. TFT (Temporal Fusion Transformer)

Упрощённая версия TFT с ключевыми компонентами:
- Variable selection
- LSTM для temporal processing
- Multi-head attention
- Gated residual networks

**Конфигурация:**
- `tft_default.yaml`

**Особенности:**
- Самая мощная модель в наборе
- Требует больше данных для обучения
- Хорошо работает с разнородными признаками

## Общие параметры

Все модели поддерживают:

### Архитектура
- `input_size` - размерность входных признаков (определяется автоматически)
- `hidden_size` - размер скрытых слоёв
- `num_layers` - количество слоёв
- `seq_length` - длина последовательности
- `output_size` - размер выхода
- `dropout` - dropout rate

### Обучение
- `epochs` - количество эпох
- `batch_size` - размер батча
- `learning_rate` - начальный learning rate
- `optimizer` - adam, adamw, sgd
- `scheduler` - onecycle, cosine, plateau, null
- `early_stopping` - patience для early stopping

### Последовательности
- `stride` - шаг sliding window
- `predict_horizon` - горизонт предсказания

### Продвинутые
- `device` - auto, cuda, cpu
- `mixed_precision` - использовать AMP
- `gradient_clip` - gradient clipping

## Sequence Dataset

`SequenceDataset` создаёт последовательности из данных с помощью sliding window:

```python
from src.modeling.models.neural.sequential import SequenceDataset, create_sequence_dataloader

# Создать dataset
dataset = SequenceDataset(
    X=X_train.values,
    y=y_train.values,
    seq_length=50,
    stride=1,
    predict_horizon=0,
)

# Или использовать helper функцию
dataloader = create_sequence_dataloader(
    X=X_train,
    y=y_train,
    seq_length=50,
    batch_size=128,
    shuffle=True,
)
```

## Data Augmentation

Модуль включает различные техники аугментации для sequence данных:

```python
from src.modeling.models.neural.sequential import get_default_augmentation

# Получить готовый набор аугментаций
aug = get_default_augmentation(mode="medium")  # light, medium, heavy

# Применить к данным
X_augmented = aug(X_tensor)
```

**Доступные аугментации:**
- `Jitter` - добавление шума
- `Scaling` - случайное масштабирование
- `MagnitudeWarp` - smooth warping амплитуды
- `TimeWarp` - деформация временной оси
- `WindowSlicing` - случайные окна
- `RandomCrop` - обрезка последовательностей

## Визуализация

```python
from src.modeling.models.neural.sequential.visualization import (
    plot_training_history,
    plot_attention_weights,
    plot_predictions,
)

# Training curves
history = model.get_training_history()
plot_training_history(history, save_path="training.png")

# Attention weights (для моделей с attention)
if hasattr(model, "get_attention_weights"):
    weights = model.get_attention_weights()
    plot_attention_weights(weights, save_path="attention.png")

# Predictions
plot_predictions(y_test, predictions, save_path="predictions.png")
```

## Выбор модели

**LSTM/GRU:**
- ✅ Простые, надёжные
- ✅ Хорошо работают с небольшими данными
- ❌ Медленное обучение на длинных последовательностях

**TCN:**
- ✅ Быстрое параллельное обучение
- ✅ Большой receptive field
- ❌ Требует больше памяти

**CNN+LSTM:**
- ✅ Лучшее из обоих миров
- ✅ Эффективное извлечение признаков
- ❌ Больше гиперпараметров

**Seq2Seq:**
- ✅ Хорошо для многошагового прогнозирования
- ✅ Интерпретируемость через attention
- ❌ Сложнее в обучении

**TFT:**
- ✅ Максимальное качество
- ✅ Variable selection
- ❌ Требует много данных и вычислений

## Best Practices

1. **Длина последовательности:**
   - Начните с 50-100 шагов
   - Увеличивайте если есть долгосрочные зависимости
   - Уменьшайте если данных мало

2. **Batch size:**
   - Больше = быстрее, но может хуже качество
   - Для LSTM/GRU: 64-256
   - Для TCN: 128-512

3. **Learning rate:**
   - OneCycle scheduler обычно лучший выбор
   - Начинайте с 0.001
   - Уменьшайте если нестабильность

4. **Regularization:**
   - Dropout: 0.2-0.3
   - Gradient clipping: 0.5-1.0
   - Early stopping: 10-20 epochs

5. **Mixed precision:**
   - Включайте для GPU с Tensor Cores
   - Ускоряет обучение в 2-3 раза
   - Не влияет на качество

## Производительность

Примерное время обучения на 1 эпоху (NVIDIA RTX 3080, batch_size=128, seq_length=50):

- LSTM: ~5 секунд
- GRU: ~4 секунды
- TCN: ~3 секунды
- CNN+LSTM: ~6 секунд
- Seq2Seq: ~8 секунд
- TFT: ~12 секунд

## Требования

```
torch >= 2.0.0
numpy >= 1.20.0
pandas >= 1.3.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scikit-learn >= 1.0.0
```

## Примеры использования

См. также:
- `examples/sequential_models_demo.py` - демонстрация всех моделей
- `tests/unit/test_sequential_models.py` - юнит-тесты
- `configs/models/*.yaml` - конфигурационные файлы
