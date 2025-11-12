"""
Демонстрация использования sequential моделей.

Показывает как использовать LSTM, GRU, TCN, CNN+LSTM и другие модели
для временных рядов.
"""

import numpy as np
import pandas as pd
import torch

from src.modeling.models.neural.sequential import (
    CNNLSTMModel,
    GRUModel,
    LSTMModel,
    TCNModel,
    get_default_augmentation,
)
from src.modeling.models.neural.sequential.visualization import (
    plot_training_history,
)


def generate_synthetic_data(n_samples: int = 1000, n_features: int = 10):
    """
    Генерация синтетических данных временного ряда.

    Returns:
        X: DataFrame с признаками
        y: Series с таргетом
    """
    # Создаём временной ряд с трендом и сезонностью
    t = np.arange(n_samples)

    # Базовый сигнал
    trend = 0.001 * t
    seasonality = 0.5 * np.sin(2 * np.pi * t / 100)
    noise = 0.1 * np.random.randn(n_samples)

    base_signal = trend + seasonality + noise

    # Создаём признаки
    features = {}
    for i in range(n_features):
        # Признаки с разными задержками и преобразованиями
        lag = np.random.randint(1, 10)
        features[f"feature_{i}"] = np.roll(base_signal, lag) + 0.1 * np.random.randn(n_samples)

    X = pd.DataFrame(features)

    # Таргет: бинарная классификация (будет ли рост)
    y = pd.Series((base_signal > np.roll(base_signal, 1)).astype(int))

    # Убираем первые несколько строк из-за лагов
    X = X.iloc[10:]
    y = y.iloc[10:]

    return X.reset_index(drop=True), y.reset_index(drop=True)


def demo_lstm():
    """Демонстрация LSTM модели."""
    print("=" * 60)
    print("DEMO: LSTM Model")
    print("=" * 60)

    # Генерируем данные
    X, y = generate_synthetic_data(n_samples=1000, n_features=10)

    # Split
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))

    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_val = X.iloc[train_size : train_size + val_size]
    y_val = y.iloc[train_size : train_size + val_size]
    X_test = X.iloc[train_size + val_size :]
    y_test = y.iloc[train_size + val_size :]

    print(f"Train size: {len(X_train)}")
    print(f"Val size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")

    # Создаём модель
    model = LSTMModel(
        input_size=X.shape[1],
        hidden_size=64,
        num_layers=2,
        seq_length=30,
        output_size=1,
        dropout=0.2,
        task="classification",
        epochs=20,  # Мало для демо
        batch_size=32,
        learning_rate=0.001,
        early_stopping=5,
    )

    print(f"\nModel info: {model.get_model_info()}")

    # Обучаем
    print("\nОбучение модели...")
    model.fit(X_train, y_train, X_val, y_val)

    # Предсказания
    print("\nПредсказания на тестовых данных...")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # Метрики
    from sklearn.metrics import accuracy_score, roc_auc_score

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities[:, 1])

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test AUC: {auc:.4f}")

    # Визуализация
    history = model.get_training_history()
    plot_training_history(history, save_path="artifacts/lstm_training.png")
    print("\nГрафики сохранены в artifacts/")


def demo_gru():
    """Демонстрация GRU модели."""
    print("\n" + "=" * 60)
    print("DEMO: GRU Model")
    print("=" * 60)

    X, y = generate_synthetic_data(n_samples=1000, n_features=10)

    train_size = int(0.7 * len(X))
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]

    model = GRUModel(
        input_size=X.shape[1],
        hidden_size=64,
        num_layers=2,
        seq_length=30,
        epochs=10,
        task="classification",
    )

    print("Обучение GRU...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = (predictions == y_test.values[model.seq_length :]).mean()
    print(f"Test Accuracy: {accuracy:.4f}")


def demo_tcn():
    """Демонстрация TCN модели."""
    print("\n" + "=" * 60)
    print("DEMO: TCN Model")
    print("=" * 60)

    X, y = generate_synthetic_data(n_samples=1000, n_features=10)

    train_size = int(0.7 * len(X))
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]

    model = TCNModel(
        input_size=X.shape[1],
        hidden_size=64,
        num_layers=3,
        seq_length=30,
        kernel_size=3,
        epochs=10,
        task="classification",
    )

    print(f"TCN Receptive Field: {model.get_receptive_field()}")

    print("Обучение TCN...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = (predictions == y_test.values[model.seq_length :]).mean()
    print(f"Test Accuracy: {accuracy:.4f}")


def demo_cnn_lstm():
    """Демонстрация CNN+LSTM модели."""
    print("\n" + "=" * 60)
    print("DEMO: CNN+LSTM Model")
    print("=" * 60)

    X, y = generate_synthetic_data(n_samples=1000, n_features=10)

    train_size = int(0.7 * len(X))
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]

    model = CNNLSTMModel(
        input_size=X.shape[1],
        hidden_size=64,
        num_layers=2,
        seq_length=30,
        cnn_channels=[16, 32, 64],
        kernel_size=3,
        epochs=10,
        task="classification",
    )

    print("Обучение CNN+LSTM...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = (predictions == y_test.values[model.seq_length :]).mean()
    print(f"Test Accuracy: {accuracy:.4f}")


def demo_augmentation():
    """Демонстрация data augmentation."""
    print("\n" + "=" * 60)
    print("DEMO: Data Augmentation")
    print("=" * 60)

    # Создаём тестовые данные
    X = torch.randn(16, 30, 10)  # (batch, seq_len, features)

    # Получаем аугментацию
    aug = get_default_augmentation(mode="medium")

    # Применяем
    X_aug = aug(X)

    # Вычисляем разницу
    diff = torch.abs(X - X_aug).mean()

    print(f"Original shape: {X.shape}")
    print(f"Augmented shape: {X_aug.shape}")
    print(f"Mean difference: {diff:.4f}")
    print("Аугментация применена успешно!")


def demo_comparison():
    """Сравнение всех моделей."""
    print("\n" + "=" * 60)
    print("DEMO: Model Comparison")
    print("=" * 60)

    X, y = generate_synthetic_data(n_samples=1000, n_features=10)

    train_size = int(0.7 * len(X))
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]

    models = {
        "LSTM": LSTMModel(
            input_size=X.shape[1],
            hidden_size=32,
            num_layers=1,
            seq_length=20,
            epochs=5,
        ),
        "GRU": GRUModel(
            input_size=X.shape[1],
            hidden_size=32,
            num_layers=1,
            seq_length=20,
            epochs=5,
        ),
        "TCN": TCNModel(
            input_size=X.shape[1],
            hidden_size=32,
            num_layers=2,
            seq_length=20,
            epochs=5,
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\nОбучение {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = (predictions == y_test.values[model.seq_length :]).mean()
        results[name] = accuracy
        print(f"{name} Test Accuracy: {accuracy:.4f}")

    # Лучшая модель
    best_model = max(results, key=results.get)
    print(f"\n🏆 Лучшая модель: {best_model} ({results[best_model]:.4f})")


if __name__ == "__main__":
    print("Sequential Models Demo\n")
    print("Демонстрация работы с sequential моделями для временных рядов")

    # Проверяем доступность CUDA
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Запускаем демо
    try:
        demo_lstm()
        demo_gru()
        demo_tcn()
        demo_cnn_lstm()
        demo_augmentation()
        demo_comparison()

        print("\n" + "=" * 60)
        print("Все демонстрации успешно завершены! ✅")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()
