"""
Утилиты для моделирования.

Включает функции для воспроизводимости, управления устройствами, и другие
вспомогательные функции.
"""

import logging
import os
import random
from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    import torch
except Exception:
    torch = None

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Установить seed для воспроизводимости результатов.

    Устанавливает seed для:
    - Python random
    - NumPy
    - PyTorch (CPU и GPU)

    Args:
        seed: Значение seed
        deterministic: Использовать детерминистические алгоритмы
            (медленнее, но воспроизводимо)

    Примеры:
        >>> set_seed(42)
        >>> # Все операции теперь детерминированы
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Для multi-GPU

    if deterministic:
        # Детерминистические алгоритмы (медленнее)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Для PyTorch >= 1.8
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
    else:
        # Быстрее, но не полностью детерминировано
        torch.backends.cudnn.benchmark = True

    logger.info(f"Seed установлен: {seed} (deterministic={deterministic})")


def get_device(device: Optional[Union[str, "torch.device"]] = None, fallback_to_cpu: bool = True) -> "torch.device":
    """
    Получить устройство для вычислений (CPU/GPU).

    Args:
        device: Явно указанное устройство ('cpu', 'cuda', 'cuda:0', etc)
            или None для автоопределения
        fallback_to_cpu: Если True, вернёт CPU при недоступности GPU

    Returns:
        torch.device

    Raises:
        RuntimeError: Если GPU недоступен и fallback_to_cpu=False

    Примеры:
        >>> device = get_device()  # Автоопределение
        >>> device = get_device('cuda:0')  # Конкретный GPU
    """
    if device is not None:
        if isinstance(device, str):
            device = torch.device(device)
        return device

    # Автоопределение
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(
            f"Используется GPU: {torch.cuda.get_device_name(0)} "
            f"(память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)"
        )
    elif fallback_to_cpu:
        device = torch.device("cpu")
        logger.warning("GPU недоступен, используется CPU")
    else:
        raise RuntimeError("GPU недоступен, а fallback_to_cpu=False")

    return device


def get_available_devices() -> list[str]:
    """
    Получить список доступных устройств.

    Returns:
        Список имён устройств (например, ['cpu', 'cuda:0', 'cuda:1'])

    Примеры:
        >>> devices = get_available_devices()
        >>> print(devices)
        ['cpu', 'cuda:0']
    """
    devices = ["cpu"]

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        devices.extend([f"cuda:{i}" for i in range(n_gpus)])

    return devices


def check_gpu_memory(device: Optional["torch.device"] = None) -> dict[str, float]:
    """
    Проверить использование памяти GPU.

    Args:
        device: Устройство для проверки (по умолчанию текущее GPU)

    Returns:
        Словарь с информацией о памяти (allocated, cached, total в GB)

    Примеры:
        >>> memory = check_gpu_memory()
        >>> print(f"Использовано: {memory['allocated']:.2f} GB")
    """
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "cached": 0.0, "total": 0.0}

    if device is None:
        device = torch.device("cuda")

    allocated = torch.cuda.memory_allocated(device) / 1e9
    cached = torch.cuda.memory_reserved(device) / 1e9
    total = torch.cuda.get_device_properties(device).total_memory / 1e9

    return {
        "allocated": allocated,
        "cached": cached,
        "total": total,
        "free": total - allocated,
    }


def clear_gpu_memory(device: Optional["torch.device"] = None) -> None:
    """
    Очистить кэш GPU памяти.

    Args:
        device: Устройство для очистки (по умолчанию все GPU)

    Примеры:
        >>> clear_gpu_memory()
        >>> # GPU память освобождена
    """
    if not torch.cuda.is_available():
        logger.warning("GPU недоступен, нечего очищать")
        return

    torch.cuda.empty_cache()

    if device:
        torch.cuda.synchronize(device)

    logger.info("GPU память очищена")


def count_parameters(model: "torch.nn.Module") -> dict[str, int]:
    """
    Подсчитать количество параметров в PyTorch модели.

    Args:
        model: PyTorch модель

    Returns:
        Словарь с количеством параметров (total, trainable, non_trainable)

    Примеры:
        >>> model = MyNeuralNetwork()
        >>> params = count_parameters(model)
        >>> print(f"Всего параметров: {params['total']:,}")
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    return {"total": total, "trainable": trainable, "non_trainable": non_trainable}


def ensure_reproducibility(seed: int = 42, disable_gpu_nondeterminism: bool = True) -> None:
    """
    Максимально обеспечить воспроизводимость экспериментов.

    Args:
        seed: Значение seed
        disable_gpu_nondeterminism: Отключить недетерминистические операции на GPU

    Примечание:
        Некоторые операции в PyTorch/CUDA могут быть недетерминистическими
        даже с этими настройками. Для полной воспроизводимости используйте CPU.

    Примеры:
        >>> ensure_reproducibility(seed=42)
    """
    set_seed(seed, deterministic=True)

    # Дополнительные переменные окружения для CUDA
    if disable_gpu_nondeterminism and torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        logger.info("Включен детерминистический режим CUDA")


def get_model_size(model_path: Path) -> float:
    """
    Получить размер сохранённой модели в MB.

    Args:
        model_path: Путь к файлу модели

    Returns:
        Размер в мегабайтах

    Примеры:
        >>> size = get_model_size(Path("model.pt"))
        >>> print(f"Размер модели: {size:.2f} MB")
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    size_bytes = model_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    return size_mb


def move_to_device(
    data: Union["torch.Tensor", dict, list, tuple], device: "torch.device"
) -> Union["torch.Tensor", dict, list, tuple]:
    """
    Рекурсивно переместить данные на устройство.

    Args:
        data: Данные (tensor, dict, list, tuple)
        device: Целевое устройство

    Returns:
        Данные на целевом устройстве

    Примеры:
        >>> batch = {"X": torch.tensor([1, 2, 3]), "y": torch.tensor([0, 1, 0])}
        >>> batch = move_to_device(batch, device)
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        moved = [move_to_device(item, device) for item in data]
        return type(data)(moved)
    else:
        return data


def log_system_info() -> None:
    """
    Вывести информацию о системе в лог.

    Примеры:
        >>> log_system_info()
    """
    logger.info("=" * 60)
    logger.info("Информация о системе:")
    logger.info(f"  PyTorch версия: {torch.__version__}")
    logger.info(f"  CUDA доступна: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"  CUDA версия: {torch.version.cuda}")
        logger.info(f"  Количество GPU: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            memory = props.total_memory / 1e9
            logger.info(f"    GPU {i}: {name} ({memory:.1f} GB)")

    logger.info(f"  Количество CPU: {os.cpu_count()}")
    logger.info("=" * 60)


def validate_split_sizes(train_size: float, val_size: float, test_size: float) -> None:
    """
    Проверить корректность размеров сплитов.

    Args:
        train_size: Размер train выборки
        val_size: Размер validation выборки
        test_size: Размер test выборки

    Raises:
        ValueError: Если размеры некорректны
    """
    total = train_size + val_size + test_size

    if not np.isclose(total, 1.0):
        raise ValueError(f"Сумма размеров должна быть равна 1.0, получено: {total:.4f}")

    if train_size <= 0 or val_size < 0 or test_size < 0:
        raise ValueError("Все размеры должны быть неотрицательными, train_size > 0")

    if train_size < 0.5:
        logger.warning(f"train_size слишком мал: {train_size:.2%}")


class TimingContext:
    """
    Контекстный менеджер для измерения времени выполнения.

    Примеры:
        >>> with TimingContext("Training") as timer:
        >>>     model.fit(X, y)
        >>> print(f"Время обучения: {timer.elapsed:.2f}s")
    """

    def __init__(self, name: str = "Operation", log_result: bool = True):
        """
        Args:
            name: Название операции
            log_result: Логировать результат
        """
        self.name = name
        self.log_result = log_result
        self.elapsed: Optional[float] = None
        self._start_time: Optional[float] = None

    def __enter__(self):
        """Начать отсчёт времени."""
        import time

        self._start_time = time.time()
        return self

    def __exit__(self, *args):
        """Завершить отсчёт времени."""
        import time

        self.elapsed = time.time() - self._start_time

        if self.log_result:
            logger.info(f"{self.name}: {self.elapsed:.2f}s")
