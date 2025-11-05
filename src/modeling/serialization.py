"""
Model Serialization.

Сохранение и загрузка моделей в различных форматах.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import joblib
import torch

from src.modeling.base import BaseModel

logger = logging.getLogger(__name__)


class ModelSerializer:
    """
    Класс для сериализации и десериализации моделей.

    Поддерживает различные форматы: pickle, joblib, PyTorch, ONNX.
    """

    SUPPORTED_FORMATS = ["pickle", "joblib", "torch", "onnx"]

    @staticmethod
    def save(
        model: BaseModel,
        path: Path,
        format: Literal["pickle", "joblib", "torch", "onnx", "auto"] = "auto",
        save_metadata: bool = True,
        **kwargs,
    ) -> None:
        """
        Сохранить модель.

        Args:
            model: Модель для сохранения
            path: Путь для сохранения
            format: Формат сохранения ('pickle', 'joblib', 'torch', 'onnx', 'auto')
            save_metadata: Сохранить метаданные вместе с моделью
            **kwargs: Дополнительные параметры для формата

        Примеры:
            >>> ModelSerializer.save(model, Path('model.pkl'))
            >>> ModelSerializer.save(model, Path('model.pt'), format='torch')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Автоопределение формата по расширению
        if format == "auto":
            chosen_format = ModelSerializer._detect_format_from_path(path)
        else:
            chosen_format = format

        # Проверяем формат
        if chosen_format not in ModelSerializer.SUPPORTED_FORMATS:
            raise ValueError(
                f"Неподдерживаемый формат: {chosen_format}. "
                f"Доступные: {ModelSerializer.SUPPORTED_FORMATS}"
            )

        logger.info(
            f"Сохранение модели {model.__class__.__name__} в {path} ({chosen_format})"
        )

        # Сохраняем в зависимости от формата
        if chosen_format == "pickle":
            ModelSerializer._save_pickle(model, path, **kwargs)
        elif chosen_format == "joblib":
            ModelSerializer._save_joblib(model, path, **kwargs)
        elif chosen_format == "torch":
            ModelSerializer._save_torch(model, path, **kwargs)
        elif chosen_format == "onnx":
            ModelSerializer._save_onnx(model, path, **kwargs)

        # Сохраняем метаданные
        if save_metadata:
            metadata_path = path.parent / f"{path.stem}_metadata.json"
            ModelSerializer._save_metadata(model, metadata_path)

    @staticmethod
    def load(
        path: Path,
        format: Literal["pickle", "joblib", "torch", "onnx", "auto"] = "auto",
        load_metadata: bool = True,
        **kwargs,
    ) -> BaseModel:
        """
        Загрузить модель.

        Args:
            path: Путь к сохранённой модели
            format: Формат загрузки ('pickle', 'joblib', 'torch', 'onnx', 'auto')
            load_metadata: Загрузить метаданные если доступны
            **kwargs: Дополнительные параметры для формата

        Returns:
            Загруженная модель

        Примеры:
            >>> model = ModelSerializer.load(Path('model.pkl'))
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Модель не найдена: {path}")

        # Автоопределение формата
        if format == "auto":
            chosen_format = ModelSerializer._detect_format_from_path(path)
        else:
            chosen_format = format

        logger.info(f"Загрузка модели из {path} ({chosen_format})")

        # Загружаем в зависимости от формата
        if chosen_format == "pickle":
            model = ModelSerializer._load_pickle(path, **kwargs)
        elif chosen_format == "joblib":
            model = ModelSerializer._load_joblib(path, **kwargs)
        elif chosen_format == "torch":
            model = ModelSerializer._load_torch(path, **kwargs)
        elif chosen_format == "onnx":
            model = ModelSerializer._load_onnx(path, **kwargs)
        else:
            raise ValueError(f"Неподдерживаемый формат: {chosen_format}")

        # Загружаем метаданные если есть
        if load_metadata:
            metadata_path = path.parent / f"{path.stem}_metadata.json"
            if metadata_path.exists():
                metadata = ModelSerializer._load_metadata(metadata_path)
                if hasattr(model, "metadata"):
                    model.metadata.update(metadata)

        return model

    @staticmethod
    def _detect_format_from_path(path: Path) -> str:
        """Определить формат по расширению файла."""
        ext = path.suffix.lower()

        if ext in [".pkl", ".pickle"]:
            return "pickle"
        elif ext in [".joblib", ".jbl"]:
            return "joblib"
        elif ext in [".pt", ".pth"]:
            return "torch"
        elif ext == ".onnx":
            return "onnx"
        else:
            # Дефолт - pickle
            logger.warning(f"Неизвестное расширение {ext}, использую pickle")
            return "pickle"

    @staticmethod
    def _save_pickle(model: BaseModel, path: Path, **kwargs) -> None:
        """Сохранить в pickle."""
        with open(path, "wb") as f:
            pickle.dump(model, f, **kwargs)

    @staticmethod
    def _load_pickle(path: Path, **kwargs) -> BaseModel:
        """Загрузить из pickle."""
        with open(path, "rb") as f:
            return pickle.load(f, **kwargs)

    @staticmethod
    def _save_joblib(model: BaseModel, path: Path, **kwargs) -> None:
        """Сохранить через joblib (лучше для больших numpy arrays)."""
        joblib.dump(model, path, **kwargs)

    @staticmethod
    def _load_joblib(path: Path, **kwargs) -> BaseModel:
        """Загрузить через joblib."""
        return joblib.load(path, **kwargs)

    @staticmethod
    def _save_torch(model: BaseModel, path: Path, **kwargs) -> None:
        """Сохранить PyTorch модель."""
        if hasattr(model, "state_dict"):
            # Сохраняем state_dict + метаданные
            save_dict = {
                "state_dict": model.state_dict(),
                "hyperparams": (
                    model.hyperparams if hasattr(model, "hyperparams") else {}
                ),
                "metadata": model.metadata if hasattr(model, "metadata") else {},
                "model_class": model.__class__.__name__,
            }
            torch.save(save_dict, path, **kwargs)
        else:
            # Fallback to pickle
            logger.warning(
                f"Модель {model.__class__.__name__} не имеет state_dict, "
                f"использую pickle"
            )
            ModelSerializer._save_pickle(model, path, **kwargs)

    @staticmethod
    def _load_torch(
        path: Path, model_class: Optional[type] = None, **kwargs
    ) -> BaseModel:
        """
        Загрузить PyTorch модель.

        Args:
            path: Путь к модели
            model_class: Класс модели (для восстановления архитектуры)
            **kwargs: Дополнительные параметры
        """
        checkpoint = torch.load(path, **kwargs)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            # Загружаем из нашего формата
            if model_class is None:
                raise ValueError(
                    "Для загрузки PyTorch модели необходимо указать model_class"
                )

            # Создаём экземпляр модели
            hyperparams = checkpoint.get("hyperparams", {})
            model = model_class(**hyperparams)

            # Загружаем веса
            model.load_state_dict(checkpoint["state_dict"])

            # Восстанавливаем метаданные
            if hasattr(model, "metadata"):
                model.metadata.update(checkpoint.get("metadata", {}))

            return model
        else:
            # Прямая загрузка (старый формат)
            return checkpoint

    @staticmethod
    def _save_onnx(model: BaseModel, path: Path, **kwargs) -> None:
        """
        Экспортировать в ONNX (только для PyTorch моделей).

        Requires:
            - dummy_input: Пример входных данных для трассировки
        """
        if not hasattr(model, "state_dict"):
            raise ValueError(
                f"ONNX export поддерживается только для PyTorch моделей. "
                f"Модель {model.__class__.__name__} не является PyTorch моделью."
            )

        dummy_input = kwargs.pop("dummy_input", None)
        if dummy_input is None:
            raise ValueError(
                "Для ONNX export необходимо указать dummy_input "
                "(пример входных данных)"
            )

        # Экспорт в ONNX
        torch.onnx.export(model, dummy_input, path, **kwargs)

        logger.info(f"Модель экспортирована в ONNX: {path}")

    @staticmethod
    def _load_onnx(path: Path, **kwargs) -> Any:
        """
        Загрузить ONNX модель.

        Требует onnxruntime.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "Для загрузки ONNX моделей необходимо установить onnxruntime: "
                "pip install onnxruntime"
            )

        session = ort.InferenceSession(str(path), **kwargs)
        logger.info(f"ONNX модель загружена: {path}")

        return session

    @staticmethod
    def _save_metadata(model: BaseModel, path: Path) -> None:
        """Сохранить метаданные модели в JSON."""
        metadata = {
            "model_class": model.__class__.__name__,
            "hyperparams": model.get_params() if hasattr(model, "get_params") else {},
            "metadata": model.get_metadata() if hasattr(model, "get_metadata") else {},
        }

        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    @staticmethod
    def _load_metadata(path: Path) -> Dict[str, Any]:
        """Загрузить метаданные модели из JSON."""
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def get_model_info(path: Path) -> Dict[str, Any]:
        """
        Получить информацию о сохранённой модели без загрузки.

        Args:
            path: Путь к модели

        Returns:
            Словарь с информацией о модели

        Примеры:
            >>> info = ModelSerializer.get_model_info(Path('model.pkl'))
            >>> print(info['model_class'], info['size_mb'])
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Модель не найдена: {path}")

        info = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "size_mb": path.stat().st_size / (1024 * 1024),
            "format": ModelSerializer._detect_format_from_path(path),
        }

        # Пытаемся загрузить метаданные
        metadata_path = path.parent / f"{path.stem}_metadata.json"
        if metadata_path.exists():
            try:
                metadata = ModelSerializer._load_metadata(metadata_path)
                info.update(metadata)
            except Exception as e:
                logger.warning(f"Не удалось загрузить метаданные: {e}")

        return info
