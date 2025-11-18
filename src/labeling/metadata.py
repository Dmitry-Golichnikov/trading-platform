"""Метаданные разметки."""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class LabelingMetadata:
    """Метаданные разметки таргетов."""

    labeling_id: str
    method: str
    config: Dict[str, Any]
    dataset_id: str
    total_samples: int
    class_distribution: Dict[int, int]
    filters_applied: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"

    # Дополнительная статистика
    statistics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразование в словарь.

        Returns:
            Словарь с метаданными
        """
        data = asdict(self)
        # Преобразуем datetime в строку
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LabelingMetadata":
        """
        Создание из словаря.

        Args:
            data: Словарь с метаданными

        Returns:
            Объект LabelingMetadata
        """
        # Преобразуем строку обратно в datetime
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])

        return cls(**data)

    def save(self, path: Path) -> None:
        """
        Сохранение метаданных в JSON файл.

        Args:
            path: Путь к файлу
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Метаданные разметки сохранены: {path}")

    @classmethod
    def load(cls, path: Path) -> "LabelingMetadata":
        """
        Загрузка метаданных из JSON файла.

        Args:
            path: Путь к файлу

        Returns:
            Объект LabelingMetadata
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Метаданные разметки загружены: {path}")

        return cls.from_dict(data)

    def add_statistics(self, key: str, value: Any) -> None:
        """
        Добавление статистики.

        Args:
            key: Ключ статистики
            value: Значение
        """
        self.statistics[key] = value

    def get_summary(self) -> str:
        """
        Получить краткое резюме метаданных.

        Returns:
            Строка с резюме
        """
        filters_repr = ", ".join(self.filters_applied) if self.filters_applied else "None"

        summary = [
            f"Labeling ID: {self.labeling_id}",
            f"Method: {self.method}",
            f"Dataset ID: {self.dataset_id}",
            f"Total samples: {self.total_samples}",
            f"Class distribution: {self.class_distribution}",
            f"Filters applied: {filters_repr}",
            f"Created at: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Version: {self.version}",
        ]

        if self.statistics:
            summary.append("Statistics:")
            for key, value in self.statistics.items():
                summary.append(f"  {key}: {value}")

        return "\n".join(summary)


def generate_labeling_id(method: str, dataset_id: str) -> str:
    """
    Генерация уникального ID разметки.

    Args:
        method: Метод разметки
        dataset_id: ID датасета

    Returns:
        Уникальный ID
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{method}_{dataset_id}_{timestamp}"
