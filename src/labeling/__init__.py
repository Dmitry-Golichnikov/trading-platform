"""Модуль разметки таргетов."""

from src.labeling.base import BaseLabeler
from src.labeling.metadata import LabelingMetadata, generate_labeling_id
from src.labeling.pipeline import LabelingPipeline

__all__ = [
    "BaseLabeler",
    "LabelingPipeline",
    "LabelingMetadata",
    "generate_labeling_id",
]
