from __future__ import annotations

from PySide6.QtGui import QStandardItem, QStandardItemModel


class IndicatorTreeModel(QStandardItemModel):
    """Дерево категорий индикаторов."""

    def __init__(self) -> None:
        super().__init__()
        self.setHorizontalHeaderLabels(["Индикаторы"])

    def populate(self, catalog: dict[str, list[str]]) -> None:
        self.clear()
        self.setHorizontalHeaderLabels(["Индикаторы"])
        for category, indicators in catalog.items():
            category_item = QStandardItem(category)
            category_item.setEditable(False)
            for indicator in indicators:
                indicator_item = QStandardItem(indicator)
                indicator_item.setEditable(False)
                category_item.appendRow(indicator_item)
            self.appendRow(category_item)
