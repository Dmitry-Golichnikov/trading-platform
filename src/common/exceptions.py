"""
Пользовательские исключения для модулей проекта.
"""


class ProjectBaseException(Exception):
    """Базовое исключение для всех кастомных исключений проекта."""

    pass


class DataLoadError(ProjectBaseException):
    """Ошибка при загрузке данных из источника."""

    pass


class DataValidationError(ProjectBaseException):
    """Ошибка валидации данных."""

    pass


class StorageError(ProjectBaseException):
    """Ошибка при работе с хранилищем данных."""

    pass


class CatalogError(ProjectBaseException):
    """Ошибка при работе с каталогом датасетов."""

    pass


class PreprocessingError(ProjectBaseException):
    """Ошибка при предобработке данных."""

    pass
