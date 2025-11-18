"""
Data augmentation для sequence данных временных рядов.

Включает различные техники аугментации, которые не нарушают
каузальность и сохраняют паттерны данных.
"""

import numpy as np
import torch


class SequenceAugmentor:
    """
    Базовый класс для аугментации последовательностей.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Применить аугментацию."""
        raise NotImplementedError


class Jitter(SequenceAugmentor):
    """
    Добавить случайный шум к данным.

    Args:
        sigma: Стандартное отклонение шума
    """

    def __init__(self, sigma: float = 0.03):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Добавить Gaussian noise.

        Args:
            x: shape (batch, seq_len, features) или (seq_len, features)

        Returns:
            Augmented tensor
        """
        noise = torch.randn_like(x) * self.sigma
        return x + noise


class Scaling(SequenceAugmentor):
    """
    Масштабировать значения случайным коэффициентом.

    Args:
        sigma: Стандартное отклонение для log-normal распределения
    """

    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Применить случайное масштабирование.

        Args:
            x: shape (batch, seq_len, features) или (seq_len, features)

        Returns:
            Scaled tensor
        """
        # Log-normal распределение для положительных коэффициентов
        scale_factor = torch.exp(torch.randn_like(x) * self.sigma)
        return x * scale_factor


class MagnitudeWarp(SequenceAugmentor):
    """
    Smooth random warping of magnitude.

    Args:
        sigma: Стандартное отклонение
        knots: Количество контрольных точек для сплайна
    """

    def __init__(self, sigma: float = 0.2, knots: int = 4):
        self.sigma = sigma
        self.knots = knots

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Применить magnitude warping.

        Args:
            x: shape (batch, seq_len, features) или (seq_len, features)

        Returns:
            Warped tensor
        """
        orig_shape = x.shape
        if len(orig_shape) == 2:
            x = x.unsqueeze(0)

        batch_size, seq_len, n_features = x.shape

        # Создаём smooth warping curve
        warp_steps = np.arange(0, seq_len, seq_len // self.knots)
        warp_steps = np.append(warp_steps, seq_len - 1)

        # Random warping values
        random_warps = np.random.normal(loc=1.0, scale=self.sigma, size=(batch_size, len(warp_steps)))

        # Интерполируем для всех timesteps
        warper = np.zeros((batch_size, seq_len))
        for i in range(batch_size):
            warper[i] = np.interp(np.arange(seq_len), warp_steps, random_warps[i])

        # Применяем warping
        warper = torch.FloatTensor(warper).to(x.device)
        warper = warper.unsqueeze(-1)  # (batch, seq_len, 1)

        result = x * warper

        if len(orig_shape) == 2:
            result = result.squeeze(0)

        return result


class TimeWarp(SequenceAugmentor):
    """
    Деформация временной оси (speed up / slow down).

    Args:
        sigma: Стандартное отклонение для деформации
        knots: Количество контрольных точек
    """

    def __init__(self, sigma: float = 0.2, knots: int = 4):
        self.sigma = sigma
        self.knots = knots

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Применить time warping (изменяет скорость последовательности).

        Args:
            x: shape (batch, seq_len, features) или (seq_len, features)

        Returns:
            Time-warped tensor
        """
        orig_shape = x.shape
        if len(orig_shape) == 2:
            x = x.unsqueeze(0)

        batch_size, seq_len, n_features = x.shape

        result = torch.zeros_like(x)

        for i in range(batch_size):
            # Создаём warping mapping
            warp_steps = np.arange(0, seq_len, seq_len // self.knots)
            warp_steps = np.append(warp_steps, seq_len - 1)

            random_warps = np.random.normal(loc=1.0, scale=self.sigma, size=len(warp_steps))
            random_warps = np.cumsum(random_warps)
            random_warps = (random_warps / random_warps[-1]) * (seq_len - 1)

            # Интерполируем
            time_warp = np.interp(np.arange(seq_len), warp_steps, random_warps)

            # Применяем warping через интерполяцию
            for j in range(n_features):
                result[i, :, j] = torch.FloatTensor(
                    np.interp(time_warp, np.arange(seq_len), x[i, :, j].cpu().numpy())
                ).to(x.device)

        if len(orig_shape) == 2:
            result = result.squeeze(0)

        return result


class WindowSlicing(SequenceAugmentor):
    """
    Выбрать случайное окно из последовательности.

    Args:
        reduce_ratio: Насколько уменьшить последовательность (0 < ratio < 1)
    """

    def __init__(self, reduce_ratio: float = 0.9):
        self.reduce_ratio = reduce_ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Выбрать случайное окно.

        Args:
            x: shape (batch, seq_len, features) или (seq_len, features)

        Returns:
            Sliced tensor (может быть короче)
        """
        orig_shape = x.shape
        if len(orig_shape) == 2:
            x = x.unsqueeze(0)

        batch_size, seq_len, n_features = x.shape

        target_len = int(seq_len * self.reduce_ratio)
        if target_len < 1:
            target_len = 1

        result = []
        for i in range(batch_size):
            start = np.random.randint(0, seq_len - target_len + 1)
            end = start + target_len
            result.append(x[i, start:end, :])

        result = torch.stack(result)

        if len(orig_shape) == 2:
            result = result.squeeze(0)

        return result


class RandomCrop(SequenceAugmentor):
    """
    Обрезать последовательность до заданной длины.

    Args:
        crop_size: Размер обрезанной последовательности
    """

    def __init__(self, crop_size: int):
        self.crop_size = crop_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Случайная обрезка последовательности.

        Args:
            x: shape (batch, seq_len, features) или (seq_len, features)

        Returns:
            Cropped tensor
        """
        orig_shape = x.shape
        if len(orig_shape) == 2:
            x = x.unsqueeze(0)

        batch_size, seq_len, n_features = x.shape

        if seq_len <= self.crop_size:
            if len(orig_shape) == 2:
                x = x.squeeze(0)
            return x

        result = []
        for i in range(batch_size):
            start = np.random.randint(0, seq_len - self.crop_size + 1)
            result.append(x[i, start : start + self.crop_size, :])

        result = torch.stack(result)

        if len(orig_shape) == 2:
            result = result.squeeze(0)

        return result


class ComposedAugmentation(SequenceAugmentor):
    """
    Композиция нескольких аугментаций.

    Args:
        augmentations: Список аугментаций для применения
        p: Вероятность применения каждой аугментации
    """

    def __init__(self, augmentations: list[SequenceAugmentor], p: float = 0.5):
        self.augmentations = augmentations
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Применить случайный набор аугментаций.

        Args:
            x: Input tensor

        Returns:
            Augmented tensor
        """
        for aug in self.augmentations:
            if np.random.random() < self.p:
                x = aug(x)

        return x


def get_default_augmentation(mode: str = "light") -> ComposedAugmentation:
    """
    Получить стандартный набор аугментаций.

    Args:
        mode: 'light', 'medium', или 'heavy'

    Returns:
        ComposedAugmentation
    """
    if mode == "light":
        return ComposedAugmentation(
            [
                Jitter(sigma=0.03),
                Scaling(sigma=0.1),
            ],
            p=0.3,
        )

    elif mode == "medium":
        return ComposedAugmentation(
            [
                Jitter(sigma=0.05),
                Scaling(sigma=0.15),
                MagnitudeWarp(sigma=0.2, knots=4),
            ],
            p=0.5,
        )

    elif mode == "heavy":
        return ComposedAugmentation(
            [
                Jitter(sigma=0.08),
                Scaling(sigma=0.2),
                MagnitudeWarp(sigma=0.3, knots=4),
                TimeWarp(sigma=0.2, knots=4),
            ],
            p=0.5,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")


# Пример использования
if __name__ == "__main__":
    # Создаём тестовые данные
    x = torch.randn(32, 50, 10)  # (batch, seq_len, features)

    # Применяем аугментацию
    aug = get_default_augmentation(mode="medium")
    x_aug = aug(x)

    print(f"Original shape: {x.shape}")
    print(f"Augmented shape: {x_aug.shape}")
    print(f"Mean difference: {torch.mean(torch.abs(x - x_aug)):.4f}")
