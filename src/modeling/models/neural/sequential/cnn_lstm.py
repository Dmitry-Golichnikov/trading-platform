"""
CNN+LSTM гибридная модель для временных рядов.

CNN извлекает локальные паттерны, LSTM моделирует долгосрочные зависимости.
Комбинация преимуществ обоих подходов.
"""

from typing import Optional

import torch
import torch.nn as nn

from src.modeling.models.neural.sequential.base import BaseSequentialModel


class CNNLSTMModel(BaseSequentialModel):
    """
    Гибридная CNN+LSTM модель.

    CNN слои извлекают локальные паттерны из последовательности,
    затем LSTM обрабатывает эти признаки для долгосрочных зависимостей.

    Параметры:
        input_size: Размерность входных признаков
        hidden_size: Размер скрытого слоя LSTM
        num_layers: Количество LSTM слоёв
        seq_length: Длина входной последовательности
        output_size: Размер выхода
        cnn_channels: Список размеров каналов для CNN слоёв
        kernel_size: Размер ядра свёртки
        dropout: Dropout rate
        task: 'classification' или 'regression'
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        seq_length: int,
        output_size: int = 1,
        cnn_channels: Optional[list[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        task: str = "classification",
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            seq_length=seq_length,
            output_size=output_size,
            dropout=dropout,
            task=task,
            **kwargs,
        )

        if cnn_channels is None:
            cnn_channels = [32, 64, 128]

        self.cnn_channels = cnn_channels
        self.kernel_size = kernel_size

        # CNN layers
        cnn_layers = []
        in_channels = input_size

        for out_channels in cnn_channels:
            cnn_layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,  # same padding
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],  # Выход CNN
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Fully connected
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch_size, seq_length, input_size)

        Returns:
            Output tensor, shape (batch_size, output_size)
        """
        # CNN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, input_size, seq_length)

        # CNN feature extraction
        cnn_out = self.cnn(x)  # (batch, cnn_channels[-1], seq_length)

        # Transpose back for LSTM
        cnn_out = cnn_out.transpose(1, 2)  # (batch, seq_length, cnn_channels[-1])

        # LSTM
        lstm_out, _ = self.lstm(cnn_out)  # (batch, seq_length, hidden_size)

        # Last timestep
        last_output = lstm_out[:, -1, :]

        # Output
        output = self.fc(last_output)

        return output

    def get_model_info(self) -> dict:
        """Получить информацию о модели."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_type": "CNN+LSTM",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "cnn_channels": self.cnn_channels,
            "kernel_size": self.kernel_size,
            "seq_length": self.seq_length,
            "output_size": self.output_size,
            "dropout": self.dropout_rate,
            "task": self.task,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "device": str(self.device),
        }


class CNNGRUModel(BaseSequentialModel):
    """
    Гибридная CNN+GRU модель.

    Аналогична CNN+LSTM, но использует GRU для более быстрого обучения.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        seq_length: int,
        output_size: int = 1,
        cnn_channels: Optional[list[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        task: str = "classification",
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            seq_length=seq_length,
            output_size=output_size,
            dropout=dropout,
            task=task,
            **kwargs,
        )

        if cnn_channels is None:
            cnn_channels = [32, 64, 128]

        self.cnn_channels = cnn_channels
        self.kernel_size = kernel_size

        # CNN
        cnn_layers = []
        in_channels = input_size

        for out_channels in cnn_channels:
            cnn_layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # GRU
        self.gru = nn.GRU(
            input_size=cnn_channels[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x.transpose(1, 2)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.transpose(1, 2)
        gru_out, _ = self.gru(cnn_out)
        last_output = gru_out[:, -1, :]
        output = self.fc(last_output)
        return output


class ResidualCNNLSTMModel(BaseSequentialModel):
    """
    CNN+LSTM с residual connections.

    Добавляет skip connections для лучшего градиентного потока.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        seq_length: int,
        output_size: int = 1,
        cnn_channels: Optional[list[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        task: str = "classification",
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            seq_length=seq_length,
            output_size=output_size,
            dropout=dropout,
            task=task,
            **kwargs,
        )

        if cnn_channels is None:
            cnn_channels = [64, 128, 256]

        self.cnn_channels = cnn_channels

        # Residual CNN blocks
        self.cnn_blocks = nn.ModuleList()
        in_channels = input_size

        for out_channels in cnn_channels:
            block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
            )
            self.cnn_blocks.append(block)

            # Projection для residual connection
            if in_channels != out_channels:
                self.cnn_blocks.append(nn.Conv1d(in_channels, out_channels, 1))
            else:
                self.cnn_blocks.append(nn.Identity())

            in_channels = out_channels

        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass с residual connections."""
        x = x.transpose(1, 2)

        # Residual CNN blocks
        for i in range(0, len(self.cnn_blocks), 2):
            residual = x
            x = self.cnn_blocks[i](x)

            # Projection если нужно
            if not isinstance(self.cnn_blocks[i + 1], nn.Identity):
                residual = self.cnn_blocks[i + 1](residual)

            x = nn.ReLU()(x + residual)

        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)

        return output


class MultiScaleCNNLSTMModel(BaseSequentialModel):
    """
    Multi-scale CNN+LSTM модель.

    Использует несколько CNN веток с разными kernel sizes
    для захвата паттернов на разных масштабах.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        seq_length: int,
        output_size: int = 1,
        kernel_sizes: Optional[list[int]] = None,
        cnn_channels: int = 64,
        dropout: float = 0.2,
        task: str = "classification",
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            seq_length=seq_length,
            output_size=output_size,
            dropout=dropout,
            task=task,
            **kwargs,
        )

        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]  # Разные масштабы

        self.kernel_sizes = kernel_sizes

        # Multi-scale CNN branches
        self.cnn_branches = nn.ModuleList()
        for ks in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(input_size, cnn_channels, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(cnn_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.cnn_branches.append(branch)

        # Concatenated features
        combined_channels = cnn_channels * len(kernel_sizes)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=combined_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass с multi-scale features."""
        x = x.transpose(1, 2)  # (batch, input_size, seq_len)

        # Apply all CNN branches
        branch_outputs = []
        for branch in self.cnn_branches:
            branch_out = branch(x)
            branch_outputs.append(branch_out)

        # Concatenate along channel dimension
        multi_scale = torch.cat(branch_outputs, dim=1)  # (batch, combined_channels, seq_len)

        # Transpose for LSTM
        multi_scale = multi_scale.transpose(1, 2)

        # LSTM
        lstm_out, _ = self.lstm(multi_scale)
        last_output = lstm_out[:, -1, :]

        # Output
        output = self.fc(last_output)

        return output

    def get_model_info(self) -> dict:
        """Получить информацию о модели."""
        info = super().get_model_info()
        info["model_type"] = "MultiScaleCNN+LSTM"
        info["kernel_sizes"] = self.kernel_sizes
        return info
