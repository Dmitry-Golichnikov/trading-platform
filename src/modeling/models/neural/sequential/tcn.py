"""
TCN (Temporal Convolutional Network) для временных рядов.

TCN использует dilated causal convolutions для обработки последовательностей.
Преимущества: параллельное обучение (в отличие от RNN), длинная память.
"""

from typing import List

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from src.modeling.models.neural.sequential.base import BaseSequentialModel


class Chomp1d(nn.Module):
    """
    Удаляет последние элементы из последовательности для causal convolution.
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch, channels, seq_len)
        Returns:
            Trimmed tensor
        """
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Базовый блок TCN с dilated causal convolutions.

    Состоит из двух causal convolution слоёв с dropout и residual connection.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Первая causal convolution
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Вторая causal convolution
        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Sequential блок
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )

        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        """Инициализация весов."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch, channels, seq_len)
        Returns:
            Output tensor, same shape as input
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    TCN состоящая из нескольких TemporalBlock с увеличивающимся dilation.
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    n_inputs=in_channels,
                    n_outputs=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch, channels, seq_len)
        Returns:
            Output tensor, shape (batch, num_channels[-1], seq_len)
        """
        return self.network(x)


class TCNModel(BaseSequentialModel):
    """
    TCN модель для временных рядов.

    Использует dilated causal convolutions для обработки последовательностей.
    Эффективнее RNN для длинных последовательностей.

    Параметры:
        input_size: Размерность входных признаков
        hidden_size: Размер скрытых слоёв (базовый)
        num_layers: Количество TCN блоков
        seq_length: Длина входной последовательности
        output_size: Размер выхода
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

        self.kernel_size = kernel_size

        # Создаём список каналов для каждого слоя
        # Постепенно увеличиваем размер от input_size до hidden_size
        num_channels = [hidden_size] * num_layers

        # TCN
        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # Fully connected для выхода
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
        # TCN ожидает (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, input_size, seq_length)

        # TCN
        tcn_out = self.tcn(x)  # (batch, hidden_size, seq_length)

        # Берём последний временной шаг
        last_output = tcn_out[:, :, -1]  # (batch, hidden_size)

        # Fully connected
        output = self.fc(last_output)

        return output

    def get_receptive_field(self) -> int:
        """
        Вычислить receptive field TCN.

        Returns:
            Размер receptive field
        """
        return 1 + 2 * (self.kernel_size - 1) * (2**self.num_layers - 1)

    def get_model_info(self) -> dict:
        """Получить информацию о модели."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_type": "TCN",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "kernel_size": self.kernel_size,
            "seq_length": self.seq_length,
            "output_size": self.output_size,
            "receptive_field": self.get_receptive_field(),
            "dropout": self.dropout_rate,
            "task": self.task,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "device": str(self.device),
        }


class ResidualTCNModel(BaseSequentialModel):
    """
    TCN с дополнительными residual connections между блоками.

    Улучшает градиентный поток для глубоких сетей.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        seq_length: int,
        output_size: int = 1,
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

        self.kernel_size = kernel_size

        # Input projection
        self.input_proj = nn.Conv1d(input_size, hidden_size, 1)

        # TCN blocks
        self.tcn_blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation

            self.tcn_blocks.append(
                TemporalBlock(
                    n_inputs=hidden_size,
                    n_outputs=hidden_size,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    dropout=dropout,
                )
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
        """
        Forward pass с global residual connections.

        Args:
            x: Input tensor, shape (batch_size, seq_length, input_size)

        Returns:
            Output tensor, shape (batch_size, output_size)
        """
        # Transpose
        x = x.transpose(1, 2)  # (batch, input_size, seq_length)

        # Project to hidden_size
        x = self.input_proj(x)

        # TCN blocks с global residuals
        for i, block in enumerate(self.tcn_blocks):
            residual = x
            x = block(x)
            # Global residual каждые 2 блока
            if (i + 1) % 2 == 0 and i > 0:
                x = x + residual

        # Last timestep
        last_output = x[:, :, -1]

        # Output
        output = self.fc(last_output)

        return output
