"""
LSTM (Long Short-Term Memory) модель для временных рядов.

LSTM - это рекуррентная нейронная сеть с механизмом "памяти",
которая хорошо справляется с долгосрочными зависимостями.
"""

from typing import Optional

import torch
import torch.nn as nn

from src.modeling.models.neural.sequential.base import BaseSequentialModel


class LSTMModel(BaseSequentialModel):
    """
    LSTM модель для временных рядов.

    Поддерживает:
    - Uni-directional и bi-directional LSTM
    - Несколько слоёв
    - Dropout между слоями
    - Классификация и регрессия

    Параметры:
        input_size: Размерность входных признаков
        hidden_size: Размер скрытого слоя LSTM
        num_layers: Количество LSTM слоёв
        seq_length: Длина входной последовательности
        output_size: Размер выхода (1 для бинарной, n_classes для мультикласса)
        dropout: Dropout rate (применяется между LSTM слоями)
        bidirectional: Использовать ли bidirectional LSTM
        task: 'classification' или 'regression'
        **kwargs: Дополнительные параметры для BaseSequentialModel
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        seq_length: int,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
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

        self.bidirectional = bidirectional

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Входы в формате (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0.0,  # Dropout только если >1 слоя
            bidirectional=bidirectional,
        )

        # Размер выхода LSTM (удваивается при bidirectional)
        lstm_output_size = hidden_size * (2 if bidirectional else 1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов модели."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0)

        for module in self.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch_size, seq_length, input_size)

        Returns:
            Output tensor, shape (batch_size, output_size)
        """
        # LSTM forward
        # lstm_out: (batch, seq_len, hidden_size * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_size)
        # c_n: (num_layers * num_directions, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Используем последний выход последовательности
        # lstm_out[:, -1, :] - последний временной шаг
        last_output = lstm_out[:, -1, :]

        # Fully connected layers
        output = self.fc(last_output)

        return output

    def get_model_info(self) -> dict:
        """Получить информацию о модели."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_type": "LSTM",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "seq_length": self.seq_length,
            "output_size": self.output_size,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout_rate,
            "task": self.task,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "device": str(self.device),
        }


class StackedLSTMModel(BaseSequentialModel):
    """
    Stacked LSTM с residual connections для глубоких сетей.

    Добавляет skip connections между LSTM слоями для лучшего
    обучения глубоких моделей.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        seq_length: int,
        output_size: int = 1,
        dropout: float = 0.2,
        task: str = "classification",
        **kwargs,
    ):
        if num_layers < 2:
            raise ValueError("StackedLSTMModel требует минимум 2 слоя")

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

        # Первый LSTM слой
        self.lstm_layers = nn.ModuleList()
        self.lstm_layers.append(
            nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
        )

        # Остальные LSTM слои
        for _ in range(num_layers - 1):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                )
            )

        # Dropout слои
        self.dropout = nn.Dropout(dropout)

        # Fully connected
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass с residual connections.

        Args:
            x: Input tensor, shape (batch_size, seq_length, input_size)

        Returns:
            Output tensor, shape (batch_size, output_size)
        """
        # Первый слой
        out, _ = self.lstm_layers[0](x)
        out = self.dropout(out)

        # Остальные слои с residual connections
        for lstm_layer in self.lstm_layers[1:]:
            residual = out
            out, _ = lstm_layer(out)
            out = self.dropout(out)
            # Residual connection
            out = out + residual

        # Последний временной шаг
        last_output = out[:, -1, :]

        # Fully connected
        output = self.fc(last_output)

        return output


class AttentionLSTMModel(BaseSequentialModel):
    """
    LSTM с attention mechanism.

    Attention позволяет модели фокусироваться на наиболее важных
    временных шагах в последовательности.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        seq_length: int,
        output_size: int = 1,
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

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # Fully connected
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

        self.attention_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass с attention.

        Args:
            x: Input tensor, shape (batch_size, seq_length, input_size)

        Returns:
            Output tensor, shape (batch_size, output_size)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)

        # Attention scores
        attention_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, seq_len, 1)

        # Сохраняем веса для визуализации
        self.attention_weights = attention_weights.detach()

        # Взвешенная сумма
        context = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, hidden)

        # Fully connected
        output = self.fc(context)

        return output

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Получить attention веса последнего forward pass."""
        return self.attention_weights
