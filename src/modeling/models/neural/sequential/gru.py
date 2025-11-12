"""
GRU (Gated Recurrent Unit) модель для временных рядов.

GRU - это упрощённая версия LSTM, которая обычно обучается быстрее
и требует меньше памяти, сохраняя сопоставимое качество.
"""

from typing import Optional

import torch
import torch.nn as nn

from src.modeling.models.neural.sequential.base import BaseSequentialModel


class GRUModel(BaseSequentialModel):
    """
    GRU модель для временных рядов.

    GRU имеет более простую архитектуру чем LSTM (2 гейта вместо 3),
    что делает её быстрее в обучении при схожем качестве.

    Параметры:
        input_size: Размерность входных признаков
        hidden_size: Размер скрытого слоя GRU
        num_layers: Количество GRU слоёв
        seq_length: Длина входной последовательности
        output_size: Размер выхода
        dropout: Dropout rate
        bidirectional: Использовать ли bidirectional GRU
        task: 'classification' или 'regression'
        **kwargs: Дополнительные параметры
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

        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Размер выхода GRU
        gru_output_size = hidden_size * (2 if bidirectional else 1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов модели."""
        for name, param in self.gru.named_parameters():
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
        # GRU forward
        gru_out, h_n = self.gru(x)

        # Используем последний выход
        last_output = gru_out[:, -1, :]

        # Fully connected
        output = self.fc(last_output)

        return output

    def get_model_info(self) -> dict:
        """Получить информацию о модели."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_type": "GRU",
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


class AttentionGRUModel(BaseSequentialModel):
    """
    GRU с attention mechanism.

    Комбинирует эффективность GRU с возможностью attention
    фокусироваться на важных временных шагах.
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

        # GRU
        self.gru = nn.GRU(
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
        # GRU
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden)

        # Attention
        attention_scores = self.attention(gru_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Сохраняем для визуализации
        self.attention_weights = attention_weights.detach()

        # Weighted sum
        context = torch.sum(gru_out * attention_weights, dim=1)

        # Output
        output = self.fc(context)

        return output

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Получить attention веса последнего forward pass."""
        return self.attention_weights


class MultiHeadAttentionGRUModel(BaseSequentialModel):
    """
    GRU с multi-head attention.

    Использует несколько attention heads для захвата
    различных аспектов последовательности.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        seq_length: int,
        output_size: int = 1,
        dropout: float = 0.2,
        num_heads: int = 4,
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

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) должен делиться на num_heads ({num_heads})")

        self.num_heads = num_heads

        # GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

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
        Forward pass с multi-head attention.

        Args:
            x: Input tensor, shape (batch_size, seq_length, input_size)

        Returns:
            Output tensor, shape (batch_size, output_size)
        """
        # GRU
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden)

        # Multi-head attention (self-attention)
        attn_out, _ = self.multihead_attn(
            query=gru_out,
            key=gru_out,
            value=gru_out,
        )

        # Residual connection + layer norm
        attn_out = self.layer_norm(gru_out + attn_out)

        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)  # (batch, hidden)

        # Output
        output = self.fc(pooled)

        return output

    def get_model_info(self) -> dict:
        """Получить информацию о модели."""
        info = super().get_model_info()
        info["model_type"] = "MultiHeadAttentionGRU"
        info["num_heads"] = self.num_heads
        return info
