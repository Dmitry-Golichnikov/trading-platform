"""
TFT (Temporal Fusion Transformer) для временных рядов.

Упрощённая реализация TFT с основными компонентами:
- Variable selection
- Temporal processing (LSTM)
- Multi-head attention
- Gating mechanisms

Полная реализация TFT очень сложна, эта версия содержит ключевые идеи.
"""

from typing import Optional

import torch
import torch.nn as nn

from src.modeling.models.neural.sequential.base import BaseSequentialModel


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) - ключевой компонент TFT.

    Адаптивно применяет нелинейные преобразования с gating.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_size = context_size

        # Fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size)

        if context_size is not None:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)

        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Gating layer
        self.gate = nn.Linear(hidden_size, output_size)

        # Output projection
        if input_size != output_size:
            self.skip = nn.Linear(input_size, output_size)
        else:
            self.skip = None

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor
            context: Optional context tensor

        Returns:
            Output tensor
        """
        # Initial projection
        hidden = self.fc1(x)

        # Add context if provided
        if context is not None and self.context_size is not None:
            hidden = hidden + self.context_fc(context)

        # Non-linear transformation
        hidden = nn.ELU()(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)

        # Gating
        gate = torch.sigmoid(self.gate(hidden))

        # Apply gating
        output = gate * hidden

        # Skip connection
        if self.skip is not None:
            skip = self.skip(x)
        else:
            skip = x

        # Add skip and normalize
        output = self.layer_norm(output + skip)

        return output


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network.

    Выбирает наиболее релевантные переменные для каждого timestep.
    """

    def __init__(
        self,
        input_size: int,
        num_inputs: int,
        hidden_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size

        # GRN для каждой переменной
        self.variable_grns = nn.ModuleList(
            [GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout) for _ in range(num_inputs)]
        )

        # GRN для весов переменных
        self.weight_grn = GatedResidualNetwork(
            input_size * num_inputs,
            hidden_size,
            num_inputs,
            dropout,
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (..., num_inputs, input_size)

        Returns:
            Selected variables, shape (..., hidden_size)
        """
        # Flatten inputs для весов
        flatten = x.flatten(start_dim=-2)  # (..., num_inputs * input_size)

        # Variable weights
        weights = self.weight_grn(flatten)
        weights = self.softmax(weights)  # (..., num_inputs)

        # Process each variable
        processed_vars = []
        for i, grn in enumerate(self.variable_grns):
            var = x[..., i, :]  # (..., input_size)
            processed = grn(var)
            processed_vars.append(processed)

        # Stack и взвешиваем
        processed_vars = torch.stack(processed_vars, dim=-2)  # (..., num_inputs, hidden_size)
        weights = weights.unsqueeze(-1)  # (..., num_inputs, 1)

        # Weighted sum
        output = torch.sum(processed_vars * weights, dim=-2)  # (..., hidden_size)

        return output


class SimplifiedTFTModel(BaseSequentialModel):
    """
    Упрощённая версия Temporal Fusion Transformer.

    Включает основные компоненты TFT:
    - Variable selection
    - LSTM для temporal processing
    - Multi-head attention
    - Gated residual networks

    Параметры:
        input_size: Размерность входных признаков
        hidden_size: Размер скрытых слоёв
        num_layers: Количество LSTM слоёв
        seq_length: Длина последовательности
        output_size: Размер выхода
        num_heads: Количество attention heads
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
        num_heads: int = 4,
        dropout: float = 0.1,
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

        # Input embedding
        self.input_embedding = nn.Linear(input_size, hidden_size)

        # Variable selection (упрощённая версия)
        # В полном TFT это более сложный механизм
        self.variable_selection = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )

        # LSTM для temporal processing
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Gating для LSTM output
        self.lstm_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )

        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_size)

        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Gating для attention output
        self.attention_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )

        # Layer normalization
        self.ln2 = nn.LayerNorm(hidden_size)

        # Position-wise feed-forward (GRN)
        self.feed_forward = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size * 4,
            output_size=hidden_size,
            dropout=dropout,
        )

        # Output layers
        self.output_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )

        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch_size, seq_length, input_size)

        Returns:
            Output tensor, shape (batch_size, output_size)
        """
        batch_size, seq_len, _ = x.shape

        # Input embedding
        embedded = self.input_embedding(x)  # (batch, seq, hidden)

        # Variable selection
        selected = self.variable_selection(embedded)

        # LSTM processing
        lstm_out, _ = self.lstm(selected)

        # Gated LSTM output
        gate = self.lstm_gate(lstm_out)
        lstm_gated = lstm_out * gate

        # Residual + layer norm
        lstm_gated = self.ln1(lstm_gated + selected)

        # Self-attention
        attn_out, _ = self.self_attention(
            query=lstm_gated,
            key=lstm_gated,
            value=lstm_gated,
        )

        # Gated attention output
        attn_gate = self.attention_gate(attn_out)
        attn_gated = attn_out * attn_gate

        # Residual + layer norm
        attn_gated = self.ln2(attn_gated + lstm_gated)

        # Position-wise feed-forward
        ff_out = self.feed_forward(attn_gated)

        # Temporal aggregation (используем последний timestep)
        last_output = ff_out[:, -1, :]  # (batch, hidden)

        # Output processing
        output_features = self.output_grn(last_output)
        output = self.output_layer(output_features)

        return output

    def get_model_info(self) -> dict:
        """Получить информацию о модели."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_type": "SimplifiedTFT",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "seq_length": self.seq_length,
            "output_size": self.output_size,
            "dropout": self.dropout_rate,
            "task": self.task,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "device": str(self.device),
        }
