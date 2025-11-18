"""
Seq2Seq модель с Attention mechanism.

Encoder-Decoder архитектура с attention для обработки
последовательностей и предсказания будущих значений.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modeling.models.neural.sequential.base import BaseSequentialModel


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention mechanism.

    Вычисляет attention веса между encoder outputs и decoder hidden state.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Вычислить attention context.

        Args:
            encoder_outputs: shape (batch, seq_len, hidden_size)
            decoder_hidden: shape (batch, hidden_size)

        Returns:
            context: shape (batch, hidden_size)
            attention_weights: shape (batch, seq_len)
        """
        # Expand decoder hidden для broadcasting
        decoder_hidden = decoder_hidden.unsqueeze(1)  # (batch, 1, hidden)

        # Вычисляем attention scores
        # score = V * tanh(W1 * encoder_output + W2 * decoder_hidden)
        energy = torch.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden))  # (batch, seq_len, hidden)

        attention_scores = self.V(energy).squeeze(-1)  # (batch, seq_len)

        # Softmax для получения весов
        attention_weights = F.softmax(attention_scores, dim=1)

        # Взвешенная сумма encoder outputs
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            encoder_outputs,  # (batch, seq_len, hidden)
        ).squeeze(
            1
        )  # (batch, hidden)

        return context, attention_weights


class LuongAttention(nn.Module):
    """
    Luong (Multiplicative) Attention mechanism.

    Более простой вариант attention с матричным умножением.
    """

    def __init__(self, hidden_size: int, method: str = "dot"):
        super().__init__()
        self.hidden_size = hidden_size
        self.method = method

        if method == "general":
            self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == "concat":
            self.W = nn.Linear(hidden_size * 2, hidden_size)
            self.V = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Вычислить attention context.

        Args:
            encoder_outputs: shape (batch, seq_len, hidden_size)
            decoder_hidden: shape (batch, hidden_size)

        Returns:
            context: shape (batch, hidden_size)
            attention_weights: shape (batch, seq_len)
        """
        if self.method == "dot":
            # scores = decoder_hidden · encoder_outputs^T
            attention_scores = torch.bmm(
                decoder_hidden.unsqueeze(1),  # (batch, 1, hidden)
                encoder_outputs.transpose(1, 2),  # (batch, hidden, seq_len)
            ).squeeze(
                1
            )  # (batch, seq_len)

        elif self.method == "general":
            # scores = decoder_hidden · W · encoder_outputs^T
            attention_scores = torch.bmm(
                self.W(decoder_hidden).unsqueeze(1),
                encoder_outputs.transpose(1, 2),
            ).squeeze(1)

        elif self.method == "concat":
            # scores = V · tanh(W · [decoder_hidden; encoder_outputs])
            seq_len = encoder_outputs.size(1)
            decoder_hidden_expanded = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
            concat = torch.cat([decoder_hidden_expanded, encoder_outputs], dim=2)
            energy = torch.tanh(self.W(concat))
            attention_scores = self.V(energy).squeeze(-1)

        # Softmax
        attention_weights = F.softmax(attention_scores, dim=1)

        # Context
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs,
        ).squeeze(1)

        return context, attention_weights


class Seq2SeqAttentionModel(BaseSequentialModel):
    """
    Seq2Seq модель с Attention для временных рядов.

    Encoder кодирует входную последовательность,
    Decoder с attention генерирует предсказания.

    Параметры:
        input_size: Размерность входных признаков
        hidden_size: Размер скрытого слоя
        num_layers: Количество слоёв в encoder и decoder
        seq_length: Длина входной последовательности
        output_size: Размер выхода
        dropout: Dropout rate
        attention_type: 'bahdanau' или 'luong'
        cell_type: 'lstm' или 'gru'
        task: 'classification' или 'regression'
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        seq_length: int,
        output_size: int = 1,
        dropout: float = 0.2,
        attention_type: str = "bahdanau",
        cell_type: str = "lstm",
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

        self.attention_type = attention_type
        self.cell_type = cell_type.lower()

        # Encoder
        if self.cell_type == "lstm":
            self.encoder = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif self.cell_type == "gru":
            self.encoder = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            raise ValueError(f"Неизвестный cell_type: {cell_type}")

        # Attention mechanism
        if attention_type == "bahdanau":
            self.attention = BahdanauAttention(hidden_size)
        elif attention_type == "luong":
            self.attention = LuongAttention(hidden_size, method="general")
        else:
            raise ValueError(f"Неизвестный attention_type: {attention_type}")

        # Decoder (single step)
        if self.cell_type == "lstm":
            self.decoder_cell = nn.LSTMCell(
                input_size=hidden_size,  # Используем context как вход
                hidden_size=hidden_size,
            )
        else:
            self.decoder_cell = nn.GRUCell(
                input_size=hidden_size,
                hidden_size=hidden_size,
            )

        # Output projection
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),  # context + hidden
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

        self.attention_weights_history: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch_size, seq_length, input_size)

        Returns:
            Output tensor, shape (batch_size, output_size)
        """
        # Encode
        if self.cell_type == "lstm":
            encoder_outputs, (h_n, c_n) = self.encoder(x)
            # Используем последний слой
            decoder_hidden = h_n[-1]  # (batch, hidden)
            decoder_cell = c_n[-1]
        else:
            encoder_outputs, h_n = self.encoder(x)
            decoder_hidden = h_n[-1]
            decoder_cell = None

        # Attention
        context, attention_weights = self.attention(encoder_outputs, decoder_hidden)

        # Сохраняем attention веса
        self.attention_weights_history = attention_weights.detach()

        # Decode (single step для классификации/регрессии)
        if self.cell_type == "lstm":
            decoder_hidden, decoder_cell = self.decoder_cell(context, (decoder_hidden, decoder_cell))
        else:
            decoder_hidden = self.decoder_cell(context, decoder_hidden)

        # Комбинируем context и decoder hidden
        combined = torch.cat([context, decoder_hidden], dim=1)

        # Output
        output = self.fc(combined)

        return output

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Получить attention веса последнего forward pass."""
        return self.attention_weights_history

    def get_model_info(self) -> dict:
        """Получить информацию о модели."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_type": "Seq2SeqAttention",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "seq_length": self.seq_length,
            "output_size": self.output_size,
            "dropout": self.dropout_rate,
            "attention_type": self.attention_type,
            "cell_type": self.cell_type,
            "task": self.task,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "device": str(self.device),
        }


class MultiStepSeq2SeqModel(BaseSequentialModel):
    """
    Seq2Seq для многошагового прогнозирования.

    Decoder генерирует последовательность предсказаний.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        seq_length: int,
        output_size: int = 1,
        output_steps: int = 1,  # Количество шагов для предсказания
        dropout: float = 0.2,
        task: str = "regression",  # Обычно регрессия для многошагового
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

        self.output_steps = output_steps

        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Decoder
        self.decoder = nn.LSTM(
            input_size=output_size,  # Вход - предыдущее предсказание
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention
        self.attention = BahdanauAttention(hidden_size)

        # Output projection
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass для многошагового предсказания.

        Args:
            x: Input tensor, shape (batch_size, seq_length, input_size)

        Returns:
            Output tensor, shape (batch_size, output_steps, output_size)
        """
        batch_size = x.size(0)

        # Encode
        encoder_outputs, (h_n, c_n) = self.encoder(x)

        # Decoder loop
        decoder_hidden = h_n
        decoder_cell = c_n

        outputs = []
        decoder_input = torch.zeros(batch_size, 1, self.output_size).to(x.device)

        for _ in range(self.output_steps):
            # Decoder step
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder(
                decoder_input,
                (decoder_hidden, decoder_cell),
            )

            # Attention
            context, _ = self.attention(encoder_outputs, decoder_hidden[-1])

            # Combine and predict
            combined = torch.cat([context, decoder_hidden[-1]], dim=1)
            output = self.fc(combined)

            outputs.append(output.unsqueeze(1))

            # Следующий вход - текущее предсказание
            decoder_input = output.unsqueeze(1)

        # Concatenate all steps
        outputs = torch.cat(outputs, dim=1)  # (batch, output_steps, output_size)

        # Для совместимости с BaseSequentialModel возвращаем (batch, output_size)
        # Берём последний шаг или среднее
        return outputs[:, -1, :]
