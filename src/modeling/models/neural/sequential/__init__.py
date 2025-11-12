"""
Нейросетевые модели для временных рядов.

Этот модуль содержит реализации sequential моделей:
- LSTM, GRU - рекуррентные сети
- Seq2Seq with Attention
- TCN - Temporal Convolutional Network
- TFT - Temporal Fusion Transformer
- CNN+LSTM - гибридная архитектура
"""

# Utilities
from src.modeling.models.neural.sequential.augmentation import (
    ComposedAugmentation,
    Jitter,
    MagnitudeWarp,
    RandomCrop,
    Scaling,
    SequenceAugmentor,
    TimeWarp,
    WindowSlicing,
    get_default_augmentation,
)

# Base classes
from src.modeling.models.neural.sequential.base import BaseSequentialModel
from src.modeling.models.neural.sequential.cnn_lstm import (
    CNNGRUModel,
    CNNLSTMModel,
    MultiScaleCNNLSTMModel,
    ResidualCNNLSTMModel,
)
from src.modeling.models.neural.sequential.datasets import (
    MultiHorizonSequenceDataset,
    SequenceDataset,
    create_sequence_dataloader,
)
from src.modeling.models.neural.sequential.gru import (
    AttentionGRUModel,
    GRUModel,
    MultiHeadAttentionGRUModel,
)

# Models
from src.modeling.models.neural.sequential.lstm import (
    AttentionLSTMModel,
    LSTMModel,
    StackedLSTMModel,
)
from src.modeling.models.neural.sequential.seq2seq_attention import (
    MultiStepSeq2SeqModel,
    Seq2SeqAttentionModel,
)
from src.modeling.models.neural.sequential.tcn import (
    ResidualTCNModel,
    TCNModel,
)
from src.modeling.models.neural.sequential.tft import (
    SimplifiedTFTModel,
)

__all__ = [
    # Base
    "BaseSequentialModel",
    "SequenceDataset",
    "MultiHorizonSequenceDataset",
    "create_sequence_dataloader",
    # LSTM
    "LSTMModel",
    "StackedLSTMModel",
    "AttentionLSTMModel",
    # GRU
    "GRUModel",
    "AttentionGRUModel",
    "MultiHeadAttentionGRUModel",
    # Seq2Seq
    "Seq2SeqAttentionModel",
    "MultiStepSeq2SeqModel",
    # TCN
    "TCNModel",
    "ResidualTCNModel",
    # CNN+LSTM
    "CNNLSTMModel",
    "CNNGRUModel",
    "ResidualCNNLSTMModel",
    "MultiScaleCNNLSTMModel",
    # TFT
    "SimplifiedTFTModel",
    # Augmentation
    "SequenceAugmentor",
    "Jitter",
    "Scaling",
    "MagnitudeWarp",
    "TimeWarp",
    "WindowSlicing",
    "RandomCrop",
    "ComposedAugmentation",
    "get_default_augmentation",
]
