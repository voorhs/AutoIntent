from typing import Any

from peft import get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
)

from .base import BasePEFTScorer


class PTuningScorer(BasePEFTScorer):
    """Scorer using P-Tuning for text classification.

    Args:
        num_virtual_tokens: Number of virtual tokens
        encoder_reparameterization_type: Encoder type (MLP/LSTM)
        encoder_hidden_size: Encoder hidden size
    """

    name = "p_tuning"

    def __init__(
        self,
        num_virtual_tokens: int = 20,
        encoder_reparameterization_type: str = "MLP",
        encoder_hidden_size: int = 128,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        self.num_virtual_tokens = num_virtual_tokens
        self.encoder_reparameterization_type = encoder_reparameterization_type
        self.encoder_hidden_size = encoder_hidden_size

    def _create_model(self, num_labels: int) -> AutoModelForSequenceClassification:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_config.model_name, num_labels=num_labels
        )
        config = {
            "num_virtual_tokens": self.num_virtual_tokens,
            "encoder_reparameterization_type": self.encoder_reparameterization_type,
            "encoder_hidden_size": self.encoder_hidden_size,
        }
        return get_peft_model(base_model, config)
