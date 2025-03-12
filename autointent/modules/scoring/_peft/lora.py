from typing import Any

from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
)

from .base import BasePEFTScorer


class LoRAScorer(BasePEFTScorer):
    """Scorer using LoRA fine-tuning for text classification.

    Args:
        target_modules: list of module names to apply LoRA to
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
    """

    name = "lora"

    def __init__(
        self,
        target_modules: list[str],
        r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.2,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        self.target_modules = target_modules
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

    def _create_model(self, num_labels: int) -> AutoModelForSequenceClassification:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_config.model_name, num_labels=num_labels
        )
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
        )
        return get_peft_model(base_model, config)
