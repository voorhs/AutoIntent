from typing import Any

from peft import AdaLoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
)

from .base import BasePEFTScorer


class AdaLoRAScorer(BasePEFTScorer):
    """Scorer using AdaLoRA fine-tuning for text classification.

    Args:
        target_modules: List of module names to apply AdaLoRA to
        r: Initial AdaLoRA rank
        lora_alpha: AdaLoRA alpha parameter
        lora_dropout: AdaLoRA dropout rate
    """

    name = "adalora"

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
        config = AdaLoraConfig(
            task_type=TaskType.SEQ_CLS,
            init_r=self.r,
            target_modules=self.target_modules,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
        )
        return get_peft_model(base_model, config)
