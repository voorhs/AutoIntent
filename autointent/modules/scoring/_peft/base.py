"""PEFT Scorers for text classification using transformer fine-tuning."""

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from autointent.configs import STModelConfig
from autointent.custom_types import ListOfLabels
from autointent.modules.base import BaseScorer


class BasePEFTScorer(BaseScorer):
    """Base class for PEFT-based transformer scorers."""

    supports_multiclass = True
    supports_multilabel = False

    def __init__(
        self,
        model_config: STModelConfig | str | dict[str, Any] | None = None,
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        num_epochs: int = 300,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        self.model_config = STModelConfig.from_search_config(model_config)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.seed = seed
        self.device = device

    def fit(self, utterances: list[str], labels: ListOfLabels) -> None:
        if hasattr(self, "_model"):
            self.clear_cache()

        self._validate_task(labels)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_name)

        # Simplified dataset handling
        inputs = self._tokenizer(utterances, padding=True, truncation=True, return_tensors="pt")
        dataset = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": torch.tensor(labels),
        }

        model = self._create_model(len(np.unique(labels)))
        model.to(self.device)

        training_args = TrainingArguments(
            output_dir="tmp_train",
            evaluation_strategy="no",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=0.01,
            seed=self.seed,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorWithPadding(tokenizer=self._tokenizer),
            callbacks=[EarlyStoppingCallback(5, 1e-2)],
        )
        trainer.train()
        self._model = trainer.model

    def predict(self, utterances: list[str]) -> npt.NDArray[Any]:
        if not self._model or not self._tokenizer:
            raise RuntimeError("Model not trained. Call fit() first.")

        self._model.eval()
        inputs = self._tokenizer(utterances, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
        return torch.softmax(outputs.logits, dim=-1).cpu().numpy()

    def clear_cache(self) -> None:
        if self._model:
            del self._model
        self._model = None
        torch.cuda.empty_cache()

    def _create_model(self, num_labels: int) -> AutoModelForSequenceClassification:
        raise NotImplementedError("Subclasses must implement this method")
