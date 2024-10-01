from typing import Any

from .data_handler import DataHandler
from .optimization_info import OptimizationInfo
from .vector_index import VectorIndex


class Context:
    def __init__(
        self,
        multiclass_intent_records,
        multilabel_utterance_records,
        test_utterance_records,
        device,
        mode,
        multilabel_generation_config: str,
        db_dir,
        regex_sampling,
        seed,
    ) -> None:
        self.data_handler = DataHandler(
            multiclass_intent_records,
            multilabel_utterance_records,
            test_utterance_records,
            mode,
            multilabel_generation_config,
            regex_sampling,
            seed,
        )
        self.optimization_info = OptimizationInfo()
        self.vector_index = VectorIndex(db_dir, device, self.data_handler.multilabel, self.data_handler.n_classes)

        self.device = device
        self.multilabel = self.data_handler.multilabel
        self.n_classes = self.data_handler.n_classes
        self.seed = seed

    def get_best_collection(self):
        model_name = self.optimization_info.get_best_embedder()
        return self.vector_index.get_collection(model_name)

    def get_inference_config(self) -> dict[str, Any]:
        best_trials = self.optimization_info.get_best_modules()
        return {
            "metadata": {
                "device": self.device,
                "multilabel": self.multilabel,
                "n_classes": self.n_classes,
                "seed": self.seed,
                "db_dir": self.vector_index.db_dir,
            },
            "modules": [
                {"module_type": trial.module_type, "module_params": trial.module_params}
                for trial in best_trials
            ]
        }
