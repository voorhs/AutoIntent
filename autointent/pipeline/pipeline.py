import importlib.resources as ires
import json
import os
import pickle
import inspect
from ..modules.scoring.linear import LinearScorer
import numpy as np
import yaml

from .. import Context
from ..nodes import (
    Node,
    PredictionNode,
    RegExpNode,
    RetrievalNode,
    ScoringNode,
)
from .utils import NumpyEncoder


class Pipeline:
    available_nodes = {
        "regexp": RegExpNode,
        "retrieval": RetrievalNode,
        "scoring": ScoringNode,
        "prediction": PredictionNode,
    }

    def serialize(self):
        return pickle.dumps(self)

    @classmethod
    def load(cls, serialized_pipeline):
        return pickle.loads(serialized_pipeline)

    def predict(self, texts):
        results = []
        for text in texts:
            current_input = text
            for node_config in self.config["nodes"]:
                node_type = node_config["node_type"]
                best_module = self.best_modules[node_type]

                if node_type == 'scoring':
                    # Для модуля scoring передаем текст в виде списка кортежей
                    current_input = best_module.predict([(current_input, '')])
                elif node_type == 'prediction':
                    # Для модуля prediction преобразуем вход в двумерный numpy массив
                    current_input = np.atleast_2d(current_input)
                    current_input = best_module.predict(current_input)
                else:
                    # Для других модулей оставляем как есть
                    current_input = best_module.predict([current_input])

                # Если результат - список или numpy массив, берем первый элемент
                if isinstance(current_input, (list, np.ndarray)) and len(current_input) == 1:
                    current_input = current_input[0]

            results.append(current_input)
        return results

    def load_best_modules(self, saved_modules, context):
        self.best_modules = {}
        for node_type, module_info in saved_modules.items():
            node_class = self.available_nodes[node_type]
            module_class = node_class.modules_available[module_info['module_type']]

            module_init_params = inspect.signature(module_class.__init__).parameters
            filtered_params = {k: v for k, v in module_info['parameters'].items()
                               if k in module_init_params}

            module = module_class(**filtered_params)

            # Вызываем fit для инициализации модуля
            if hasattr(module, 'fit'):
                module.fit(context)

            self.best_modules[node_type] = module

    def save_best_modules(self):
        saved_modules = {}
        for node_type, module in self.best_modules.items():
            # Получаем список параметров конструктора модуля
            module_init_params = inspect.signature(type(module).__init__).parameters

            # Сохраняем только те атрибуты, которые соответствуют параметрам конструктора
            params = {k: v for k, v in module.__dict__.items()
                      if k in module_init_params and not k.startswith('_')}

            saved_modules[node_type] = {
                'module_type': type(module).__name__,
                'parameters': params
            }
        return saved_modules

    def get_best_module_config(self, node_type):
        # Получаем конфигурацию лучшего модуля для данного типа узла
        node_metrics = self.context.optimization_logs.cache["metrics"][node_type]
        best_index = np.argmax(node_metrics)
        return self.context.optimization_logs.cache["configs"][node_type][best_index]

    def __init__(self, config_path: os.PathLike, mode: str, verbose: bool):
        self.config = load_config(config_path, mode)
        self.verbose = verbose
        self.best_modules = {}  # Инициализируем best_modules как пустой словарь
        self.context = None

    def optimize(self, context):
        self.context = context
        for node_config in self.config["nodes"]:
            node_type = node_config["node_type"]
            node: Node = self.available_nodes[node_type](
                modules_search_spaces=node_config["modules"],
                metric=node_config["metric"],
                verbose=self.verbose
            )
            node.fit(context)
            print(f"Fitted {node_type}!")

            best_config = self.get_best_module_config(node_type)
            module_class = node.modules_available[best_config['module_type']]

            module_init_params = inspect.signature(module_class.__init__).parameters
            module_params = {k: v for k, v in best_config.items()
                             if k in module_init_params and k != 'module_type'}

            module = module_class(**module_params)

            # Вызываем fit для инициализации модуля
            if hasattr(module, 'fit'):
                module.fit(context)

            self.best_modules[node_type] = module

    def dump(self, logs_dir: os.PathLike, run_name: str):
        optimization_results = self.context.optimization_logs.dump()

        # create appropriate directory
        if logs_dir == "":
            logs_dir = os.getcwd()
        logs_dir = os.path.join(logs_dir, run_name)
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        # dump config and optimization results
        logs_path = os.path.join(logs_dir, "logs.json")
        json.dump(optimization_results, open(logs_path, "w"), indent=4, ensure_ascii=False, cls=NumpyEncoder)
        config_path = os.path.join(logs_dir, "config.yaml")
        yaml.dump(self.config, open(config_path, "w"))

        if self.verbose:
            print(
                make_report(
                    optimization_results, nodes=[node_config["node_type"] for node_config in self.config["nodes"]]
                )
            )

        # dump train and test data splits
        train_data, test_data = self.context.data_handler.dump()
        train_path = os.path.join(logs_dir, "train_data.json")
        test_path = os.path.join(logs_dir, "test_data.json")
        json.dump(train_data, open(train_path, "w"), indent=4, ensure_ascii=False)
        json.dump(test_data, open(test_path, "w"), indent=4, ensure_ascii=False)


def load_config(config_path: os.PathLike, mode: str):
    """load config from the given path or load default config which is distributed along with the autointent package"""
    if config_path != "":
        file = open(config_path)
    else:
        config_name = "default-multilabel-config.yaml" if mode != "multiclass" else "default-multiclass-config.yaml"
        file = ires.files("autointent.datafiles").joinpath(config_name).open()
    return yaml.safe_load(file)


def make_report(logs: dict[str], nodes: list[str]) -> str:
    ids = [np.argmax(logs["metrics"][node]) for node in nodes]
    configs = []
    for i, node in zip(ids, nodes):
        cur_config = logs["configs"][node][i]
        cur_config["metric_value"] = logs["metrics"][node][i]
        configs.append(cur_config)
    messages = [json.dumps(c, indent=4) for c in configs]
    return "\n".join(messages)
