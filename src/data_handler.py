import itertools as it
import os
from pprint import pprint

from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sklearn.model_selection import train_test_split


class DataHandler:
    def __init__(
        self, intent_records: os.PathLike, db_path: os.PathLike = "../data/chroma"
    ):
        (
            self.n_classes,
            self.utterances_train,
            self.utterances_test,
            self.labels_train,
            self.labels_test,
        ) = split_sample_utterances(intent_records)

        self.client = PersistentClient(path=db_path)
        self.cache = dict(
            best_assets=dict(
                retrieval=None,  # str, name of best retriever
                scoring=None,  # np.ndarray of shape (n_samples, n_classes), from best scorer
                prediction=None,  # np.ndarray of shape (n_samples,), from best predictor
            ),
            metrics=dict(retrieval=[], scoring=[], prediction=[]),
            configs=dict(retrieval=[], scoring=[], prediction=[]),
        )

    def get_collection(self, model_name: str, device="cuda"):
        emb_func = SentenceTransformerEmbeddingFunction(
            model_name=model_name,
            trust_remote_code=True,
            device=device
        )
        db_name = model_name.replace("/", "_")
        collection = self.client.get_or_create_collection(
            name=db_name,
            embedding_function=emb_func,
            metadata={"n_classes": self.n_classes},
        )
        return collection

    def create_collection(self, model_name: str, device="cuda"):
        collection = self.get_collection(model_name, device)
        db_name = model_name.replace("/", "_")
        collection.add(
            documents=self.utterances_train,
            ids=[f"{i}-{db_name}" for i in range(len(self.utterances_train))],
            metadatas=[{"intent_id": lab} for lab in self.labels_train],
        )
        return collection

    def delete_collection(self, model_name: str):
        db_name = model_name.replace("/", "_")
        self.client.delete_collection(db_name)

    def log_module_optimization(
            self,
            node_type: str,
            module_type: str,
            module_config: dict,
            metric_value: float,
            metric_name: str,
            assets,
            verbose=False,
        ):
        """
        Purposes:
        - save optimization results in a text form (hyperparameters and corresponding metrics)
        - update best assets
        """

        # "update leaderboard" if it's a new best metric
        metrics_list = self.cache["metrics"][node_type]
        previous_best = max(metrics_list, default=-float("inf"))
        if metric_value > previous_best:
            self.cache["best_assets"][node_type] = assets

        # logging
        logs = dict(
            module_type=module_type,
            metric_name=metric_name,
            metric_value=metric_value,
            **module_config,
        )
        self.cache["configs"][node_type].append(logs)
        if verbose:
            pprint(logs)
        metrics_list.append(metric_value)

    def get_best_collection(self, device="cuda"):
        model_name = self.cache["best_assets"]["retrieval"]
        return self.get_collection(model_name, device)

    def get_best_scores(self):
        return self.cache["best_assets"]["scoring"]

    def dump_logs(self):
        res = dict(
            metrics=self.cache["metrics"],
            configs=self.cache["configs"],
        )
        return res


def get_sample_utterances(intent_records: list[dict]):
    """get plain list of all sample utterances and their intent labels"""
    utterances = [intent["sample_utterances"] for intent in intent_records]
    labels = [
        [intent["intent_id"]] * len(uts) for intent, uts in zip(intent_records, utterances)
    ]

    utterances = list(it.chain.from_iterable(utterances))
    labels = list(it.chain.from_iterable(labels))

    return utterances, labels


def split_sample_utterances(intent_records: list[dict]):
    """
    Return: utterances_train, utterances_test, labels_train, labels_test

    TODO: ensure stratified train test splitting (test set must contain all classes)
    """

    utterances, labels = get_sample_utterances(intent_records)
    n_classes = len(set(labels))
    splits = train_test_split(
        utterances,
        labels,
        test_size=0.25,
        random_state=0,
        stratify=labels,
        shuffle=True,
    )
    res = [n_classes] + splits
    return res
