from typing import Literal, Callable

import numpy as np
from chromadb import Collection

from ..base import Context, ScoringModule
from .weighting import apply_weights


class KNNScorer(ScoringModule):
    def __init__(self, k: int, weights: Literal["uniform", "distance", "closest"] | bool):
        """
        Arguments
        ---
        - `k`: int, number of closest neighbors to consider during inference;
        - `weights`: bool or str from "uniform", "distance", "closest"
            - uniform (equivalent to False): a unit weight for each sample
            - distance (equivalent to True): weight is calculated as 1 / (distance_to_neighbor + 1e-5),
            - closest: each sample has a non zero weight iff is the closest sample of some class
        - `device`: str, something like "cuda:0" or "cuda:0,1,2", a device to store embedding function
        """
        self.k = k
        if isinstance(weights, bool):
            weights = "distance" if weights else "uniform"
        self.weights = weights

    def fit(self, context: Context):
        self._multilabel = context.multilabel
        self._collection = context.get_best_collection()
        self._n_classes = context.n_classes
        self._converter = context.vector_index.metadata_as_labels

    def predict(self, utterances: list[str]):
        labels, distances = query(self._collection, self.k, utterances, self._converter)
        probs = apply_weights(labels, distances, self.weights, self._n_classes, self._multilabel)
        return probs

    def clear_cache(self):
        model = self._collection._embedding_function._model
        model.to(device="cpu")
        del model
        self._collection = None


def query(
    collection: Collection,
    k: int,
    utterances: list[str],
    converter: Callable
):
    """
    Return
    ---

    `labels`:
    - multiclass case: np.ndarray of shape (n_samples, n_neighbors) with integer labels from [0,n_classes-1]
    - multilabel case: np.ndarray of shape (n_samples, n_neighbors, n_classes) with binary labels

    `distances`: np.ndarray of shape (n_samples, n_neighbors) with integer labels from 0..n_classes-1
    """
    query_res = collection.query(
        query_texts=utterances,
        n_results=k,
        include=["metadatas", "documents", "distances"],  # one can add "embeddings"
    )

    res_labels = np.array([converter(candidates) for candidates in query_res["metadatas"]])
    res_distances = np.array(query_res["distances"])

    return res_labels, res_distances
