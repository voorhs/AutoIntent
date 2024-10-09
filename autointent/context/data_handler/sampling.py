import contextlib
import random

import sre_yield

from .schemas import Dataset, Utterance


def distribute_shots(n, k):
    """randomly distribute `k` samples among `n` bins"""
    samples_per_bin = [0] * n
    for _ in range(k):
        i_bin = random.randint(0, n - 1)
        samples_per_bin[i_bin] += 1
    return samples_per_bin


def generate_from_template(template, n):
    """generate `n` samples from `template`, or fewer if its impossible"""
    iterator = iter(sre_yield.AllStrings(template))
    with contextlib.suppress(StopIteration):
        res = [next(iterator) for _ in range(n)]
    return res


def generate_from_templates(patterns: list[str], n_shots: int):
    shots_per_pattern = distribute_shots(len(patterns), n_shots)
    res = []
    for pattern, n in zip(patterns, shots_per_pattern, strict=False):
        new_samples = generate_from_template(pattern, n)
        res.extend(new_samples)
    return res


def sample_from_regex(
    dataset: Dataset, n_shots: int, random_seed: int = 0,
) -> Dataset:
    random.seed(random_seed)

    for intent in dataset.intents:
        generated_texts = generate_from_templates(intent.regexp_full_match, n_shots)
        for generated_text in generated_texts:
            utterance = Utterance(text=generated_text, label=intent.id)
            dataset.utterances.append(utterance)

    return dataset
