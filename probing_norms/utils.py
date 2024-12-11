from collections import defaultdict
from typing import Callable, List, TypeVar, Tuple, Iterable

import json
import os
import pickle

import numpy as np
import pandas as pd


A = TypeVar("A")


def read_file(
    path: str, parse_line: Callable[[str], A] = lambda line: line.strip()
) -> List[A]:
    with open(path, "r") as f:
        return list(map(parse_line, f.readlines()))


def reverse_dict(d: dict) -> dict:
    return {v: k for k, v in d.items()}


def implies(p: bool, q: bool):
    return not p or q


def logit(probas: np.ndarray):
    return np.log(probas / (1 - probas))


def sigmoid(logit: np.ndarray):
    return 1 / (1 + np.exp(-logit))


def cache_np(path, func, *args, **kwargs):
    if os.path.exists(path):
        return np.load(path)
    else:
        result = func(*args, **kwargs)
        np.save(path, result)
        return result


def cache_json(path, func, *args, **kwargs):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        result = func(*args, **kwargs)
        with open(path, "w") as f:
            json.dump(result, f, indent=4)
        return result


def cache_df(path, func, *args, **kwargs):
    try:
        return pd.read_pickle(path)
    except FileNotFoundError:
        result = func(*args, **kwargs)
        result.to_pickle(path)
        return result


def cache_pickle(path, func, *args, **kwargs):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        result = func(*args, **kwargs)
        with open(path, "wb") as f:
            pickle.dump(result, f)
        return result


class multimap(defaultdict):
    """A mapping of {key: [val1, val2, ...]}."""

    def __init__(self, pairs: Iterable[tuple] = (), symmetric=False):
        """Given (key, val) pairs, return {key: [val, ...], ...}.
        If `symmetric` is True, treat (key, val) as (key, val) plus (val, key)."""
        self.default_factory = list
        for key, val in pairs:
            self[key].append(val)
            if symmetric:
                self[val].append(key)
