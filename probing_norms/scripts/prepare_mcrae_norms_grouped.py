import pdb
import numpy as np
import pandas as pd

from probing_norms.constants import MCRAE_SEP, MCRAE_PREFIXES
from probing_norms.data import load_mcrae_feature_norms
from probing_norms.utils import multimap, cache_json


def load_mapping_features():
    """Maps old features names to new features names based on
    Stella's annotations and comments.

    """
    # https://lampers.slack.com/archives/C07S0K5JDCL/p1740065597270899?thread_ts=1740045355.565199&cid=C07S0K5JDCL
    mapping1 = {
        "an_instrument": "a_scientific_instrument",
        "used_for_holding": "used_for_holding_things_together",
        "has_2_prongs": "has_prongs",
        "has_3_prongs": "has_prongs",
        "has_4_prongs": "has_prongs",
    }

    # https://docs.google.com/spreadsheets/d/1xDXnQIwfngg4QGdUoYSQq5qSi95ImP1xtgz01X0UsKQ/edit?gid=1000224099#gid=1000224099
    df = pd.read_csv("data/mcrae-norms-grouped-input.csv")
    idxs = df["merge to"].notnull()
    df = df[idxs]
    mapping2 = dict(df[["norm", "merge to"]].values)
    mapping3 = dict(df[["norm.1", "merge to"]].values)

    mapping = {**mapping1, **mapping2, **mapping3}
    mapping = {k: v for k, v in mapping.items() if k != v}

    # for k, v in mapping.items():
    #     print("{:12s} → {}".format(k, v))

    return mapping


def load_feature_new_to_features_mcrae():
    """New feature name to list of mapped McRae features."""
    concept_feature = load_mcrae_feature_norms()
    features = sorted(set(f for _, f in concept_feature))
    mapping = load_mapping_features()
    feature1_to_features = [(mapping.get(f, f), f) for f in features]
    return multimap(feature1_to_features)


def load_mcrae_feature_norms_grouped():
    mapping = load_mapping_features()
    concept_feature = load_mcrae_feature_norms()
    concept_feature_new = [(c, mapping.get(f, f)) for c, f in concept_feature]
    return sorted(set(concept_feature_new))


def normalize_norm(text):
    if MCRAE_SEP in text:
        prefix, text = text.split(MCRAE_SEP)
        assert prefix in MCRAE_PREFIXES, prefix
        if prefix == "beh":
            text = "living being that " + text
        elif prefix == "inbeh":
            text = "inanimate object that " + text
        elif prefix == "eg":
            text = "is an example of " + text
        elif prefix in {"has_units", "worn_by_men"}:
            text = text + " " + prefix
        else:
            pass

    if text.startswith("a_") or text.startswith("an_"):
        text = "is_" + text

    text = text.replace("_", " ").strip()
    return text


def prepare_mcrae_feature_list(num_min_concepts=5):
    concept_feature = load_mcrae_feature_norms_grouped()
    feature_concept = [(f, c) for c, f in concept_feature]
    feature_to_concepts = multimap(feature_concept)
    feature_to_concepts = {
        f: cs for f, cs in feature_to_concepts.items() if len(cs) >= num_min_concepts
    }

    mcrae_not_in_things = pd.read_csv("data/mcrae-not-in-things.csv")
    mcrae_not_in_things = dict(mcrae_not_in_things[["McRae Concept", "THINGS mapped concept"]].values)

    def map_to_things(c):
        if c in mcrae_not_in_things:
            return mcrae_not_in_things[c]
        else:
            return c

    def prepare_datum(f):
        concepts1 = feature_to_concepts[f]
        concepts2 = [map_to_things(c) for c in concepts1]
        concepts2 = [c for c in concepts2 if not pd.isna(c)]
        concepts2 = sorted(set(concepts2))
        return {
            "norm": f,
            "norm (language)": normalize_norm(f),
            "concepts-mcrae": concepts1,
            "concepts-mcrae-in-things ": concepts2,
        }

    data = [prepare_datum(f) for f in sorted(feature_to_concepts.keys())]
    # df = pd.DataFrame(data)
    # df.to_csv("data/mcrae-norms-grouped-2.csv", index=False)

    return data


if __name__ == "__main__":
    cache_json("data/mcrae-norms-grouped-with-concepts.json", prepare_mcrae_feature_list, 5)