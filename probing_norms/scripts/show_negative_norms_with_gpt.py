import pdb
import os

from itertools import groupby

import pandas as pd

from probing_norms.utils import read_json
from probing_norms.predict import McRaeMappedNormsLoader
from probing_norms.scripts.get_negative_norms_with_gpt import concept_mapping, get_path

question_types = ["pos", "neg", "maybe"]

norms_loader = McRaeMappedNormsLoader()
features_to_concepts, feature_to_id, features = norms_loader()

concept_mapping_rev = {v: k for k, v in concept_mapping.items()}


def get_key_with_default(dict, key):
    try:
        return dict[key]
    except KeyError:
        print(f"WARN Key not found: {key}")
        return key + "†"


def get_key(item):
    return item["feature"]


def prepare_response(response):
    result = response.split(", ")
    result = [get_key_with_default(concept_mapping_rev, r) for r in result]
    result = set(result)
    return sorted(result)


def grouper(key_group):
    key, group = key_group
    extra = {
        entry["question-type"]: prepare_response(entry["response"]) for entry in group
    }
    return {
        "feature": key,
        "pos-orig": sorted(set(features_to_concepts[key])),
        **extra,
    }


def add_counts(entry):
    to_count = ["pos-orig", "pos", "neg", "maybe"]
    counts = {"num. " + t: len(entry[t]) for t in to_count}
    counts_inter = {
        "num. " + t1 + " ∩ " + t2: len(set(entry[t1]) & set(entry[t2]))
        for t1, t2 in [
            ("pos", "pos-orig"),
            ("pos", "maybe"),
            ("pos", "neg"),
            ("maybe", "neg"),
        ]
    }
    union1 = set(entry["pos"]) | set(entry["maybe"]) | set(entry["neg"])
    union2 = {elem for elem in union1 if "†" not in elem}
    return {
        **entry,
        **counts,
        **counts_inter,
        "num. pos ∪ maybe ∪ neg": len(union1),
        "num. pos ∪ maybe ∪ neg (without †)": len(union2),
        "pos ∩ neg": set(entry["pos"]) & set(entry["neg"]),
    }


results = [
    read_json(get_path(feature_id, question_type))
    for feature_id in feature_to_id.values()
    for question_type in question_types
    if os.path.exists(get_path(feature_id, question_type))
]

results = sorted(results, key=get_key)
results = map(grouper, groupby(results, key=get_key))
results = map(add_counts, results)
results = list(results)

cols_list = ["pos-orig", "pos", "maybe", "neg", "pos ∩ neg"]
df = pd.DataFrame(results)
for col in cols_list:
    df[col] = df[col].apply(lambda x: ", ".join(x))
df.to_csv("output/negative_norms_with_gpt.csv", index=False, sep="\t")
