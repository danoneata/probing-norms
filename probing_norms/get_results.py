import csv
import pdb

from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from probing_norms.utils import read_json
from probing_norms.predict import (
    NORMS_PRIMING,
    NORMS_NUM_RUNS,
    load_features_metadata,
    sample_features,
)

NORMS_MODEL = "chatgpt-gpt3.5-turbo"
DATASET_NAME = "things"

FEATURE_TO_CONCEPTS, FEATURE_TO_ID = load_features_metadata(
    priming=NORMS_PRIMING,
    model=NORMS_MODEL,
    num_runs=NORMS_NUM_RUNS,
)
FEATURES_SELECTED = sample_features(FEATURE_TO_CONCEPTS)

SCORE_FUNCS = {
    "leave-one-concept-out": accuracy_score,
    "repeated-k-fold": f1_score,
}

FEATURE_TYPES = [
    "pali-gemma-224",
    "siglip-224",
    "vit-mae-large",
    "dino-v2",
    "swin-v2",
    "max-vit-large",
    "random-siglip",
]


def load_result(embeddings_level, split_type, feature_type, feature):
    def evaluate(data):
        true = np.array([datum["true"] for datum in data])
        pred = np.array([datum["pred"] for datum in data])
        pred = (pred > 0.5).astype(int)
        return SCORE_FUNCS[split_type](true, pred)

    def get_path(feature):
        feature_norm = "{}_{}_{}".format(NORMS_PRIMING, NORMS_MODEL, NORMS_NUM_RUNS)
        feature_id = FEATURE_TO_ID[feature]
        return "output/linear-probe-predictions/{}/{}/{}-{}-{}-{}".format(
            embeddings_level,
            split_type,
            DATASET_NAME,
            feature_type,
            feature_norm,
            feature_id,
        )

    try:
        path = get_path(feature) + ".json"
        results = read_json(path)
        scores = [evaluate(result["preds"]) for result in results]
    except FileNotFoundError:
        scores = [np.nan]

    score = 100 * np.mean(scores)
    score = score.item()
    return {
        "feature": feature,
        "level": embeddings_level,
        "model": feature_type,
        "split": split_type,
        "score": score,
    }


def get_results_levels_and_splits():

    levels_and_splits = [
        ("instance", "leave-one-concept-out"),
        ("concept", "leave-one-concept-out"),
        ("concept", "repeated-k-fold"),
    ]
    results = [
        load_result(level, split, feature_type, feature)
        for feature in tqdm(FEATURES_SELECTED)
        for level, split in levels_and_splits
        for feature_type in FEATURE_TYPES
    ]
    df = pd.DataFrame(results)
    df = df.groupby(["level", "split", "model"])["score"].mean()
    df = df.reset_index()
    df = df.pivot_table(
        index=["level", "split"],
        columns="model",
        values="score",
    )
    print(df.to_csv())


def get_results_per_metacategory(level, split):
    def load_taxonomy():
        def get_majority(annotations):
            counts = Counter(annotations)
            top_elem, *_ = counts.most_common(1)
            metacategory, count = top_elem
            assert count >= 2
            return metacategory

        def get1(row):
            feature, _, annot1, annot2, annot3, *_ = row
            metacategory = get_majority([annot1, annot2, annot3])
            return feature, metacategory
            # return {
            #     "feature": feature,
            #     "metacategory": metacategory,
            # }

        path = "data/gpt-feature-norms-taxonomy.csv"
        with open(path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            _ = next(reader)
            _ = next(reader)
            results = [get1(row) for i, row in enumerate(reader) if i <= 99]
            results = dict(results)

        return results

    taxonomy = load_taxonomy()
    features = list(taxonomy.keys())

    results = [
        {
            **load_result(level, split, feature_type, feature),
            "metacategory": taxonomy[feature],
        }
        for feature in tqdm(features)
        for feature_type in FEATURE_TYPES
    ]

    df = pd.DataFrame(results)
    model_performance = df.groupby(["metacategory", "model"])["score"].mean()
    model_performance = model_performance.reset_index()
    order_models = model_performance.groupby("model")["score"].mean().sort_values().index
    order_metacategory = sorted(df["metacategory"].unique())

    # df = df.groupby(["metacategory", "model"])["score"].mean()
    # print(df)

    fig, ax = plt.subplots(figsize=(4, 14))
    sns.set(style="whitegrid", context="poster", font="Arial")
    sns.barplot(
        data=df,
        x="score",
        y="metacategory",
        hue="model",
        hue_order=order_models,
        order=order_metacategory,
        errorbar=None,
        ax=ax,
    )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_ylabel("")
    fig.set_tight_layout(True)
    st.pyplot(fig)


if __name__ == "__main__":
    get_results_levels_and_splits()
    # get_results_per_metacategory("concept", "repeated-k-fold")
