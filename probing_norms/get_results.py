import csv
import pickle
import pdb
import sys

from collections import Counter
from functools import partial
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from matplotlib import pyplot as plt
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    recall_score,
    precision_score,
)
from tqdm import tqdm

from probing_norms.utils import cache_df, cache_json, read_json, multimap
from probing_norms.predict import NORMS_LOADERS

NORMS_MODEL = "chatgpt-gpt3.5-turbo"
DATASET_NAME = "things"

SCORE_FUNCS = {
    "score-accuracy": accuracy_score,
    "score-precision": partial(precision_score, zero_division=0),
    "score-recall": recall_score,
    "score-f1": f1_score,
    "score-roc-auc": roc_auc_score,
}

SPLIT_TO_SCORE_FUNCS = {
    "leave-one-concept-out": ["score-accuracy"],
    "repeated-k-fold": ["score-precision", "score-recall", "score-f1"],
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


OUTPUT_PATH = "output/linear-probe-predictions/{}/{}/{}-{}-{}-{}"


# Names used for plotting or other output.
METACATEGORY_NAMES = {
    "visual-colour": "visual: color",
    "visual-form_and_surface": "visual: form & surface",
    "visual-motion": "visual: motion",
}

FEATURE_NAMES = {
    "fasttext-word": "FastText",
    "gemma-2b-word": "Gemma",
    "glove-6b-300d-word": "GloVe 6B",
    "glove-840b-300d-word": "GloVe 840B",
    "pali-gemma-224": "PaliGemma",
    "siglip-224": "SigLIP",
    "vit-mae-large": "ViT-MAE",
    "dino-v2": "DINO v2",
    "swin-v2": "Swin-V2",
    "max-vit-large": "Max ViT",
    "random-siglip": "Random SigLIP",
    "random-predictor": "Random predictor",
}

NORMS_NAMES = {
    "mcrae-mapped": "McRae++",
    "binder-4": "Binder",
}


def load_result(embeddings_level, split_type, feature_type, feature, kwargs_path):
    def evaluate(data):
        true = np.array([datum["true"] for datum in data])
        pred = np.array([datum["pred"] for datum in data])
        pred = (pred > 0.5).astype(int)
        return [
            {
                "score-type": score_type,
                "score": 100 * SCORE_FUNCS[score_type](true, pred),
            }
            for score_type in SPLIT_TO_SCORE_FUNCS[split_type]
        ]

    def get_path(*, feature_norm_str, feature_id):
        return OUTPUT_PATH.format(
            embeddings_level,
            split_type,
            DATASET_NAME,
            feature_type,
            feature_norm_str,
            feature_id,
        )

    path = get_path(**kwargs_path) + ".json"
    results = read_json(path)
    scores = [score for result in results for score in evaluate(result["preds"])]

    df = pd.DataFrame(scores)
    df = df.groupby("score-type")["score"].mean()
    scores_dict = df.to_dict()

    return {
        "feature": feature,
        "level": embeddings_level,
        "model": feature_type,
        "split": split_type,
        **scores_dict,
    }


def load_result_random_predictor(norms_loader):
    level = "concept"
    split = "repeated-k-fold"

    def get_random_vector(n, p):
        return np.random.choice([0, 1], size=n, p=[1 - p, p])

    def evaluate_feature(num_total, num_pos):
        n = num_total
        p = num_pos / n
        true = get_random_vector(n, p)
        pred = get_random_vector(n, p)
        return [
            {
                "score-type": score_type,
                "score": 100 * SCORE_FUNCS[score_type](true, pred),
            }
            for score_type in SPLIT_TO_SCORE_FUNCS[split]
        ]

    feature_to_concepts, _, features_selected = norms_loader()
    selected_concepts = norms_loader.load_concepts()
    num_concepts = len(selected_concepts)
    return [
        {
            "model": "random-predictor",
            "split": split,
            "level": level,
            "feature": feature,
            **{
                result["score-type"]: result["score"]
                for result in evaluate_feature(
                    num_concepts, len(feature_to_concepts[feature])
                )
            },
        }
        for feature in tqdm(features_selected)
    ]


def get_results_levels_and_splits():
    norms_loader = NORMS_LOADERS["generated-gpt35"]()
    _, feature_to_id, features_selected = norms_loader()

    levels_and_splits = [
        ("instance", "leave-one-concept-out"),
        ("concept", "leave-one-concept-out"),
        ("concept", "repeated-k-fold"),
    ]
    results = [
        load_result(
            level,
            split,
            feature_type,
            feature,
            {
                "feature_norm_str": norms_loader.get_suffix(),
                "feature_id": feature_to_id[feature],
            },
        )
        for feature in tqdm(features_selected)
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


def plot_results_per_metacategory(results):
    df = pd.DataFrame(results)
    # st.write(df)
    # dfx = df.pivot_table(index=["metacategory", "feature"], columns="model", values="score")
    # dfx = dfx.reset_index()
    # dfx.to_csv("output/results-per-feature.csv")
    df["model"] = df["model"].map(FEATURE_NAMES)
    df["metacategory"] = df["metacategory"].map(lambda x: METACATEGORY_NAMES.get(x, x))
    st.write(df)

    model_performance = df.groupby(["metacategory", "model"])["score-f1"].mean()
    model_performance = model_performance.reset_index()
    order_models = (
        model_performance.groupby("model")["score-f1"].mean().sort_values().index
    )
    order_metacategory = sorted(df["metacategory"].unique())

    fig, ax = plt.subplots(figsize=(3.75, 20))
    sns.set(style="whitegrid", context="poster", font="Arial")
    sns.barplot(
        data=df,
        x="score-f1",
        y="metacategory",
        hue="model",
        hue_order=order_models,
        order=order_metacategory,
        errorbar=None,
        ax=ax,
    )
    sns.move_legend(ax, "lower right", bbox_to_anchor=(1, 1), ncol=2, title="")
    ax.set_ylabel("")
    ax.set_xlabel("F1 score")
    # fig.set_tight_layout(True)
    st.pyplot(fig)
    return fig


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

    norms_loader = NORMS_LOADERS["generated-gpt35"]()
    _, feature_to_id, features_selected = norms_loader()

    results = [
        {
            **load_result(
                level,
                split,
                feature_type,
                feature,
                {
                    "feature_norm_str": norms_loader.get_suffix(),
                    "feature_id": feature_to_id[feature],
                },
            ),
            "metacategory": taxonomy[feature],
        }
        for feature in tqdm(features)
        for feature_type in FEATURE_TYPES
    ]

    plot_results_per_metacategory(results)


def get_results_per_metacategory_mcrae_mapped():
    def load_taxonomy():
        cols = ["Feature", "BR_Label"]
        df = pd.read_csv("data/norms/mcrae/CONCS_FEATS_concstats_brm.txt", sep="\t")
        df = df[cols]
        df = df.drop_duplicates()
        feature_metacategory = df.values.tolist()
        return dict(feature_metacategory)

    norms_loader = NORMS_LOADERS["mcrae-mapped"]()
    feature_to_concepts, feature_to_id, features_selected = norms_loader()
    taxonomy = load_taxonomy()
    level = "concept"
    split = "repeated-k-fold"
    MODELS = FEATURE_TYPES + ["fasttext-word", "glove-840b-300d-word"]

    def load_results():
        return [
            load_result(
                level,
                split,
                feature_type,
                feature,
                {
                    "feature_norm_str": norms_loader.get_suffix(),
                    "feature_id": feature_to_id[feature],
                },
            )
            for feature in tqdm(features_selected)
            for feature_type in MODELS
        ]

    results1 = cache_json("/tmp/per-metacategory-mcrae-mapped-text.json", load_results)
    results2 = load_result_random_predictor(norms_loader)
    results = results1 + results2

    for r in results:
        r["metacategory"] = taxonomy[r["feature"]]

    fig = plot_results_per_metacategory(results)
    fig.savefig("output/plots/per-metacategory-mcrae-mapped.pdf", bbox_inches="tight")


def get_results_binder_norms():
    level = "concept"
    split = "repeated-k-fold"

    norms_loader = NORMS_LOADERS["binder-4"]()
    feature_to_concepts, feature_to_id, features_selected = norms_loader()
    results = [
        {
            **load_result(
                level,
                split,
                feature_type,
                feature,
                {
                    "feature_norm_str": f"binder-{thresh}",
                    "feature_id": feature_to_id[feature],
                },
            ),
            "thresh": thresh,
        }
        for thresh in [4, 5]
        for feature in tqdm(features_selected)
        for feature_type in FEATURE_TYPES
    ]
    df = pd.DataFrame(results)
    order_models = df.groupby("model")["score"].mean().sort_values().index

    fig = sns.catplot(
        data=df,
        x="score",
        y="feature",
        hue="model",
        row="thresh",
        kind="bar",
        hue_order=order_models,
        height=16,
        aspect=0.5,
    )
    st.pyplot(fig)


def get_classifiers_agreement_binder_norms():
    level = "concept"
    split = "repeated-k-fold"

    thresh = 4
    norms_loader = NORMS_LOADERS[f"binder-{thresh}"]()
    feature_to_concepts, feature_to_id, features_selected = norms_loader()

    def get_agreement(level, split, feature_type, feature, kwargs_path):
        def get_path(*, feature_norm_str, feature_id):
            return OUTPUT_PATH.format(
                level,
                split,
                DATASET_NAME,
                feature_type,
                feature_norm_str,
                feature_id,
            )

        def cossim(c1, c2):
            norm1 = np.linalg.norm(c1)
            norm2 = np.linalg.norm(c2)
            return np.dot(c1, c2) / (norm1 * norm2)

        path = get_path(**kwargs_path) + ".pkl"
        with open(path, "rb") as f:
            outputs = pickle.load(f)

        coefs = [output["clf"].coef_.squeeze() for output in outputs]
        sims = [cossim(c1, c2) for c1, c2 in combinations(coefs, 2)]

        sims_mean = np.mean(sims)
        sims_std = np.std(sims)

        return {
            "feature": feature,
            "level": level,
            "model": feature_type,
            "split": split,
            "sim-mean": sims_mean,
            "sim-std": sims_std,
            "sim": "{:.2f}±{:.1f}".format(sims_mean, 2 * sims_std),
        }

    results = [
        get_agreement(
            level,
            split,
            feature_type,
            feature,
            {
                "feature_norm_str": norms_loader.get_suffix(),
                "feature_id": feature_to_id[feature],
            },
        )
        for feature in tqdm(features_selected)
        for feature_type in FEATURE_TYPES
    ]
    df = pd.DataFrame(results)
    df = df.pivot_table(index="feature", columns="model", values="sim-mean")
    df = df.reset_index()
    df["num-concepts"] = df["feature"].apply(lambda x: len(feature_to_concepts[x]))
    df.to_csv("output/classifier-agreement-binder-norms.csv")


def get_results_paper_tabel_main():
    level = "concept"
    split = "repeated-k-fold"

    def get_results(norms_type):
        norms_loader = NORMS_LOADERS[norms_type]()
        _, feature_to_id, features_selected = norms_loader()

        results = [
            load_result(
                level,
                split,
                feature_type,
                feature,
                {
                    "feature_norm_str": norms_loader.get_suffix(),
                    "feature_id": feature_to_id[feature],
                },
            )
            for feature in tqdm(features_selected)
            for feature_type in FEATURE_TYPES
        ]
        cols_score = SPLIT_TO_SCORE_FUNCS[split]
        df = pd.DataFrame(results)
        df = df.groupby(["model"])[cols_score].mean()
        return df

    dfs = {
        NORMS_NAMES[norms_type]: cache_df(
            f"/tmp/paper-main-table-{norms_type}-all-scores.pkl",
            get_results,
            norms_type,
        )
        for norms_type in ["mcrae-mapped", "binder-4"]
    }
    df = pd.concat(dfs, axis=1)
    df = df.sort_values(("McRae++", "score-f1"), ascending=True)
    df = df.reset_index()
    df["model"] = df["model"].map(FEATURE_NAMES)
    print(df.to_latex(float_format="%.1f", index=False))


def get_results_text_models():
    level = "concept"
    split = "repeated-k-fold"

    def get_results(norms_type):
        norms_loader = NORMS_LOADERS[norms_type]()
        _, feature_to_id, features_selected = norms_loader()

        results = [
            load_result(
                level,
                split,
                f"{f}-{m}",
                feature,
                {
                    "feature_norm_str": norms_loader.get_suffix(),
                    "feature_id": feature_to_id[feature],
                },
            )
            for feature in tqdm(features_selected)
            for f in ["fasttext", "gemma-2b", "glove-840b-300d"]
            for m in ["word"]
        ]
        cols_score = SPLIT_TO_SCORE_FUNCS[split]
        df = pd.DataFrame(results)
        df = df.groupby(["model"])[cols_score].mean()
        return df

    # df = cache_df(f"/tmp/text-models-mcrae-mapped.pkl", get_results, "mcrae-mapped")
    dfs = {
        NORMS_NAMES[norms_type]: get_results(norms_type)
        for norms_type in ["mcrae-mapped", "binder-4"]
    }
    df = pd.concat(dfs, axis=1)
    df = df.sort_values(("McRae++", "score-f1"), ascending=True)
    df = df.reset_index()
    df["model"] = df["model"].map(FEATURE_NAMES)
    print(df.to_latex(float_format="%.1f", index=False))


def get_results_per_feature_norm():
    level = "concept"
    split = "repeated-k-fold"
    EMBS = ["siglip-224", "fasttext-word", "glove-840b-300d-word", "gemma-2b-word"]

    norms_type = "mcrae-mapped"
    norms_loader = NORMS_LOADERS[norms_type]()
    feature_to_concepts, feature_to_id, features_selected = norms_loader()

    def get_results_1():
        return [
            load_result(
                level,
                split,
                emb,
                feature,
                {
                    "feature_norm_str": norms_loader.get_suffix(),
                    "feature_id": feature_to_id[feature],
                },
            )
            for feature in tqdm(features_selected)
            for emb in EMBS
        ]

    import random

    def random_score(feature, score_func):
        n = len(feature_to_concepts[feature])
        total_num_concepts = 1854
        pred = [random.random() <= 0.2 for _ in range(total_num_concepts)]
        true = np.zeros(total_num_concepts)
        true[:n] = 1
        return score_func(true, pred)

    results = cache_json("/tmp/per-feature-norm-xx.json", get_results_1)
    df = pd.DataFrame(results)
    df = df.pivot_table(index="feature", columns="model", values="score")
    df = df.reset_index()
    df["num-concepts"] = df["feature"].apply(lambda x: len(feature_to_concepts[x]))
    df["f1-random-pred"] = df["feature"].apply(
        partial(random_score, score_func=f1_score)
    )
    df["recall-random-pred"] = df["feature"].apply(
        partial(random_score, score_func=recall_score)
    )
    # cols = ["feature", "num-concepts", "f1-random-pred"] + EMBS
    # df = df[cols]
    # df.to_csv("output/results-per-feature-2.csv")

    fig, ax = plt.subplots()
    # sns.scatterplot(data=df, x="siglip-224", y="fasttext-word", ax=ax)
    # ax.plot([0, 100], [0, 100], color="gray", linestyle="--")
    # ax.set_aspect("equal", adjustable="box")
    fig = sns.lmplot(data=df, x="num-concepts", y="gemma-2b-word")
    # fig = sns.lmplot(data=df, x="num-concepts", y="recall-random-pred")
    fig.set_axis_labels("Number of concepts", "Score")
    st.pyplot(fig)


def get_results_random_predictor():
    split = "repeated-k-fold"

    def load_results(norms_type):
        norms_loader = NORMS_LOADERS[norms_type]()
        results = load_result_random_predictor(norms_loader)
        df = pd.DataFrame(results)
        cols_score = SPLIT_TO_SCORE_FUNCS[split]
        df = df.groupby("model")[cols_score].mean()
        return df

    dfs = {
        NORMS_NAMES[norms_type]: load_results(norms_type)
        for norms_type in ["mcrae-mapped", "binder-4"]
    }
    df = pd.concat(dfs, axis=1)
    df = df.reset_index()
    df["model"] = df["model"].map(FEATURE_NAMES)
    print(df.to_latex(float_format="%.1f", index=False))


FUNCS = {
    "levels-and-splits": get_results_levels_and_splits,
    "per-metacategory": partial(
        get_results_per_metacategory,
        "concept",
        "repeated-k-fold",
    ),
    "per-metacategory-mcrae-mapped": get_results_per_metacategory_mcrae_mapped,
    "binder-norms": get_results_binder_norms,
    "classifier-agreement-binder-norms": get_classifiers_agreement_binder_norms,
    "paper-table-main": get_results_paper_tabel_main,
    "text-models": get_results_text_models,
    "per-feature-norm": get_results_per_feature_norm,
    "random-predictor": get_results_random_predictor,
}


if __name__ == "__main__":
    what = sys.argv[1]
    FUNCS[what]()
