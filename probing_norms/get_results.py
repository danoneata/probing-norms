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
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from tqdm import tqdm

from probing_norms.utils import cache_df, cache_json, read_json, multimap
from probing_norms.predict import NORMS_LOADERS

NORMS_MODEL = "chatgpt-gpt3.5-turbo"
DATASET_NAME = "things"

SCORE_FUNCS = {
    "leave-one-concept-out": accuracy_score,
    "repeated-k-fold": f1_score,
    # "repeated-k-fold": roc_auc_score,
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
    # "fasttext-word": "fastText",
    "gemma-2b": "Gemma",
    "pali-gemma-224": "PaliGemma",
    "siglip-224": "SigLIP",
    "vit-mae-large": "ViT-MAE",
    "dino-v2": "DINO",
    "swin-v2": "Swin",
    "max-vit-large": "Max ViT",
    "random-siglip": "Random SigLIP",
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
        return SCORE_FUNCS[split_type](true, pred)

    def get_path(*, feature_norm_str, feature_id):
        return OUTPUT_PATH.format(
            embeddings_level,
            split_type,
            DATASET_NAME,
            feature_type,
            feature_norm_str,
            feature_id,
        )

    try:
        path = get_path(**kwargs_path) + ".json"
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

    model_performance = df.groupby(["metacategory", "model"])["score"].mean()
    model_performance = model_performance.reset_index()
    order_models = (
        model_performance.groupby("model")["score"].mean().sort_values().index
    )
    order_metacategory = sorted(df["metacategory"].unique())

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
    sns.move_legend(ax, "lower right", bbox_to_anchor=(1, 1), ncol=2, title="")
    ax.set_ylabel("")
    ax.set_xlabel("F1 score")
    fig.set_tight_layout(True)
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

    def load_results():
        return [
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
            for feature in tqdm(features_selected)
            for feature_type in FEATURE_TYPES
        ]

    results = cache_json("/tmp/per-metacategory-mcrae-mapped.json", load_results)
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
        df = pd.DataFrame(results)
        df = df.groupby(["model"])["score"].mean()
        return df

    dfs = {
        NORMS_NAMES[norms_type]: cache_df(
            f"/tmp/paper-main-table-{norms_type}.pkl",
            get_results,
            norms_type,
        )
        for norms_type in ["mcrae-mapped", "binder-4"]
    }
    df = pd.concat(dfs, axis=1)
    df = df.sort_values("McRae++", ascending=True)
    df = df.reset_index()
    df["model"] = df["model"].map(FEATURE_NAMES)
    print(df.to_latex(float_format="%.2f", index=False))


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
            # for f in ["fasttext", "gemma-2b", "gemma-2b-last"]
            # for m in ["word", "word-and-category"]
            for f in ["fasttext", "gemma-2b"]
            for m in ["word"]
        ]
        df = pd.DataFrame(results)
        df = df.groupby(["model"])["score"].mean()
        return df

    # df = cache_df(f"/tmp/text-models-mcrae-mapped.pkl", get_results, "mcrae-mapped")
    df = get_results("mcrae-mapped")
    print(df)

    df = get_results("binder-4")
    print(df)


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
}


if __name__ == "__main__":
    what = sys.argv[1]
    FUNCS[what]()
