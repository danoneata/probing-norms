import csv
import json
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

from adjustText import adjust_text
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
    "gemma-2b-contextual-last-word": "Gemma",
    "glove-6b-300d-word": "GloVe 6B",
    "glove-840b-300d-word": "GloVe 840B",
    "clip-word": "CLIP (text)",
    "pali-gemma-224": "PaliGemma",
    "siglip-224": "SigLIP",
    "clip": "CLIP (image)",
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


def load_result_model(norms_type, model):
    """Simplified function given the current experimental settings."""

    def do():
        level = "concept"
        split = "repeated-k-fold"

        norms_loader = NORMS_LOADERS[norms_type]()
        _, feature_to_id, features_selected = norms_loader()

        return [
            {
                "norms-type": norms_type,
                **load_result(
                    level,
                    split,
                    model,
                    feature,
                    {
                        "feature_norm_str": norms_loader.get_suffix(),
                        "feature_id": feature_to_id[feature],
                    },
                ),
            }
            for feature in tqdm(features_selected)
        ]

    return cache_json(f"/tmp/{norms_type}-{model}.json", do)


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


def get_score_random(norms_loader, feature):
    feature_to_concepts, _, _ = norms_loader()
    num_concepts_total = len(norms_loader.load_concepts())
    num_concepts = len(set(feature_to_concepts[feature]))
    return 100 * num_concepts / num_concepts_total


def load_taxonomy_mcrae():
    cols = ["Feature", "BR_Label"]
    df = pd.read_csv("data/norms/mcrae/CONCS_FEATS_concstats_brm.txt", sep="\t")
    df = df[cols]
    df = df.drop_duplicates()
    feature_metacategory = df.values.tolist()
    return dict(feature_metacategory)


def load_taxonomy_ours():
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
    taxonomy = load_taxonomy_ours()
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
    norms_type = "mcrae-mapped"
    norms_loader = NORMS_LOADERS[norms_type]()
    feature_to_concepts, feature_to_id, features_selected = norms_loader()
    taxonomy = load_taxonomy_mcrae()
    level = "concept"
    split = "repeated-k-fold"
    MODELS = FEATURE_TYPES + ["fasttext-word", "glove-840b-300d-word"]

    results1 = [r for m in MODELS for r in load_result_model(norms_type, m)]
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
    df.to_csv("output/classifier-agreement-binder-norms-2.csv")


def get_results_paper_tabel_main(model):
    results = [
        r for n in ["mcrae-mapped", "binder-4"] for r in load_result_model(n, model)
    ]
    cols = ["model", "norms-type", "score-f1", "score-precision", "score-recall"]
    df = pd.DataFrame(results)
    df = df[cols]
    df["model"] = df["model"].map(lambda x: FEATURE_NAMES.get(x, x))
    df = df.groupby(["norms-type", "model"]).mean()
    df = df.reset_index()
    df = df.pivot_table(index="model", columns="norms-type")
    cols = [
        (s, n)
        for n in ["mcrae-mapped", "binder-4"]
        for s in ["score-precision", "score-recall", "score-f1"]
    ]
    df = df[cols]
    df = df.reset_index()
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


def compare_two_models_scatterplot(model1, model2):
    norms_type = "mcrae-mapped"
    norms_loader = NORMS_LOADERS[norms_type]()
    taxonomy = load_taxonomy_mcrae()

    results = [
        r
        for m in [model1, model2]
        for r in cache_json(
            f"/tmp/{norms_type}-{m}.json", load_result_model, norms_type, m
        )
    ]

    for r in results:
        score_random = get_score_random(norms_loader, r["feature"])
        r["score-f1-selectivity"] = r["score-f1"] - score_random

    df = pd.DataFrame(results)
    df["metacategory"] = df["feature"].map(taxonomy)
    df["metacategory"] = df["metacategory"].map(lambda x: METACATEGORY_NAMES.get(x, x))
    cols = ["feature", "metacategory", "model", "score-f1-selectivity"]
    df = df[cols]
    df = df.set_index(["feature", "metacategory", "model"]).unstack(-1)
    df = df.reset_index()
    df.columns = ["-".join(c for c in cols if c).strip() for cols in df.columns.values]

    st.write(df)

    def pareto_front(points, dominates):
        return [
            p1
            for p1 in points
            if not any(dominates(p2, p1) for p2 in points if p1 != p2)
        ]

    def normalize_feature_name(text):
        SEP = "_-_"
        if SEP in text:
            prefix, text = text.split(SEP)
            assert prefix in {"beh", "inbeh", "eg", "has_units", "worn_by_men"}, prefix
        text = text.replace("_", " ").strip()
        return text

    def add_texts(ax, df):
        cols = [
            f"score-f1-selectivity-{model1}",
            f"score-f1-selectivity-{model2}",
            "feature",
        ]
        points = df[cols].values.tolist()
        points = [tuple(p) for p in points]
        xs = [x for x, _, _ in points]
        ys = [y for _, y, _ in points]
        points1 = pareto_front(
            points,
            dominates=lambda p, q: p[0] > q[0] and p[1] < q[1],
        )
        points1 = [p for p in points1 if p[0] >= 30]
        points2 = pareto_front(
            points,
            dominates=lambda p, q: p[0] < q[0] and p[1] > q[1],
        )
        points2 = [p for p in points2 if p[1] >= 30]
        points = points1 + points2
        points = list(set(points))
        texts = [
            ax.text(
                x,
                y,
                normalize_feature_name(word),
                ha="center",
                va="center",
                size=8,
            )
            for x, y, word in points
        ]
        adjust_text(
            texts,
            x=xs,
            y=ys,
            ax=ax,
            expand=(1.2, 2.1),
            force_text=(0.2, 0.6),
            # force_points=1.0,
            arrowprops=dict(arrowstyle="-", color="b", alpha=0.5),
        )

    fig, ax = plt.subplots()
    sns.set(style="whitegrid", font="Arial")
    sns.scatterplot(
        df,
        x=f"score-f1-selectivity-{model1}",
        y=f"score-f1-selectivity-{model2}",
        hue="metacategory",
        # marker="metacategory",
        ax=ax,
    )
    add_texts(ax, df)
    sns.move_legend(ax, "lower right", bbox_to_anchor=(1, 1), ncol=2, title="")
    ax.plot([0, 100], [0, 100], color="gray", linestyle="--")
    ax.set_xlabel("F1 selectivity · {}".format(FEATURE_NAMES[model1]))
    ax.set_ylabel("F1 selectivity · {}".format(FEATURE_NAMES[model2]))
    ax.set_aspect("equal", adjustable="box")
    st.pyplot(fig)

    fig.savefig(
        f"output/plots/scatterplot-{model1}-vs-{model2}.pdf", bbox_inches="tight"
    )


def get_correlation_between_models(norms_type):
    norms_loader = NORMS_LOADERS[norms_type]()
    _, _, features_selected = norms_loader()
    models = [
        "random-siglip",
        "max-vit-large",
        "vit-mae-large",
        "swin-v2",
        "dino-v2",
        "pali-gemma-224",
        "siglip-224",
        "clip",
        "fasttext-word",
        "glove-840b-300d-word",
        "gemma-2b-contextual-last-word",
        "clip-word",
    ]
    results = [r for model in models for r in load_result_model(norms_type, model)]
    cols = ["feature", "model", "score-f1"]
    feature_to_random_score = {feature: get_score_random(norms_loader, feature) for feature in norms_loader.load_concepts()}

    df = pd.DataFrame(results)
    df = df[cols]
    df["model"] = df["model"].map(lambda x: FEATURE_NAMES.get(x, x))
    # df["score-random"] = df["feature"].map(lambda x: get_score_random(norms_loader, x))
    # df["score-f1-selectivity"] = df["score-f1"] - df["score-random"]
    df = df.pivot_table(index="feature", columns="model", values="score-f1")
    corr_matrix = df.corr()

    sns.set(style="whitegrid", font="Arial")
    fig, ax = plt.subplots(figsize=(7, 7))
    models_names = [FEATURE_NAMES[m] for m in models]
    corr_matrix = corr_matrix.loc[models_names, models_names]
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        square=True,
        cbar=False,
        cmap=sns.cm.rocket_r,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    st.pyplot(fig)

    fig.savefig(f"output/plots/correlation-between-models-{norms_type}.pdf", bbox_inches="tight")


def prepare_results_for_stella():
    norms_type = "mcrae-mapped"
    norms_loader = NORMS_LOADERS[norms_type]()
    feature_to_concepts, feature_to_id, features_selected = norms_loader()
    taxonomy = load_taxonomy_mcrae()
    level = "concept"
    split = "repeated-k-fold"
    models = ["dino-v2", "gemma-2b-contextual-last-word"]

    def get_results_model(model):
        return [
            load_result(
                level,
                split,
                model,
                feature,
                {
                    "feature_norm_str": norms_loader.get_suffix(),
                    "feature_id": feature_to_id[feature],
                },
            )
            for feature in tqdm(features_selected)
        ]

    results = [
        {
            "model": model,
            **result,
        }
        for model in models
        for result in cache_json(
            f"/tmp/{norms_type}-{model}.json",
            lambda: get_results_model(model),
        )
    ]

    # Add metacategory
    for r in results:
        concepts = sorted(set(feature_to_concepts[r["feature"]]))
        r["concepts"] = concepts
        r["metacategory"] = taxonomy[r["feature"]]
        r["score-random"] = get_score_random(norms_loader, r["feature"])
        r["score-f1-selectivity"] = r["score-f1"] - r["score-random"]

    with open("output/results-for-stella.json", "w") as f:
        json.dump(results, f, indent=2)


def prepare_classifiers_for_stella(model):
    norms_type = "mcrae-mapped"
    norms_loader = NORMS_LOADERS[norms_type]()
    _, feature_to_id, features_selected = norms_loader()
    level = "concept"
    split = "repeated-k-fold"

    def get_path(feature):
        feature_id = feature_to_id[feature]
        return OUTPUT_PATH.format(
            level,
            split,
            DATASET_NAME,
            model,
            norms_loader.get_suffix(),
            feature_id,
        )

    def load_pickle(path):
        with open(path + ".pkl", "rb") as f:
            return pickle.load(f)

    def get_clfs(data):
        return [output["clf"] for output in data]

    data = {
        feature: get_clfs(load_pickle(get_path(feature)))
        for feature in tqdm(features_selected)
    }

    print(len(data))

    with open(f"output/classifiers-for-stella-{model}.pkl", "wb") as f:
        pickle.dump(data, f)


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
    "per-feature-norm": get_results_per_feature_norm,
    "random-predictor": get_results_random_predictor,
    "compare-two-models-scatterplot": compare_two_models_scatterplot,
    "get-correlation-between-models": get_correlation_between_models,
    "prepare-results-for-stella": prepare_results_for_stella,
    "prepare-classifiers-for-stella": prepare_classifiers_for_stella,
}


if __name__ == "__main__":
    what = sys.argv[1]
    what, *args = what.split(":")
    FUNCS[what](*args)
