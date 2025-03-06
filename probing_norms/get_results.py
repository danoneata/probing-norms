import csv
import json
import pickle
import pdb
import os
import sys

from collections import Counter
from functools import partial
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from adjustText import adjust_text
from matplotlib import gridspec
from matplotlib import pyplot as plt
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    recall_score,
    precision_score,
)
from tqdm import tqdm

from probing_norms.data import DIR_LOCAL
from probing_norms.utils import cache_df, cache_json, read_json, read_file, multimap
from probing_norms.scripts.prepare_mcrae_norms_grouped import load_mapping_features
from probing_norms.predict import NORMS_LOADERS, FEATURE_TYPE_TO_MODALITY

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
    "clip",
]


OUTPUT_PATH = "output/{}-predictions/{}/{}/{}-{}-{}-{}"

MAIN_TABLE_MODELS = [
    "random-siglip",
    "vit-mae-large",
    "max-vit-large",
    "max-vit-large-in21k",
    "swin-v2-ssl",
    "dino-v2",
    #
    "siglip-224",
    "pali-gemma-224",
    "clip-dfn2b",
    "clip",
    #
    "glove-840b-300d-word",
    "fasttext-word",
    "deberta-v3-contextual-layers-0-to-6-word",
    "clip-word",
    "gemma-2b-contextual-layers-9-to-18-seq-last-word",
]


# Names used for plotting or other output.
METACATEGORY_SHORT_NAMES = {
    # "encyclopaedic": "encycl.",
    "visual-colour": "visual: color",
    "visual-form_and_surface": "visual: form & surface",
    "visual-motion": "visual: motion",
}

METACATEGORY_NAMES = {
    "visual-colour": "visual: color",
    "visual-form_and_surface": "visual: form & surface",
    "visual-motion": "visual: motion",
}

FEATURE_NAMES = {
    "random-predictor": "Random predictor",
    "random-siglip": "Random SigLIP",
    "vit-mae-large": "ViT-MAE",
    "max-vit-large": "Max ViT (IN-1K)",
    "max-vit-large-in21k": "Max ViT (IN-21K)",
    "swin-v2": "Swin-V2 (FT)",
    "swin-v2-ssl": "Swin-V2",
    "dino-v2": "DINOv2",
    "siglip-224": "SigLIP",
    "pali-gemma-224": "PaliGemma",
    "clip": "CLIP (image)",
    "clip-dfn2b": "CLIP DFN-2B (image)",
    "glove-6b-300d-word": "GloVe 6B",
    "glove-840b-300d-word": "GloVe 840B",
    "fasttext-word": "FastText",
    "deberta-v3-contextual-last-word": "DeBERTa v3",
    "deberta-v3-contextual-layers-0-to-6-word": "DeBERTa v3",
    "gemma-2b-word": "Gemma",
    "gemma-2b-contextual-last-word": "Gemma",
    "gemma-2b-contextual-layers-9-to-18-seq-last-word": "Gemma",
    "clip-word": "CLIP (text)",
}

NORMS_NAMES = {
    "mcrae-x-things": "McRae×THINGS",
    "mcrae-mapped": "McRae++",
    "binder-4": "Binder",
    "binder-median": "Binder",
}

SCORE_NAMES = {
    "score-accuracy": "Accuracy",
    "score-precision": "Precision",
    "score-recall": "Recall",
    "score-f1": "F1",
    "score-f1-selectivity": "F1 selectivity",
    "score-roc-auc": "ROC AUC",
}


# Second level of type aggregation.
BINDER_METACATEGORIES_2 = {
    "Vision": "Sensory",
    "Audition": "Sensory",
    "Somatic": "Sensory",
    "Gustation": "Sensory",
    "Olfaction": "Sensory",
    "Motor": "Motor",
    "Spatial": "Space",
    "Temporal": "Time",
    "Causal": "Time",
    "Social": "Social",
    "Cognition": "Social",
    "Emotion": "Emotion",
    "Drive": "Drive",
    "Attention": "Drive",
}


def normalize_feature_name(text):
    SEP = "_-_"
    if SEP in text:
        prefix, text = text.split(SEP)
        assert prefix in {"beh", "inbeh", "eg", "has_units", "worn_by_men"}, prefix
    text = text.replace("_", " ").strip()
    return text


def load_result_path(path, score_types):
    def evaluate(data):
        true = np.array([datum["true"] for datum in data])
        pred = np.array([datum["pred"] for datum in data])
        pred = (pred > 0.5).astype(int)
        return [
            {
                "score-type": score_type,
                "score": 100 * SCORE_FUNCS[score_type](true, pred),
            }
            for score_type in score_types
        ]

    results = read_json(path)
    scores = [score for result in results for score in evaluate(result["preds"])]

    df = pd.DataFrame(scores)
    df = df.groupby("score-type")["score"].mean()
    return df.to_dict()


def load_result(
    classifier_type,
    embeddings_level,
    split_type,
    feature_type,
    feature_norm_str,
    feature_id,
):
    path = (
        OUTPUT_PATH.format(
            classifier_type,
            embeddings_level,
            split_type,
            DATASET_NAME,
            feature_type,
            feature_norm_str,
            feature_id,
        )
        + ".json"
    )
    scores_dict = load_result_path(path, SPLIT_TO_SCORE_FUNCS[split_type])
    return {
        "level": embeddings_level,
        "model": feature_type,
        "split": split_type,
        **scores_dict,
    }


def load_result_features(
    classifier_type, embeddings_level, split_type, feature_type, norms_type
):
    """Aggregates results for all features."""

    def do():
        norms_loader = NORMS_LOADERS[norms_type]()
        _, feature_to_id, features_selected = norms_loader()
        return [
            {
                "norms-type": norms_type,
                "feature": feature,
                **load_result(
                    classifier_type,
                    embeddings_level,
                    split_type,
                    feature_type,
                    norms_loader.get_suffix(),
                    feature_to_id[feature],
                ),
            }
            for feature in tqdm(features_selected)
        ]

    path = "tmp/{}-{}-{}-{}-{}.json".format(
        classifier_type,
        embeddings_level,
        split_type,
        feature_type,
        norms_type,
    )
    return cache_json(path, do)


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


def get_score_random_features(norms_type):
    path = f"tmp/score-random-{norms_type}.json"
    if not os.path.exists(path):
        norms_loader = NORMS_LOADERS[norms_type]()
        _, _, features_selected = norms_loader()

    def do():
        return {
            feature: get_score_random(norms_loader, feature)
            for feature in features_selected
        }

    return cache_json(path, do)


def load_taxonomy_mcrae_x_things():
    def collapse(values):
        try:
            assert len(set(values)) == 1
        except AssertionError:
            print(values)
            # pdb.set_trace()
        return values[0]

    taxonomy_mcrae = load_taxonomy_mcrae()
    mapping = load_mapping_features()
    taxonomy = [(mapping.get(f, f), t) for f, t in taxonomy_mcrae.items()]
    taxonomy = multimap(taxonomy)
    taxonomy = {k: collapse(vs) for k, vs in taxonomy.items()}
    return taxonomy


def load_taxonomy_mcrae():
    cols = ["Feature", "BR_Label"]
    path = "data/norms/mcrae/CONCS_FEATS_concstats_brm.txt"
    path = DIR_LOCAL / path
    df = pd.read_csv(path, sep="\t")
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


def load_taxonomy_binder():
    def parse_line(line):
        metacategory, feature, *_ = line.split(" ")
        return feature, metacategory

    path = "data/binder-types.txt"
    path = DIR_LOCAL / path
    return dict(read_file(path, parse_line))


def load_taxonomy_binder_2():
    taxonomy = load_taxonomy_binder()
    return {k: BINDER_METACATEGORIES_2[v] for k, v in taxonomy.items()}


def get_results_levels_and_splits():
    classifier_type = "linear-probe"
    norms_loader = NORMS_LOADERS["generated-gpt35"]()
    _, feature_to_id, features_selected = norms_loader()

    levels_and_splits = [
        ("instance", "leave-one-concept-out"),
        ("concept", "leave-one-concept-out"),
        ("concept", "repeated-k-fold"),
    ]
    results = [
        {
            "feature": feature,
            **load_result(
                classifier_type,
                level,
                split,
                feature_type,
                norms_loader.get_suffix(),
                feature_to_id[feature],
            ),
        }
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


def plot_results_per_metacategory(results, order_models=None, order_metacategory=None):
    metric = "score-f1-selectivity"

    df = pd.DataFrame(results)
    df["modality"] = df["model"].map(FEATURE_TYPE_TO_MODALITY)
    df["model"] = df["model"].map(FEATURE_NAMES)

    idxs = df["model"] == df["model"].iloc[0]

    model_performance = df.groupby(["model", "modality"])[metric].mean()
    model_performance = model_performance.reset_index()

    if order_models:
        order_models = [FEATURE_NAMES[m] for m in order_models]
    else:
        order_models = model_performance.sort_values(["modality", metric])["model"]

    if not order_metacategory:
        order_metacategory = sorted(df["metacategory"].unique())

    def update_metacategory(name_orig):
        name = METACATEGORY_SHORT_NAMES.get(name_orig, name_orig)
        name = name.replace(": ", "\n")
        name = name + f"\n({count_metacategory[name_orig]})"
        return name

    count_metacategory = Counter(df[idxs]["metacategory"])
    df["metacategory"] = df["metacategory"].map(update_metacategory)
    order_metacategory = [update_metacategory(m) for m in order_metacategory]

    modalities = ["image", "text"]
    modality_to_color = {
        "image": "rocket_r",
        "text": "mako_r",
    }
    num_models = Counter(model_performance["modality"])
    palette = [
        c
        for m in modalities
        for c in sns.color_palette(modality_to_color[m], n_colors=num_models[m])
    ]

    # fig, ax = plt.subplots(figsize=(3.75, 20))
    sns.set(style="whitegrid", context="poster", font="Arial")
    fig, ax = plt.subplots(figsize=(24, 3.75))
    sns.barplot(
        data=df,
        y=metric,
        x="metacategory",
        hue="model",
        hue_order=order_models,
        order=order_metacategory,
        palette=palette,
        # errorbar=None,
        err_kws={"linewidth": 1},
        ax=ax,
    )
    sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1), ncol=6, title="")
    ax.set_xlabel("")
    ax.set_ylabel(SCORE_NAMES[metric])
    st.pyplot(fig)
    return fig


def get_results_per_metacategory(level, split):
    classifier_type = "linear-probe"
    taxonomy = load_taxonomy_ours()
    features = list(taxonomy.keys())

    norms_loader = NORMS_LOADERS["generated-gpt35"]()
    _, feature_to_id, features_selected = norms_loader()

    results = [
        {
            "metacategory": taxonomy[feature],
            "feature": feature,
            **load_result(
                classifier_type,
                level,
                split,
                feature_type,
                norms_loader.get_suffix(),
                feature_to_id[feature],
            ),
        }
        for feature in tqdm(features)
        for feature_type in FEATURE_TYPES
    ]

    plot_results_per_metacategory(results)


def get_results_per_metacategory_mcrae_mapped():
    classifier_type = "linear-probe"
    norms_type = "mcrae-mapped"
    taxonomy = load_taxonomy_mcrae()
    level = "concept"
    split = "repeated-k-fold"
    models = [
        "random-siglip",
        "vit-mae-large",
        "max-vit-large-in21k",
        "swin-v2-ssl",
        "dino-v2",
        # "siglip-224",
        "pali-gemma-224",
        "clip",
        "glove-840b-300d-word",
        "fasttext-word",
        "deberta-v3-contextual-layers-0-to-6-word",
        "clip-word",
        "gemma-2b-contextual-layers-9-to-18-seq-last-word",
    ]

    scores_random = get_score_random_features(norms_type)
    results = [
        r
        for m in models
        for r in load_result_features(classifier_type, level, split, m, norms_type)
    ]

    for r in results:
        r["metacategory"] = taxonomy[r["feature"]]
        r["score-f1-selectivity"] = r["score-f1"] - scores_random[r["feature"]]

    fig = plot_results_per_metacategory(results, models)
    fig.savefig("output/plots/per-metacategory-mcrae-mapped.pdf", bbox_inches="tight")


def get_results_per_metacategory_binder():
    classifier_type = "linear-probe"
    norms_type = "binder-median"
    taxonomy = load_taxonomy_binder()
    level = "concept"
    split = "repeated-k-fold"
    models = [
        "random-siglip",
        "vit-mae-large",
        "max-vit-large-in21k",
        "swin-v2-ssl",
        "dino-v2",
        # "siglip-224",
        "pali-gemma-224",
        "clip",
        "glove-840b-300d-word",
        "fasttext-word",
        "deberta-v3-contextual-layers-0-to-6-word",
        "clip-word",
        "gemma-2b-contextual-layers-9-to-18-seq-last-word",
    ]

    scores_random = get_score_random_features(norms_type)
    results = [
        r
        for m in models
        for r in load_result_features(classifier_type, level, split, m, norms_type)
    ]

    for r in results:
        r["metacategory"] = BINDER_METACATEGORIES_2[taxonomy[r["feature"]]]
        r["score-f1-selectivity"] = r["score-f1"] - scores_random[r["feature"]]

    order_metacategory = [
        "Sensory",
        "Motor",
        "Space",
        "Time",
        "Social",
        "Emotion",
        "Drive",
    ]
    fig = plot_results_per_metacategory(results, models, order_metacategory)
    fig.savefig("output/plots/per-metacategory-binder.pdf", bbox_inches="tight")


def get_results_binder_norms():
    classifier_type = "linear-probe"
    level = "concept"
    split = "repeated-k-fold"
    norm_type = "binder-median"
    metric = "score-f1-selectivity"

    MODELS = [
        "random-siglip",
        "vit-mae-large",
        "max-vit-large-in21k",
        "swin-v2-ssl",
        "dino-v2",
        # "siglip-224",
        "pali-gemma-224",
        "clip",
        #
        "glove-840b-300d-word",
        "fasttext-word",
        "deberta-v3-contextual-layers-0-to-6-word",
        "clip-word",
        "gemma-2b-contextual-layers-9-to-18-seq-last-word",
    ]

    results = [
        {
            "model": model,
            **result,
        }
        for model in MODELS
        for result in load_result_features(
            classifier_type, level, split, model, norm_type
        )
    ]
    score_random = get_score_random_features(norm_type)
    df = pd.DataFrame(results)
    df["score-f1-selectivity"] = df["score-f1"] - df["feature"].map(score_random)
    df["modality"] = df["model"].map(FEATURE_TYPE_TO_MODALITY)
    df["model"] = df["model"].map(FEATURE_NAMES)

    model_performance = df.groupby(["model", "modality"])[metric].mean()
    model_performance = model_performance.reset_index()
    # order_models = model_performance.sort_values(["modality", metric])["model"]
    order_models = [FEATURE_NAMES[m] for m in MODELS]
    order_features = read_file("data/binder-types.txt", lambda x: x.split()[1])

    taxonomy1 = load_taxonomy_binder()
    taxonomy2 = {k: BINDER_METACATEGORIES_2[v] for k, v in taxonomy1.items()}

    counts_taxonomy2 = Counter(taxonomy2.values())

    # df["metacategory-1"] = df["feature"].map(taxonomy)
    # df["metacategory-2"] = df["metacategory-1"].map(METACATEOGY_GROUPS)
    df = df.pivot_table(index="model", columns="feature", values=metric)
    df = df[order_features]
    df = df.reindex(order_models)

    rows = [
        ["Sensory", "Motor"],
        ["Space", "Time", "Social", "Emotion", "Drive"],
    ]

    rows_sizes = [[counts_taxonomy2[elem] for elem in row] for row in rows]

    st.write(df)

    def make1(ax, df, metacategory, to_hide_y):
        cols = [c for c in df.columns if taxonomy2[c] == metacategory]
        sns.heatmap(
            df[cols],
            annot=True,
            # square=True,
            cbar=False,
            fmt=".0f",
            cmap="rocket_r",
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(metacategory)
        if to_hide_y:
            ax.set_yticks([])
            ax.set_yticklabels([])
        # ax.set_yticklabels([])
        # ax.set_xticklabels([])

    def annotate_ax(ax):
        ax.text(0.5, 0.5, "ax%d" % (i + 1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)

    sns.set(style="whitegrid", font="Arial")

    fig = plt.figure(figsize=(11, 10))
    gs = gridspec.GridSpec(2, sum(rows_sizes[0]), hspace=0.4)
    for i, row in enumerate(rows):
        s = 0
        for j, elem in enumerate(row):
            e = s + rows_sizes[i][j]
            fig.add_subplot(gs[i, s:e])
            s = e

    i = 0
    for row in rows:
        for j, elem in enumerate(row):
            make1(fig.axes[i], df, elem, j > 0)
            # annotate_ax(fig.axes[i])
            i += 1

    # fig.tight_layout()
    st.pyplot(fig)
    fig.savefig("output/plots/binder-results-per-norm.pdf", bbox_inches="tight")


def get_classifiers_agreement_binder_norms():
    level = "concept"
    split = "repeated-k-fold"

    thresh = 4
    norms_loader = NORMS_LOADERS[f"binder-{thresh}"]()
    feature_to_concepts, feature_to_id, features_selected = norms_loader()

    def get_agreement(level, split, feature_type, feature, kwargs_path):
        def get_path(*, feature_norm_str, feature_id):
            return OUTPUT_PATH.format(
                "linear-probe",
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


def get_results_paper_table_main_row(*models):
    classifier_type = "linear-probe"
    embeddings_level = "concept"
    split_type = "repeated-k-fold"
    norm_types = ["mcrae-x-things", "mcrae-mapped", "binder-median"]

    scores_random_features = {k: get_score_random_features(k) for k in norm_types}

    def add_f1_sel(results, norm_type):
        for r in results:
            r["score-f1-selectivity"] = (
                r["score-f1"] - scores_random_features[norm_type][r["feature"]]
            )
        return results

    results = [
        r
        for n in norm_types
        for m in models
        for r in add_f1_sel(
            load_result_features(classifier_type, embeddings_level, split_type, m, n), n
        )
    ]
    cols = [
        "model",
        "norms-type",
        "score-f1",
        "score-precision",
        "score-recall",
        "score-f1-selectivity",
    ]
    df = pd.DataFrame(results)
    df = df[cols]
    df = df.groupby(["norms-type", "model"]).mean()
    df = df.reset_index()
    df = df.pivot_table(index="model", columns="norms-type")
    cols = [
        (s, n)
        for n in norm_types
        for s in ["score-precision", "score-recall", "score-f1", "score-f1-selectivity"]
    ]
    df = df[cols]
    df = df.reindex(models)
    df = df.reset_index()
    df["model"] = df["model"].map(lambda x: FEATURE_NAMES.get(x, x))
    print(df.to_latex(float_format="%.1f", index=False))


def get_results_paper_table_main():
    get_results_paper_table_main_row(*MAIN_TABLE_MODELS)


def get_results_all_one_norm(feature):
    classifier_type = "linear-probe"
    embeddings_level = "concept"
    splits_type = "repeated-k-fold"
    norms_type = "mcrae-mapped"

    feature_to_random_score = get_score_random_features(norms_type)
    results = [
        result
        for m in MAIN_TABLE_MODELS
        for result in load_result_features(
            classifier_type, embeddings_level, splits_type, m, norms_type
        )
        if result["feature"] == feature
    ]
    for r in results:
        r["score-f1-selectivity"] = (
            r["score-f1"] - feature_to_random_score[r["feature"]]
        )
    df = pd.DataFrame(results)
    df = df.pivot_table(index="model", columns="feature", values="score-f1-selectivity")
    df = df.reindex(MAIN_TABLE_MODELS)
    df = df.reset_index()
    df["model"] = df["model"].map(FEATURE_NAMES)
    print(df.to_string(index=False))


def get_results_per_feature_norm():
    EMBS = ["siglip-224", "fasttext-word", "glove-840b-300d-word", "gemma-2b-word"]
    classifier_type = "linear-probe"
    embeddings_level = "concept"
    splits_type = "repeated-k-fold"
    norms_type = "mcrae-mapped"
    norms_loader = NORMS_LOADERS[norms_type]()
    feature_to_concepts, feature_to_id, features_selected = norms_loader()

    import random

    def random_score(feature, score_func):
        n = len(feature_to_concepts[feature])
        total_num_concepts = 1854
        pred = [random.random() <= 0.2 for _ in range(total_num_concepts)]
        true = np.zeros(total_num_concepts)
        true[:n] = 1
        return score_func(true, pred)

    results = [
        result
        for m in EMBS
        for result in load_result_features(
            classifier_type, embeddings_level, splits_type, m, norms_type
        )
    ]
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
        for norms_type in ["mcrae-mapped", "binder-median"]
    }
    df = pd.concat(dfs, axis=1)
    df = df.reset_index()
    df["model"] = df["model"].map(FEATURE_NAMES)
    print(df.to_latex(float_format="%.1f", index=False))


def compare_two_models_scatterplot_ax(ax, model1, model2, legend="auto"):
    classifier_type = "linear-probe"
    embeddings_level = "concept"
    splits_type = "repeated-k-fold"
    norms_type = "mcrae-mapped"
    norms_loader = NORMS_LOADERS[norms_type]()
    taxonomy = load_taxonomy_mcrae()

    results = [
        r
        for m in [model1, model2]
        for r in load_result_features(
            classifier_type, embeddings_level, splits_type, m, norms_type
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

    metacategories = sorted(df["metacategory"].unique())
    sns.scatterplot(
        df,
        x=f"score-f1-selectivity-{model1}",
        y=f"score-f1-selectivity-{model2}",
        hue="metacategory",
        hue_order=metacategories,
        # marker="metacategory",
        legend=legend,
        ax=ax,
    )
    add_texts(ax, df)
    if legend:
        sns.move_legend(ax, "lower right", bbox_to_anchor=(1, 1), ncol=2, title="")
    ax.plot([0, 100], [0, 100], color="gray", linestyle="--")
    ax.set_xlabel("F1 selectivity · {}".format(FEATURE_NAMES[model1]))
    ax.set_ylabel("F1 selectivity · {}".format(FEATURE_NAMES[model2]))
    ax.set_aspect("equal", adjustable="box")


def compare_two_models_scatterplot(model1, model2):
    sns.set(style="whitegrid", font="Arial")
    fig, ax = plt.subplots()
    compare_two_models_scatterplot_ax(ax, model1, model2)
    st.pyplot(fig)
    fig.savefig(
        f"output/plots/scatterplot-{model1}-vs-{model2}.pdf", bbox_inches="tight"
    )


def compare_two_models_scatterplot_2():
    sns.set(style="whitegrid", font="Arial")
    models = [
        ["gemma-2b-contextual-layers-9-to-18-seq-last-word", "dino-v2"],
        ["clip-word", "clip"],
    ]
    fig, axs = plt.subplots(figsize=(10, 10), ncols=2, nrows=1, sharex=True)
    compare_two_models_scatterplot_ax(axs[0], *models[0], legend="auto")
    compare_two_models_scatterplot_ax(axs[1], *models[1], legend=False)
    sns.move_legend(axs[0], "lower left", bbox_to_anchor=(0, 1), ncol=5, title="")
    st.pyplot(fig)
    fig.savefig(
        f"output/plots/scatterplot-model-comparison.pdf",
        bbox_inches="tight",
    )


def get_correlation_between_models(norms_type):
    classifier_type = "linear-probe"
    embeddings_level = "concept"
    splits_type = "repeated-k-fold"
    norms_loader = NORMS_LOADERS[norms_type]()
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
        "clip-word",
    ]
    results = [
        r
        for m in models
        for r in load_result_features(
            classifier_type, embeddings_level, splits_type, m, norms_type
        )
    ]
    cols = ["feature", "model", "score-f1"]
    feature_to_random_score = {
        feature: get_score_random(norms_loader, feature)
        for feature in norms_loader.load_concepts()
    }

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

    fig.savefig(
        f"output/plots/correlation-between-models-{norms_type}.pdf",
        bbox_inches="tight",
    )


def get_correlation_between_models_2():
    classifier_type = "linear-probe"
    embeddings_level = "concept"
    splits_type = "repeated-k-fold"
    models = [
        "random-siglip",
        "vit-mae-large",
        "max-vit-large",
        "max-vit-large-in21k",
        # "swin-v2",
        "swin-v2-ssl",
        "dino-v2",
        "pali-gemma-224",
        "siglip-224",
        "clip",
        "glove-840b-300d-word",
        "fasttext-word",
        "deberta-v3-contextual-layers-0-to-6-word",
        "clip-word",
        "gemma-2b-contextual-layers-9-to-18-seq-last-word",
    ]

    def plot1(ax, norms_type):
        results = [
            r
            for m in models
            for r in load_result_features(
                classifier_type, embeddings_level, splits_type, m, norms_type
            )
        ]
        cols = ["feature", "model", "score-f1"]

        df = pd.DataFrame(results)
        df = df[cols]
        df["model"] = df["model"].map(lambda x: FEATURE_NAMES.get(x, x))
        # df["score-random"] = df["feature"].map(lambda x: get_score_random(norms_loader, x))
        # df["score-f1-selectivity"] = df["score-f1"] - df["score-random"]
        df = df.pivot_table(index="feature", columns="model", values="score-f1")
        corr_matrix = df.corr()

        models_names = [FEATURE_NAMES[m] for m in models]
        corr_matrix = corr_matrix.loc[models_names, models_names]
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt="0.2f",
            square=True,
            cbar=False,
            cmap=sns.cm.rocket_r,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(NORMS_NAMES[norms_type])
        for text in ax.texts:
            t = text.get_text()
            if t.startswith("0."):
                t1 = t.lstrip("0")
            elif t == "1.00":
                t1 = "1.0"
            else:
                t1 = t
            text.set_text(t1)

    sns.set(style="whitegrid", font="Arial")
    fig, axs = plt.subplots(
        figsize=(10, 18), ncols=2, nrows=1, sharex=True, sharey=True
    )
    plot1(axs[0], "mcrae-mapped")
    plot1(axs[1], "binder-median")
    fig.tight_layout()
    st.pyplot(fig)

    fig.savefig(
        f"output/plots/correlation-between-models.pdf",
        bbox_inches="tight",
    )


def prepare_results_for_stella():
    classifier_type = "linear-probe"
    embeddings_level = "concept"
    splits_type = "repeated-k-fold"
    norms_type = "mcrae-mapped"
    norms_loader = NORMS_LOADERS[norms_type]()
    # models = ["dino-v2", "gemma-2b-contextual-last-word"]
    models = [
        "pali-gemma-224",
        "siglip-224",
        "vit-mae-large",
        "dino-v2",
        "swin-v2",
        "swin-v2-ssl",
        "max-vit-large",
        "max-vit-large-in21k",
        "random-siglip",
        "clip",
        "fasttext-word",
        "glove-840b-300d-word",
        "deberta-v3-contextual-layers-0-to-6-word",
        "gemma-2b-contextual-layers-9-to-18-seq-last-word",
        "clip-word",
    ]

    feature_to_concepts, _, _ = norms_loader()
    taxonomy = load_taxonomy_mcrae()

    results = [
        {
            "model": m,
            **result,
        }
        for m in models
        for result in load_result_features(
            classifier_type, embeddings_level, splits_type, m, norms_type
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
            "linear-probe",
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

    with open(f"output/classifiers-for-stella-{model}.pkl", "wb") as f:
        pickle.dump(data, f)


def get_results_multiple_classifiers():
    embeddings_level = "concept"
    split_type = "repeated-k-fold"
    feature_type = "gemma-2b-contextual-last-word"
    feature_norm_str = "mcrae-mapped"
    classifiers = [
        "linear-probe",
        "linear-probe-std",
        "knn-3",
    ]

    results = [
        {
            "classifier": c,
            **r,
        }
        for c in classifiers
        for r in load_result_features(
            c, embeddings_level, split_type, feature_type, feature_norm_str
        )
    ]
    df = pd.DataFrame(results)
    cols = ["classifier", "score-f1", "score-precision", "score-recall"]
    df = df[cols]
    df = df.groupby("classifier").mean()
    print(df)


def get_norm_completeness():
    NAMES = {
        "mcrae": "McRae",
        "generated-gpt35": "Hansen",
        "mcrae-mapped": "McRae++",
    }

    def do1(norm_type, use_selected):
        norm_loader = NORMS_LOADERS[norm_type]()
        feature_to_concepts, _, features_selected = norm_loader()
        if use_selected:
            features = features_selected
        else:
            features = feature_to_concepts.keys()
        num_norms = len(features)
        num_norms_with_n_concepts = sum(
            1 for f in features if len(feature_to_concepts[f]) >= 10
        )
        use_selected_str = "✓" if use_selected else "✗"
        print(
            "{:10s} {:s} → {:5d} ({:.1f}%)".format(
                NAMES[norm_type],
                use_selected_str,
                num_norms_with_n_concepts,
                100 * num_norms_with_n_concepts / num_norms,
            )
        )

    do1("mcrae", False)
    do1("generated-gpt35", False)
    do1("mcrae-mapped", False)
    do1("mcrae-mapped", True)


def aggregate_predictions_for_demo(norms_type):
    CLASSIFIER_TYPE = "linear-probe"
    EMBEDDINGS_LEVEL = "concept"
    SPLIT_TYPE = "repeated-k-fold"
    DATASET_NAME = "things"

    norms_loader = NORMS_LOADERS[norms_type]()
    _, feature_to_id, features_selected = norms_loader()
    concepts = norms_loader.load_concepts()
    num_concepts = len(concepts)

    def get_path_results(model, feature):
        feature_id = feature_to_id[feature]
        path = OUTPUT_PATH.format(
            CLASSIFIER_TYPE,
            EMBEDDINGS_LEVEL,
            SPLIT_TYPE,
            DATASET_NAME,
            model,
            norms_loader.get_suffix(),
            feature_id,
        )
        return path + ".json"

    def get_five_folds(results):
        data = [datum for fold in range(5) for datum in results[fold]["preds"]]
        data = sorted(data, key=lambda datum: datum["i"])
        indices = [datum["i"] for datum in data]
        assert indices == list(range(num_concepts))
        preds = [datum["pred"] for datum in data]
        return preds

    results = [
        [
            get_five_folds(read_json(get_path_results(model, feature)))
            for model in MAIN_TABLE_MODELS
        ]
        for feature in features_selected
    ]

    path = f"static/demo/predictions-{norms_type}.npz"
    results_np = np.array(results)
    results_np = results_np.astype(np.float32)
    np.savez(path, results=results_np)


FUNCS = {
    "levels-and-splits": get_results_levels_and_splits,
    "per-metacategory": partial(
        get_results_per_metacategory,
        "concept",
        "repeated-k-fold",
    ),
    "per-metacategory-mcrae-mapped": get_results_per_metacategory_mcrae_mapped,
    "per-metacategory-binder": get_results_per_metacategory_binder,
    "binder-norms": get_results_binder_norms,
    "classifier-agreement-binder-norms": get_classifiers_agreement_binder_norms,
    "paper-table-main-row": get_results_paper_table_main_row,
    "paper-table-main": get_results_paper_table_main,
    "per-feature-norm": get_results_per_feature_norm,
    "random-predictor": get_results_random_predictor,
    "results-all-one-norm": get_results_all_one_norm,
    "compare-two-models-scatterplot": compare_two_models_scatterplot,
    "compare-two-models-scatterplot-2": compare_two_models_scatterplot_2,
    "correlation-between-models": get_correlation_between_models,
    "correlation-between-models-2": get_correlation_between_models_2,
    "results-multiple-classifiers": get_results_multiple_classifiers,
    "results-for-stella": prepare_results_for_stella,
    "classifiers-for-stella": prepare_classifiers_for_stella,
    "get-norm-completeness": get_norm_completeness,
    "aggregate-predictions-for-demo": aggregate_predictions_for_demo,
}


if __name__ == "__main__":
    what = sys.argv[1]
    what, *args = what.split(":")
    FUNCS[what](*args)
