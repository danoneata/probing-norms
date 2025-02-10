import pdb

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.metrics import f1_score

from probing_norms.data import DIR_THINGS
from probing_norms.utils import cache_df, cache_json, read_json, multimap
from probing_norms.predict import NORMS_LOADERS, DATASETS
from probing_norms.get_results import OUTPUT_PATH, FEATURE_NAMES

st.set_page_config(layout="wide")


def main():
    EMBEDDINGS_LEVEL = "concept"
    SPLIT_TYPE = "repeated-k-fold"
    DATASET_NAME = "things"

    dataset = DATASETS[DATASET_NAME]()

    MODELS = [
        "dino-v2",
        "fasttext-word",
        "clip",
        "clip-word",
    ]

    norms_type = "mcrae-mapped"
    norms_loader = NORMS_LOADERS[norms_type]()
    feature_to_concepts, feature_to_id, features_selected = norms_loader()

    num_concepts_total = len(norms_loader.load_concepts())
    labels = np.arange(num_concepts_total)

    with st.sidebar:
        feature = st.selectbox("Feature norm:", features_selected)

    def evaluate(results):
        def eval1(data):
            true = np.array([datum["true"] for datum in data])
            pred = np.array([datum["pred"] for datum in data])
            pred = (pred > 0.5).astype(int)
            return f1_score(true, pred)

        num_pos = len(set(feature_to_concepts[feature]))
        scores = [eval1(result["preds"]) for result in results]
        scores = np.array(scores)
        score = 100 * np.mean(scores)
        score_random = 100 * num_pos / num_concepts_total
        return score - score_random

    def get_path(model):
        feature_id = feature_to_id[feature]
        path = OUTPUT_PATH.format(
            EMBEDDINGS_LEVEL,
            SPLIT_TYPE,
            DATASET_NAME,
            model,
            norms_loader.get_suffix(),
            feature_id,
        )
        return path + ".json"

    def prepare_results(results):
        true = np.zeros(num_concepts_total)
        pred = np.zeros(num_concepts_total)
        for fold in range(5):
            for datum in results[fold]["preds"]:
                i = datum["i"]
                true[i] = datum["true"]
                pred[i] = datum["pred"] > 0.5
        return true, pred

    def get_tp_fp_fn(true, pred):
        def mapc(idxs):
            return [dataset.label_to_class[label] for label in idxs]
        tp, *_ = np.where([t and p for t, p in zip(true, pred)])
        fp, *_ = np.where([not t and p for t, p in zip(true, pred)])
        fn, *_ = np.where([t and not p for t, p in zip(true, pred)])
        return {
            "tp": mapc(tp),
            "fp": mapc(fp),
            "fn": mapc(fn),
        }

    results = {model: read_json(get_path(model)) for model in MODELS}
    scores = {m: evaluate(r) for m, r in results.items()}
    results_tp_fp_fn = {m: get_tp_fp_fn(*prepare_results(r)) for m, r in results.items()}

    num_models = len(MODELS)
    num_categories = 3
    table = [st.columns(num_models) for _ in range(1 + num_categories)]
    table = list(zip(*table))

    for i, model in enumerate(MODELS):
        table[i][0].markdown("## {}".format(model))
        for j, category in enumerate(["tp", "fp", "fn"]):
            table[i][j + 1].markdown("### {}".format(category))
            table[i][j + 1].markdown(", ".join(results_tp_fp_fn[model][category]))

    predicted_concepts = [
        concept
        for model in MODELS
        for category in ["tp", "fp", "fn"]
        for concept in results_tp_fp_fn[model][category]
    ]
    predicted_concepts = set(predicted_concepts)

    def get_image_path(concept):
        return DIR_THINGS / "object_images_CC0" / (concept + ".jpg")

    def get_pred_type(result, concept):
        if concept in result["tp"]:
            return "TP"
        elif concept in result["fp"]:
            return "FP"
        elif concept in result["fn"]:
            return "FN"
        else:
            return "TN"

    st.markdown("---")

    # cols = st.columns(1 + num_models)
    # cols[0].markdown("## Concept")
    # for i in range(num_models):
    #     model = MODELS[i]
    #     model_name = FEATURE_NAMES[model]
    #     cols[i + 1].markdown("## {} ({:.1f})".format(model_name, scores[model]))

    # for c in predicted_concepts:
    #     cols = st.columns(1 + num_models)
    #     cols[0].image(str(get_image_path(c)))
    #     cols[0].markdown("{}".format(c))
    #     for i in range(num_models):
    #         cols[i + 1].markdown(get_pred_type(results_tp_fp_fn[MODELS[i]], c))

    def transpose(table):
        return list(zip(*table))

    def get_image_block(concept):
        path = "images/qualitative-results/{}.jpg".format(concept)
        return r"\multirow{4}{*}{\includegraphics[width=0.1\textwidth]{" + path + r"}}"

    def generate_table(concept):
        block_image = [get_image_block(concept)]
        block_preds = [get_pred_type(results_tp_fp_fn[m], concept) for m in MODELS]
        block_preds = ["\\" + p for p in block_preds]
        block_image = block_image + (num_models - 1) * [""]
        table = list(zip(block_image, block_preds))
        return table

    def vcat(tables):
        return [row for table in tables for row in table]
    
    def hcat(tables):
        table = vcat(map(transpose, tables))
        return transpose(table)

    def to_emph(s):
        return r"\emph{" + s + r"}"

    def table_to_latex(table):
        return "\n".join(" & ".join(row) + r" \\" for row in table)

    predicted_concepts = list(predicted_concepts)[:5]
    st.write(" ".join(predicted_concepts))

    blocks = [generate_table(c) for c in predicted_concepts]
    header_left = [[FEATURE_NAMES[m], "{:.1f}".format(scores[m])] for m in MODELS]
    header_top = [["Model", "F1 sel"] + [e for elem in predicted_concepts for e in [to_emph(elem), ""]]]
    table = hcat([header_left] + blocks)
    table = vcat([header_top] + [table])
    first_row = r"& & \multicolumn{10}{c}{\texttt{" + feature + r"}} \\" + "\n"
    table_latex = first_row + table_to_latex(table)
    st.code(table_latex)


if __name__ == "__main__":
    main()