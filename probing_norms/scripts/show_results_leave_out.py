import numpy as np
import random
import pandas as pd
import streamlit as st

from sklearn.metrics import average_precision_score, accuracy_score
from toolz import first

from probing_norms.data import DATASETS
from probing_norms.utils import read_json, cache_json
from probing_norms.scripts.show_results_per_feature_norm import load_features_metadata


def show_images(dataset, rows, preds, idxs):
    rows[0].markdown("Embedding model ­­→")
    for i, idx in enumerate(idxs, 1):
        datum = preds[idx]
        rows[i].image(dataset.get_image_path(datum["name"]))


def show1(feature, rows, embedding, preds, idxs):
    rows[0].markdown("`{}`".format(embedding))
    for i, idx in enumerate(idxs, 1):
        datum = preds[idx]
        score = datum["pred"]
        is_correct = score > 0.5
        is_correct_str = "✅" if is_correct else "❌"
        rows[i].markdown(
        """
        {}?
        - score: {:.2f}
        - is correct: {}
        """.format(
                feature,
                score,
                is_correct_str,
            )
        )


def main():
    st.set_page_config(page_title="Results for leave-one-concept-out setup")

    EMBS = "pali-gemma-224 siglip-224 vit-mae-large swin-v2 max-vit-large".split()
    dataset = DATASETS["things"]()
    norms_model = "chatgpt-gpt3.5-turbo"

    feature_to_concepts, feature_to_id = load_features_metadata(model=norms_model)
    features = sorted(feature_to_concepts.keys())

    random.seed(42)
    features_selected_1 = random.sample(features, 64)
    # features_selected_2 = random.sample(features, 256 - 64)
    features_selected_2 = []
    features_selected = sorted(features_selected_1 + features_selected_2)

    def get_path(embedding_type, feature, ext):
        norms_type = "_".join(["mcrae", norms_model, "30"])
        norms_feature_id = feature_to_id[feature]
        return "output/linear-probe-predictions/instance/leave-one-concept-out/things-{}-{}-{}.{}".format(
            embedding_type,
            norms_type,
            norms_feature_id,
            ext,
        )

    def compute_accuracy(data):
        true = [datum["true"] for datum in data]
        pred = [datum["pred"] for datum in data]
        pred = np.array(pred) > 0.5
        return 100 * accuracy_score(true, pred)

    def get_results(preds):
        return [
            {
                "accuracy": compute_accuracy(entry["preds"]),
                "concept": entry["split"]["test-concept"],
            }
            for entry in preds
        ]

    def load_scores(features_selected):
        return [
            {
                "embedding": e,
                "feature": f,
                **r,
            }
            for e in EMBS
            for f in features_selected
            for r in get_results(read_json(get_path(e, f, "json")))
        ]

    path_scores = "/tmp/scores-leave-one-out.json"
    results = cache_json(path_scores, load_scores, features_selected)
    df = pd.DataFrame(results)

    with st.sidebar:
        feature = st.selectbox("Feature norm", features_selected)

        df_agg = df.groupby(["feature", "embedding"])["accuracy"].mean()
        df_agg = df_agg.reset_index()
        df_agg = df_agg.pivot(index="feature", columns="embedding", values="accuracy")
        df_agg = df_agg[EMBS]
        
        with st.expander("Results"):
            st.markdown("Accuracies averaged across all concepts for the {} models and available feature norms.".format(len(EMBS)))
            st.write(df_agg.style.format("{:.1f}"))
            st.markdown("Average results:")
            st.write(df_agg.mean().to_frame().T.style.format("{:.1f}"))

    concepts = feature_to_concepts[feature]
    concepts = sorted(concepts)

    df = df[df["feature"] == feature]

    df_scores = df.pivot(index="concept", columns="embedding", values="accuracy")
    df_scores = df_scores[EMBS]
    # df_scores = df_scores.reset_index()
    # df_scores["diff"] = df_scores[EMBS[0]] - df_scores[EMBS[1]]

    st.markdown(
        """
        ## Aggregated per-concept results

        Accuracy for each left-one-out concept.
        There are {} (positive) concepts associated with the feature "{}".
        We leave all the samples of that concept out and train the linear probe on the rest.
        We predict on the left-out samples and compute the accuracy:
        if the prediction score is greater than 0.5, then we consider it correct.

        Note: You can click on the table headers to sort the values.
    """.format(
            len(concepts), feature
        )
    )
    st.write(df_scores.style.format({k: "{:.1f}" for k in EMBS}))

    st.markdown("Results averaged across all concepts")
    st.write(df_scores.mean().to_frame().T.style.format("{:.1f}"))
    st.markdown("---")

    st.markdown("## Individual predictions")
    cols = st.columns(2)
    concept = cols[0].selectbox("Concept", concepts)
    sort_by = cols[1].selectbox("Sort by", ["image"] + EMBS)

    has_concept = lambda datum: datum["split"]["test-concept"] == concept
    predictions = [read_json(get_path(e, feature, "json")) for e in EMBS]
    predictions = [
        first(filter(has_concept, preds))["preds"]
        for preds in predictions
    ]

    # def read_pkl(path):
    #     import pickle
    #     with open(path, "rb") as f:
    #         return pickle.load(f)

    # clfs = [read_pkl(get_path(e, feature, "pkl")) for e in EMBS]
    # clfs = [
    #     first(filter(has_concept, datum))["clf"]
    #     for datum in clfs
    # ]

    idxs = list(range(len(predictions[0])))

    if sort_by != "image":
        j = EMBS.index(sort_by)
        idxs = sorted(idxs, key=lambda i: predictions[j][i]["pred"], reverse=True)

    num_cols = 1 + len(EMBS)
    num_rows = 1 + len(predictions[0])

    table = [st.columns(num_cols) for _ in range(num_rows)]
    table = list(zip(*table))

    show_images(dataset, table[0], predictions[0], idxs)

    for i in range(1, num_cols):
        show1(
            feature,
            table[i],
            EMBS[i - 1],
            predictions[i - 1],
            idxs,
        )


if __name__ == "__main__":
    main()
