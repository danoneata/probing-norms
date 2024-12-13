import numpy as np
import random
import pandas as pd
import streamlit as st

from functools import partial
from sklearn.metrics import average_precision_score, accuracy_score
from tqdm import tqdm

from probing_norms.data import (
    DATASETS,
    FEATURE_NORMS_OPTIONS,
    get_feature_to_concepts,
    load_gpt3_feature_norms,
)
from probing_norms.utils import read_json, cache_df, cache_json
from probing_norms.scripts.show_results_per_feature_norm import load_features_metadata


def show1(dataset, feature, concept, rows, embedding, predictions):
    rows[0].markdown("### Embedding model: `{}`".format(embedding))
    preds = [
        data["preds"]
        for data in predictions
        if data["split"]["test-concept"] == concept
    ]
    assert len(preds) == 1
    preds = preds[0]
    for datum in preds:
        score = datum["pred"]
        is_correct = score > 0.5
        is_correct_str = "✓" if is_correct else "✗"
        rows[1].markdown(
        """
        - score for "{}": {:.2f}
        - is correct: {}
        """.format(
                feature,
                score,
                is_correct_str,
            )
        )
        rows[1].image(dataset.get_image_path(datum["name"]))
        rows[1].markdown("---")


def main():
    st.set_page_config(page_title="Results for leave-one-concept-out setup")

    EMBS = "pali-gemma-224 vit-mae-large".split()
    dataset = DATASETS["things"]()
    norms_model = "chatgpt-gpt3.5-turbo"

    feature_to_concepts, feature_to_id = load_features_metadata(norms_model)
    features = sorted(feature_to_concepts.keys())

    random.seed(42)
    features_selected = random.sample(features, 64)
    features_selected = features_selected[:42]
    features_selected = sorted(features_selected)

    def get_path_preds(embedding_type, feature):
        norms_type = "_".join(["mcrae", norms_model, "30"])
        norms_feature_id = feature_to_id[feature]
        return "output/linear-probe-predictions/leave-one-concept-out/things-{}-{}-{}.json".format(
            embedding_type,
            norms_type,
            norms_feature_id,
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
            for r in get_results(read_json(get_path_preds(e, f)))
        ]

    path_scores = "/tmp/scores-leave-one-out.json"
    results = cache_json(path_scores, load_scores, features_selected)
    df = pd.DataFrame(results)

    with st.sidebar:
        feature = st.selectbox("Feature norm", features_selected)

        df_agg = df.groupby(["feature", "embedding"])["accuracy"].mean()
        df_agg = df_agg.reset_index()
        df_agg = df_agg.pivot(index="feature", columns="embedding", values="accuracy")
        with st.expander("Results"):
            st.markdown("Accuracies averaged across all concepts for the two models and available feature norms.")
            st.write(df_agg.style.format("{:.1f}"))

    concepts = feature_to_concepts[feature]
    concepts = sorted(concepts)

    df = df[df["feature"] == feature]

    df_scores = df.pivot(index="concept", columns="embedding", values="accuracy")
    df_scores = df_scores.reset_index()
    df_scores["diff"] = df_scores[EMBS[0]] - df_scores[EMBS[1]]

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
    keys = EMBS + ["diff"]
    st.write(df_scores.style.format({k: "{:.1f}" for k in keys}))

    df_agg = df.groupby(["embedding"])["accuracy"].mean()
    df_agg = df_agg.reset_index()

    st.markdown("Results averaged across all concepts")
    st.write(df_agg.style.format({"accuracy": "{:.1f}"}))
    st.markdown("---")

    st.markdown("## Individual predictions")
    concept = st.selectbox("Concept", concepts)

    num_cols = 2
    num_rows = 5

    table = [st.columns(num_cols) for _ in range(num_rows)]
    table = list(zip(*table))

    predictions = [read_json(get_path_preds(e, feature)) for e in EMBS]

    for i in range(num_cols):
        show1(
            dataset,
            feature,
            concept,
            table[i],
            EMBS[i],
            predictions[i],
        )


if __name__ == "__main__":
    main()
