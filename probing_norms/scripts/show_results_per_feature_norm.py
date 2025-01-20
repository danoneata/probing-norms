import numpy as np
import random
import pandas as pd
import streamlit as st

from functools import partial
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from probing_norms.constants import NUM_MIN_CONCEPTS
from probing_norms.data import (
    DATASETS,
    FEATURE_NORMS_OPTIONS,
    load_features_metadata,
)
from probing_norms.utils import read_json, cache_df


def compute_score(predictions):
    true = [datum["true"] for datum in predictions]
    pred = [datum["pred"] for datum in predictions]
    return 100 * average_precision_score(true, pred)


def show1(rows, feature, feature_to_concepts, predictions, dataset):
    concepts = sorted(feature_to_concepts[feature])
    score = compute_score(predictions)

    rows[0].markdown("## {}".format(feature))
    rows[1].markdown("Number of concepts: {}".format(len(concepts)))
    rows[2].markdown("Concepts: {}".format(", ".join(concepts)))
    rows[3].markdown("Average precision: {:.2f}".format(score))

    def is_correct(datum):
        class_ = dataset.label_to_class[datum["label"]]
        return class_ in concepts

    def show_datum(row, datum):
        score = datum["pred"]
        label = datum["label"]
        class_ = dataset.label_to_class[label]
        is_correct = class_ in concepts
        is_correct_str = "✓" if is_correct else "✗"
        row.image(dataset.get_image_path(datum["name"]))
        row.markdown(
            "score: {:.2f} · {} · is correct: {}".format(
                score,
                class_,
                is_correct_str,
            )
        )

    topk = 5

    rows[4].markdown("### Top {} predictions".format(topk))
    preds = sorted(predictions, key=lambda datum: -datum["pred"])
    for datum in preds[:topk]:
        show_datum(rows[4], datum)

    rows[5].markdown("### Top {} incorrect predictions".format(topk))
    preds_incorrect = [datum for datum in preds if not is_correct(datum)]
    for datum in preds_incorrect[:topk]:
        show_datum(rows[5], datum)


def create_selectbox(i, features):
    if "selected-indices" in st.session_state:
        index = st.session_state["selected-indices"][i]
    else:
        index = i
    return st.selectbox(
        "Feature norm #{}".format(i + 1),
        features,
        index=index,
    )


def main():
    st.set_page_config(page_title="Results per feature norms", layout="wide")
    dataset = DATASETS["things"]()

    num_cols = 4
    num_rows = 6

    def get_path_preds(feature):
        feature_norm = "_".join(["mcrae", model, "30"])
        feature_id = feature_to_id[feature]
        return (
            "output/linear-probe-predictions/old/things-pali-gemma-224-{}-{}.json".format(
                feature_norm,
                feature_id,
            )
        )

    def clear_session_state():
        try:
            st.session_state.pop("selected-indices")
        except KeyError:
            pass

    def pick_random_features(features):
        indices = list(range(len(features)))
        selected_indices = random.sample(indices, num_cols)
        st.session_state["selected-indices"] = selected_indices

    def load_feature_scores(features):
        scores = [
            {
                "feature": feature,
                "score": compute_score(read_json(get_path_preds(feature))),
            }
            for feature in tqdm(features)
        ]
        df = pd.DataFrame(scores)
        df = df.sort_values("score", ascending=False)
        return df

    with st.sidebar:
        model = st.selectbox(
            "Feature norm model",
            FEATURE_NORMS_OPTIONS["model"],
            on_change=clear_session_state,
        )
        st.markdown("---")

        feature_to_concepts, feature_to_id = load_features_metadata(model=model)
        features = [f for f, cs in feature_to_concepts.items() if len(cs) >= NUM_MIN_CONCEPTS]
        features = sorted(features)
        features_selected = [create_selectbox(i, features) for i in range(num_cols)]

        st.button(
            "Pick random features",
            on_click=partial(pick_random_features, features),
        )
        st.markdown("---")

        path_scores = "/tmp/scores-{}.pkl".format(model)
        df_scores = cache_df(path_scores, load_feature_scores, features)
        st.markdown("Average precision per feature norm")
        st.write(df_scores)

    table = [st.columns(num_cols) for _ in range(num_rows)]
    table = list(zip(*table))

    predictions = [read_json(get_path_preds(f)) for f in features_selected]

    for i in range(num_cols):
        show1(
            table[i],
            features_selected[i],
            feature_to_concepts,
            predictions[i],
            dataset,
        )


if __name__ == "__main__":
    main()
