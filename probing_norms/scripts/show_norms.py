import pdb
import random

import pandas as pd
import streamlit as st

from probing_norms.data import load_gpt3_feature_norms, get_feature_to_concepts


def show_norms_mcrae():
    df = pd.read_csv("data/norms/mcrae/CONCS_FEATS_concstats_brm.txt", sep="\t")
    df = df.groupby("Concept")["Feature"].agg(", ".join)
    st.write(df)


def show_norms_gpt3():
    concept_feature = load_gpt3_feature_norms()
    feature_groups = get_feature_to_concepts(concept_feature).items()
    feature_groups = [
        (feature, sorted(group))
        for feature, group in feature_groups
        if len(group) >= 10
    ]
    # feature_groups = sorted(feature_groups, key=lambda x: len(x[1]), reverse=True)

    n_features = len(feature_groups)
    st.markdown("Number of features: {}".format(n_features))
    st.write("---")

    feature_groups = random.sample(feature_groups, 10)

    for feature, group in feature_groups:
        st.markdown(
            "{} ({}) → {}".format(
                feature,
                len(group),
                ", ".join([concept for concept, _ in group]),
            )
        )


if __name__ == "__main__":
    show_norms_gpt3()