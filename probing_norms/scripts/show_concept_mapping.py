import pandas as pd

from probing_norms.data import (
    load_mcrae_feature_norms,
    load_gpt3_feature_norms,
    get_feature_to_concepts,
    DATASETS,
)
from probing_norms.utils import multimap
from toolz import partition_all

import streamlit as st

st.set_page_config(layout="wide")

norm1 = load_mcrae_feature_norms()
norm2 = load_gpt3_feature_norms()

concepts1 = sorted(set(concept for concept, _ in norm1))
concepts2 = sorted(set(concept for concept, _ in norm2))

norm1_inv = [(s, f) for f, s in norm1]
concept1_to_features = get_feature_to_concepts(norm1_inv)

# print(concepts1[:10])
# print(concepts2[:10])

dataset = DATASETS["things"]()
concept_image = [
    (dataset.label_to_class[dataset.labels[i]], dataset.image_files[i])
    for i in range(len(dataset))
]
concept_to_images = multimap(concept_image)

concepts_mcrae_not_in_things = [concept for concept in concepts1 if concept not in concepts2]
concepts_things_not_in_mcrae = [concept for concept in concepts2 if concept not in concepts1]

col1, col2 = st.columns(2)

concept1 = col1.selectbox("Concepts in McRae", concepts1)
concept2 = col2.selectbox("Concepts in THINGS", concepts2)

with col1:
    flag_str = "✓" if concept1 in concepts2 else "✗"
    st.markdown("Concept has one-to-one mapping: {}".format(flag_str))
    st.write(concept1_to_features[concept1])

with col2:
    flag_str = "✓" if concept2 in concepts1 else "✗"
    st.markdown("Concept has one-to-one mapping: {}".format(flag_str))
    # num_images_to_show = 10
    num_cols = 5
    groups = partition_all(num_cols, concept_to_images[concept2])
    for group in groups:
        cols = st.columns(num_cols)
        images = list(group)
        for j in range(num_cols):
            if j >= len(images):
                break
            path = dataset.get_image_path(images[j])
            cols[j].image(path)


# with open("output/concepts-mcrae-not-in-things.txt", "w") as f:
#     for concept in concepts1:
#         if concept not in concepts2:
#             f.write("{}\n".format(concept))

# with open("output/concepts-things-not-in-mcrae.txt", "w") as f:
#     for concept in concepts2:
#         if concept not in concepts1:
#             f.write("{}\n".format(concept))