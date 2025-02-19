from itertools import groupby

import streamlit as st

from probing_norms.data import DATASETS, DIR_THINGS, load_things_concept_mapping
from probing_norms.predict import NORMS_LOADERS
from probing_norms.utils import multimap
from probing_norms.extract_features_text import HFModelContextual

dataset = DATASETS["things"]()
concept_image = [
    (dataset.label_to_class[dataset.labels[i]], dataset.image_files[i])
    for i in range(len(dataset))
]
concept_to_images = multimap(concept_image)
norms_loader = NORMS_LOADERS["mcrae-mapped"]()
feature_to_concepts, _, _ = norms_loader()
concepts = norms_loader.load_concepts()
contexts = HFModelContextual.load_context("gpt4o_concept")
concept_to_word_and_category = load_things_concept_mapping()
concept_feature = [
    (concept, feature)
    for feature, concepts in feature_to_concepts.items()
    for concept in concepts
]
concept_to_features = multimap(set(concept_feature))

def get_image_path(concept):
    return DIR_THINGS / "object_images_CC0" / (concept + ".jpg")

def prepare(concept):
    *chars, n = concept
    word = "".join(chars)
    return word, int(n), concept

def itemize(xs):
    return "\n".join("1. {}".format(x) for x in xs)

def code(x):
    return f"```\n{x}\n```"

homonyms = [prepare(concept) for concept in concepts if concept[-1].isdigit()]
groups = groupby(homonyms, key=lambda x: x[0])

st.set_page_config(layout="wide")

for k, group in groups:
    group = list(group)
    st.write(k)
    cols = st.columns(4)
    for i, (word, n, concept) in enumerate(group):
        cols[i].markdown("{} · {}".format(n, concept_to_word_and_category[concept]))
        cols_ = cols[i].columns(3)
        for j, col in enumerate(cols_):
            path = dataset.get_image_path(concept_to_images[concept][j])
            col.image(path)
        cols[i].markdown(itemize(contexts[concept][:3]))
        cols[i].markdown(code("\n".join(sorted(concept_to_features[concept]))))
    st.markdown("---")