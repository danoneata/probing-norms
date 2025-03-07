import base64
import pdb
import random

from io import BytesIO
from pathlib import Path

import altair as alt
import h5py
import numpy as np
import pandas as pd
import streamlit as st

from toolz import concat
from matplotlib import pyplot as plt
from torchvision.datasets import ImageNet

from sklearn import manifold

from utils import cache_df, multimap

st.set_page_config(layout="wide")


ROOT = Path("/mnt/private-share/speechDatabases/imagenet12")
SPLIT = "val"
dataset = ImageNet(root=ROOT, split=SPLIT)


def load_df(min_num_features):
    columns = ["Concept", "Feature", "BR_Label"]
    df = pd.read_csv("data/norms/CONCS_FEATS_concstats_brm.txt", sep="\t")
    df = df[columns]

    columns = ["Concept", "ImageNet IDs"]
    df_concepts_imagenet = pd.read_csv("data/McRae norms - concepts _ imagenet.csv")
    df_concepts_imagenet = df_concepts_imagenet[columns]

    # Select columns that have non empty ImageNet IDs
    idxs = df_concepts_imagenet["ImageNet IDs"].notna()
    df_concepts_imagenet = df_concepts_imagenet[idxs]

    # Join the two dataframes on the `Concept`` column
    df = df.merge(df_concepts_imagenet, on="Concept")

    df_feature_counts = df.groupby("Feature").size()
    df_feature_counts = df_feature_counts.reset_index()
    df_feature_counts = df_feature_counts.rename(columns={0: "Feature count"})
    df = df.merge(df_feature_counts, on="Feature")

    # st.write(df.head())
    idxs = df["Feature count"] >= min_num_features
    df = df[idxs]

    return df


def load_image_features(selected_labels, num_samples_per_class):
    path_hdf5 = "output/features-image/imagenet12-val-dino-resnet50.h5"
    labels_and_indices = [(label, i) for i, (_, label) in enumerate(dataset.samples)]
    label_to_index = multimap(labels_and_indices)
    label_to_index = {
        label: indexes[:num_samples_per_class]
        for label, indexes in label_to_index.items()
        if label in selected_labels
    }
    labels_and_indices = [
        (label, i) for label, indexes in label_to_index.items() for i in indexes
    ]
    labels, indices = zip(*labels_and_indices)
    with h5py.File(path_hdf5, "r") as f:
        features = [f[str(i)]["feature"][()] for i in indices]
        features = np.stack(features)
    return features, labels, indices


def image_to_base64(i):
    size = 64
    img, _ = dataset[i]
    img = img.resize((size, size))
    output = BytesIO()
    img.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


def load_image_features_2d(df):
    imagenet_ids = concat(df["ImageNet IDs"].str.split(", "))
    imagenet_ids = set(imagenet_ids)
    imagenet_ids = map(int, imagenet_ids)
    imagenet_ids = sorted(imagenet_ids)

    tsne = manifold.TSNE(n_components=2, random_state=0)
    features, labels, indices = load_image_features(
        imagenet_ids,
        num_samples_per_class=5,
    )
    features_2d = tsne.fit_transform(features)
    df = pd.DataFrame(features_2d, columns=["x", "y"])

    df["label"] = labels
    df["indices"] = indices
    df["image"] = df["indices"].apply(image_to_base64)

    return df


def main():
    min_num_features = 5
    df = load_df(min_num_features)
    df_images = cache_df(
        "output/plots/image-features-tsne.pkl", load_image_features_2d, df
    )

    br_labels_and_features = df[["BR_Label", "Feature"]]
    br_labels_and_features = br_labels_and_features.drop_duplicates()
    br_labels_and_features = br_labels_and_features.values
    br_label_to_features = multimap(br_labels_and_features)

    br_labels = list(br_label_to_features.keys())

    with st.sidebar:
        br_label = st.selectbox("BR Label:", br_labels)
        feature = st.selectbox("Feature:", br_label_to_features[br_label])

        pick_random = st.button("Pick random")

        st.markdown("---")
        st.markdown(
            """
            Note: This is a subset of the original list from McRae because:

            1. I have selected only those concepts had an ImageNet equivalent;
            2. I have selected only those features that appear for at least {} concepts.
            """.format(
                min_num_features
            )
        )

    if pick_random:
        br_label = random.choice(br_labels)
        feature = random.choice(br_label_to_features[br_label])

    def fmt_feature(feature):
        return feature.replace("_", " ")

    def itemize(ss):
        return "\n".join("- " + s for s in ss)

    def fmt_concept_imagenet_id(concept_and_imagenet_id):
        concept, imagenet_ids = concept_and_imagenet_id
        imagenet_ids = [int(i) for i in imagenet_ids.split(", ")]
        imagenet_ids_with_names = [
            "`{}` ({})".format(i, ", ".join(dataset.classes[i])) for i in imagenet_ids
        ]
        imagenet_ids_with_names = ", ".join(imagenet_ids_with_names)
        return "{} → {}".format(concept, imagenet_ids_with_names)

    idxs = (df["BR_Label"] == br_label) & (df["Feature"] == feature)
    concepts_and_imagenet_ids = df[idxs][["Concept", "ImageNet IDs"]].values

    # Expand `ImageNet IDs` into multiple rows
    df["ImageNet IDs"] = df["ImageNet IDs"].str.split(", ")
    df = df.explode("ImageNet IDs")
    df = df.reset_index(drop=True)
    df["ImageNet IDs"] = df["ImageNet IDs"].astype(int)
    idxs = (df["BR_Label"] == br_label) & (df["Feature"] == feature)
    selected_imagenet_ids = df[idxs]["ImageNet IDs"].unique()

    df_images["label1"] = df_images["label"].apply(
        lambda x: x if x in selected_imagenet_ids else None
    )
    df_images["class-name"] = df_images["label"].apply(
        lambda x: ", ".join(dataset.classes[x])
    )

    # # Add `Concept` to `df_images`
    # imagenet_id_to_concepts = multimap(df[["ImageNet IDs", "Concept"]].values)
    # df_images["Concept"] = df_images["labels"].apply(lambda x: imagenet_id_to_concepts[x])

    # fmt: off
    st.markdown("### Feature: `{}` · BR Label: `{}`".format(feature, br_label))
    st.markdown("Concepts that have the `{}` feature and their corresponding ImageNet classes (ID and name):".format(feature))
    st.markdown(itemize(map(fmt_concept_imagenet_id, concepts_and_imagenet_ids)))
    # fmt: on

    # num_imagenet_ids = len(selected_imagenet_ids)
    idxs = df_images["label1"].notna()
    fig1 = (
        alt.Chart(df_images[~idxs])
        .mark_circle(size=75)
        .encode(
            x="x",
            y="y",
            color=alt.value("gray"),
            tooltip=["image", "label", "class-name"],
        )
    )
    fig2 = (
        alt.Chart(df_images[idxs])
        .mark_circle(size=75)
        .encode(
            x="x",
            y="y",
            color=alt.Color("label1:N", scale=alt.Scale(scheme="tableau20")),
            tooltip=["image", "label", "class-name"],
        )
    )
    fig = fig1 + fig2
    fig = fig.properties(
            width=800,
            height=800,
        )

    st.altair_chart(fig)


if __name__ == "__main__":
    main()
