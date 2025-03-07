import pdb
from pathlib import Path
from torchvision.datasets import ImageNet

import streamlit as st
import pandas as pd

ROOT = Path("/mnt/private-share/speechDatabases/imagenet12")
dataset = ImageNet(root=ROOT, split="val")

df = pd.read_csv("data/norms/mcrae/CONCS_FEATS_concstats_brm.txt", sep="\t")
concepts = df["Concept"].unique()
concepts = list(concepts)

def get_matching_classes(concept):
    result = [
        (i, cs) for i, cs in enumerate(dataset.classes)
        if any(concept == c or concept in c.split(" ") for c in cs)
    ]
    if len(result):
        labels, names = zip(*result) 
    else:
        labels, names = [], []
    labels = ", ".join(map(str, labels))
    names = ", ".join(" ".join(n) for n in names)
    return {
        "labels": labels,
        "names": names,
    }


df2 = [
    {
        "concept": concept,
        **get_matching_classes(concept),
    }
    for concept in concepts
]
df2 = pd.DataFrame(df2)
# st.write(df2)

SEP = ", "
class_names = [str(i) + " · " + SEP.join(cs) for i, cs in enumerate(dataset.classes)]

with st.sidebar:
    concept_ = st.selectbox("Select class", concepts)

label_selected = concepts.index(concept_)

idxs = df2["concept"] == concept_
for labels in df2[idxs]["labels"].values:
    for label in labels.split(SEP):
        label = int(label)
        class_name = class_names[label]
        st.markdown("## " + class_name)
        paths = [path for path, label_ in dataset.samples if label_ == label]
        for path in paths[:5]:
            st.image(path)
        st.markdown("---")
