import pdb
import numpy as np
from sentence_transformers import SentenceTransformer
from probing_norms.data import (
    load_gpt3_feature_norms,
    load_mcrae_feature_norms,
    get_feature_to_concepts,
)
from probing_norms.utils import cache_np


def normalize_text(text):
    SEP = "_-_"
    if SEP in text:
        prefix, text = text.split(SEP)
        assert prefix in {"beh", "inbeh", "eg", "has_units", "worn_by_men"}, prefix
        if prefix == "eg":
            text = "example: " + text
        elif prefix in {"has_units", "worn_by_men"}:
            text = text + " " + prefix
        else:
            pass

    if text.startswith("a_") or text.startswith("an_"):
        text = "is_" + text

    text = text.replace("_", " ").strip()
    return text


model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embeddings(norms):
    return model.encode(norms)


def get_similarities(norms1, norms2):
    embs1 = get_embeddings(norms1)
    embs2 = get_embeddings(norms2)
    sims = model.similarity(embs1, embs2)
    return sims.numpy()


class Norms:
    def __init__(self, concepts_norms):
        self.concepts_norms = concepts_norms
        self.norm_to_concepts = get_feature_to_concepts(concepts_norms)
        self.norms = sorted(set(norm for _, norm in concepts_norms))
        self.concepts = sorted(set(concept for concept, _ in concepts_norms))

    def __getitem__(self, i):
        return self.norms[i]

    def __len__(self):
        return len(self.norms)

    def __iter__(self):
        return iter(self.norms)


norms_mcrae = Norms(load_mcrae_feature_norms())
norms_gpt35 = Norms(load_gpt3_feature_norms())

common_concepts = set(norms_mcrae.concepts) & set(norms_gpt35.concepts)

path = "output/sims-norms-mcrae-gpt35.npy"
norms_mcrae_1 = list(map(normalize_text, norms_mcrae))
similarities = cache_np(path, get_similarities, norms_mcrae_1, norms_gpt35)

SEP = "\t"
has_match_total = 0
data = []

with open("output/map-mcrae-to-gpt35.tsv", "w") as f:

    for norm_mcrae, sims in zip(norms_mcrae, similarities):
        idxs, *_ = np.where(sims >= 0.9)
        selected_norms_str = ", ".join(
            "{} ({:.2f})".format(norms_gpt35[i], sims[i]) for i in idxs
        )
        concepts_mcrae = norms_mcrae.norm_to_concepts[norm_mcrae]
        concepts_mcrae_common = set(concepts_mcrae) & common_concepts

        concepts_gpt35 = [
            concept
            for i in idxs
            for concept in norms_gpt35.norm_to_concepts[norms_gpt35[i]]
        ]
        concepts_gpt35_common = set(concepts_gpt35) & common_concepts

        has_match = len(idxs) > 0
        has_match_total += int(has_match)

        # if has_match:
        #     print(
        #         "{} · {} → {} · {}".format(
        #             norm_mcrae, len(concepts_mcrae), selected_norms_str, len(concepts_gpt35)
        #         )
        #     )

        out = SEP.join(
            [
                norm_mcrae,
                selected_norms_str,
                str(len(concepts_mcrae_common)),
                str(len(concepts_gpt35_common)),
                str(len(concepts_mcrae)),
                str(len(concepts_gpt35)),
                ", ".join(concepts_gpt35_common - concepts_mcrae_common),
                ", ".join(concepts_mcrae_common - concepts_gpt35_common),
            ]
        )
        f.write(out + "\n")

        datum = {
            "norm": norm_mcrae,
            "concepts": concepts_gpt35 + list(concepts_mcrae_common - concepts_gpt35_common),
            "norms-gpt35": [norms_gpt35[i] for i in idxs],
        }
        data.append(datum)

print(has_match_total)
print(len(norms_mcrae))

import json
with open("output/map-mcrae-to-gpt35.json", "w") as f:
    json.dump(data, f, indent=4)