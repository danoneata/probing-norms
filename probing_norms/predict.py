import pdb

import click
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from probing_norms.data import (
    DATASETS,
    load_gpt3_feature_norms,
    get_feature_to_concepts,
)
from probing_norms.utils import cache_json


def load_embeddings(dataset_name, feature_type):
    path = "output/features-image/{}-{}.npz".format(dataset_name, feature_type)
    output = np.load(path, allow_pickle=True)
    return output["X"]


def predict1(embeddings, labels, idxss, dataset):
    X = embeddings
    y = np.array(labels)
    idxs_tr, idxs_te = idxss

    X_tr, y_tr = X[idxs_tr], y[idxs_tr]
    X_te, y_te = X[idxs_te], y[idxs_te]

    clf = LogisticRegression(max_iter=1_000, verbose=False)
    clf.fit(X_tr, y_tr)

    y_pr = clf.predict_proba(X_te)[:, 1]

    return [
        {
            "i": i.item(),
            "name": dataset.image_files[i],
            "label": dataset.labels[i],
            "pred": p.item(),
            "true": t.item(),
        }
        for i, p, t in zip(idxs_te, y_pr, y_te)
    ]


def main():
    dataset_name = "things"
    feature_type = "pali-gemma-224"
    dataset = DATASETS[dataset_name]()
    embeddings = load_embeddings(dataset_name, feature_type)

    idxs = np.arange(len(dataset))
    idxss = train_test_split(idxs, test_size=0.2, random_state=42)

    priming = "mcrae"
    # model = "chatgpt-gpt3.5-turbo"
    model = "gpt3-davinci"
    num_runs = 30
    concept_feature = load_gpt3_feature_norms(
        priming=priming,
        model=model,
        num_runs=num_runs,
    )
    feature_to_concepts = get_feature_to_concepts(concept_feature)
    features = sorted(feature_to_concepts.keys())
    feature_to_id = {feature: i for i, feature in enumerate(features)}

    def get_path(feature):
        feature_norm = "{}_{}_{}".format(priming, model, num_runs)
        feature_id = feature_to_id[feature]
        return "output/linear-probe-predictions/{}-{}-{}-{}.json".format(
            dataset_name,
            feature_type,
            feature_norm,
            feature_id,
        )

    def get_labels(feature):
        concepts = feature_to_concepts[feature]
        return [
            int(dataset.label_to_class[label] in concepts) for label in dataset.labels
        ]

    for feature in tqdm(features):
        concepts = feature_to_concepts[feature]
        if len(concepts) < 15:
            continue
        cache_json(
            get_path(feature),
            predict1,
            embeddings,
            get_labels(feature),
            idxss,
            dataset,
        )


if __name__ == "__main__":
    main()
