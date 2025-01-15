import json
import pdb
import random
import os
import pickle

from dataclasses import dataclass
from typing import Dict, List

import click
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from probing_norms.constants import NUM_MIN_CONCEPTS
from probing_norms.data import (
    DATASETS,
    load_gpt3_feature_norms,
    get_feature_to_concepts,
)
from probing_norms.utils import cache_json
from multiprocess import Pool


def aggregate_by_labels(embeddings, labels):
    unique_labels = np.unique(labels)
    agg_embeddings = np.zeros((len(unique_labels), embeddings.shape[1]))
    for i, label in enumerate(unique_labels):
        idxs = labels == label
        agg_embeddings[i] = embeddings[idxs].mean(axis=0)
    return agg_embeddings, unique_labels


AGGREGATE_EMBEDDINGS = {
    "instance": lambda x, y: (x, y),
    "concept": aggregate_by_labels,
}


def load_embeddings(dataset_name, feature_type, embeddings_level):
    path = "output/features-image/{}-{}.npz".format(dataset_name, feature_type)
    output = np.load(path, allow_pickle=True)
    embeddings = output["X"]
    labels = output["y"]
    embeddings, labels = AGGREGATE_EMBEDDINGS[embeddings_level](embeddings, labels)
    return embeddings, labels


def predict1(X, y, split, dataset):
    idxs_tr = split.tr_idxs
    idxs_te = split.te_idxs

    X_tr, y_tr = X[idxs_tr], y[idxs_tr]
    X_te, y_te = X[idxs_te], y[idxs_te]

    clf = LogisticRegression(max_iter=1_000, verbose=False)
    clf.fit(X_tr, y_tr)

    y_pr = clf.predict_proba(X_te)[:, 1]

    preds = [
        {
            "i": i.item(),
            "name": dataset.image_files[i],
            "label": dataset.labels[i],
            "pred": p.item(),
            "true": t.item(),
        }
        for i, p, t in zip(idxs_te, y_pr, y_te)
    ]
    return {
        "preds": preds,
        "clf": clf,
    }


def predict_splits(X, y, splits, dataset):
    return [
        {
            "split": split.metadata,
            **predict1(X, y, split, dataset),
        }
        for split in tqdm(splits, leave=False)
    ]


@dataclass
class Split:
    tr_idxs: np.ndarray
    te_idxs: np.ndarray
    metadata: dict


def get_train_test_split_iid_fixed(
    labels,
    features,
    **_,
) -> Dict[str, List[Split]]:
    idxs = np.arange(len(labels))
    idxss = train_test_split(idxs, test_size=0.2, random_state=42)
    get_f = lambda f: [Split(*idxss, metadata=dict(feature=f))]
    return {feature: get_f(feature) for feature in features}


def get_train_test_split_leave_one_out(
    labels,
    features,
    *,
    feature_to_concepts,
    class_to_label,
) -> Dict[str, List[Split]]:
    def get_c(concept):
        idxs = list(range(len(labels)))
        label = class_to_label[concept]
        tr_idxs = [i for i in idxs if labels[i] != label]
        te_idxs = [i for i in idxs if labels[i] == label]
        return {
            "tr_idxs": np.array(tr_idxs),
            "te_idxs": np.array(te_idxs),
        }

    def get_f(feature):
        concepts = feature_to_concepts[feature]
        return [
            Split(
                **get_c(concept),
                metadata={"feature": feature, "test-concept": concept},
            )
            for concept in concepts
        ]

    return {feature: get_f(feature) for feature in features}


GET_TRAIN_TEST_SPLIT = {
    "iid-fixed": get_train_test_split_iid_fixed,
    "leave-one-concept-out": get_train_test_split_leave_one_out,
}


@click.command()
@click.option(
    "--embeddings-level",
    "embeddings_level",
    type=click.Choice(AGGREGATE_EMBEDDINGS.keys()),
    required=True,
)
@click.option("--feature-type", "feature_type", type=str, required=True)
@click.option("--norms-model", "norms_model", type=str, required=True)
@click.option("--split-type", "split_type", type=str, required=True)
def main(embeddings_level, feature_type, norms_model, split_type):
    dataset_name = "things"
    # feature_type = "pali-gemma-224"
    dataset = DATASETS[dataset_name]()
    embeddings, labels = load_embeddings(dataset_name, feature_type, embeddings_level)

    norms_priming = "mcrae"
    norms_num_runs = 30
    concept_feature = load_gpt3_feature_norms(
        priming=norms_priming,
        model=norms_model,
        num_runs=norms_num_runs,
    )
    feature_to_concepts = get_feature_to_concepts(concept_feature)
    features = sorted(feature_to_concepts.keys())
    feature_to_id = {feature: i for i, feature in enumerate(features)}

    features_selected = [
        feature
        for feature in features
        if len(feature_to_concepts[feature]) >= NUM_MIN_CONCEPTS
    ]

    random.seed(42)
    features_selected_1 = random.sample(features_selected, 64)
    features_selected_2 = random.sample(features_selected, 256 - 64)
    features_selected = sorted(set(features_selected_1 + features_selected_2))

    def get_path(feature):
        feature_norm = "{}_{}_{}".format(norms_priming, norms_model, norms_num_runs)
        feature_id = feature_to_id[feature]
        return "output/linear-probe-predictions/{}/{}/{}-{}-{}-{}".format(
            embeddings_level,
            split_type,
            dataset_name,
            feature_type,
            feature_norm,
            feature_id,
        )

    def get_binary_labels(feature):
        concepts_str = feature_to_concepts[feature]
        concepts_num = [dataset.class_to_label[c] for c in concepts_str]
        concepts_num = set(concepts_num)
        binary_labels = [label in concepts_num for label in labels]
        return np.array(binary_labels).astype(int)

    splits = GET_TRAIN_TEST_SPLIT[split_type](
        labels,
        features_selected,
        feature_to_concepts=feature_to_concepts,
        class_to_label=dataset.class_to_label,
    )

    def cache_clf_and_preds(path, func, *args):
        path_json = path + ".json"
        path_pkl = path + ".pkl"
        paths_exist = os.path.exists(path_json) and os.path.exists(path_pkl)
        if not paths_exist:
            results = func(*args)

            data1 = [{k: r[k] for k in ("split", "preds")} for r in results]
            with open(path_json, "w") as f:
                json.dump(data1, f, indent=4)

            data2 = [{k: r[k] for k in ("split", "clf")} for r in results]
            with open(path_pkl, "wb") as f:
                pickle.dump(data2, f)

    def process_feature(feature):
        cache_clf_and_preds(
            get_path(feature),
            predict_splits,
            embeddings,
            get_binary_labels(feature),
            splits[feature],
            dataset,
        )

    # with Pool(16) as pool:
    #     processes = pool.imap(process_feature, features_selected)
    #     total = len(features_selected)
    #     for _ in tqdm(processes, total=total):
    #         pass

    for f in tqdm(features_selected):
        process_feature(f)


if __name__ == "__main__":
    main()
