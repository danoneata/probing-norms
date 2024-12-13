import pdb
import random

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


def load_embeddings(dataset_name, feature_type):
    path = "output/features-image/{}-{}.npz".format(dataset_name, feature_type)
    output = np.load(path, allow_pickle=True)
    return output["X"]


def predict1(X, y, split, dataset):
    idxs_tr = split.tr_idxs
    idxs_te = split.te_idxs

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


def predict_splits(X, y, splits, dataset):
    return [
        {
            "preds": predict1(X, y, split, dataset),
            "split": split.metadata,
        }
        for split in tqdm(splits, leave=False)
    ]


@dataclass
class Split:
    tr_idxs: np.ndarray
    te_idxs: np.ndarray
    metadata: dict


def get_train_test_split_iid_fixed(
    dataset,
    features,
    feature_to_concepts,
) -> Dict[str, List[Split]]:
    idxs = np.arange(len(dataset))
    idxss = train_test_split(idxs, test_size=0.2, random_state=42)
    get_f = lambda f: [Split(*idxss, metadata=dict(feature=f))]
    return {feature: get_f(feature) for feature in features}


def get_train_test_split_leave_one_out(
    dataset,
    features,
    feature_to_concepts,
) -> Dict[str, List[Split]]:
    def get_c(concept):
        idxs = list(range(len(dataset)))
        label = dataset.class_to_label[concept]
        tr_idxs = [i for i in idxs if dataset.labels[i] != label]
        te_idxs = [i for i in idxs if dataset.labels[i] == label]
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
@click.option("--feature-type", "feature_type", type=str, required=True)
@click.option("--norms-model", "norms_model", type=str, required=True)
@click.option("--split-type", "split_type", type=str, required=True)
def main(feature_type, norms_model, split_type):
    dataset_name = "things"
    # feature_type = "pali-gemma-224"
    dataset = DATASETS[dataset_name]()
    embeddings = load_embeddings(dataset_name, feature_type)

    norms_priming = "mcrae"
    # norms_model = "chatgpt-gpt3.5-turbo"
    # norms_model = "gpt3-davinci"
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
    features_selected = random.sample(features_selected, 64)
    print(sorted(features_selected))

    def get_path(feature):
        feature_norm = "{}_{}_{}".format(norms_priming, norms_model, norms_num_runs)
        feature_id = feature_to_id[feature]
        return "output/linear-probe-predictions/{}/{}-{}-{}-{}.json".format(
            split_type,
            dataset_name,
            feature_type,
            feature_norm,
            feature_id,
        )

    def get_labels(feature):
        concepts = feature_to_concepts[feature]
        labels = [
            int(dataset.label_to_class[label] in concepts) for label in dataset.labels
        ]
        return np.array(labels)

    splits = GET_TRAIN_TEST_SPLIT[split_type](
        dataset,
        features_selected,
        feature_to_concepts,
    )

    def process_feature(feature):
        cache_json(
            get_path(feature),
            predict_splits,
            embeddings,
            get_labels(feature),
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
