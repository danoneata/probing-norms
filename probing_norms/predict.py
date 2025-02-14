import json
import pdb
import random
import os
import pickle

from dataclasses import dataclass
from functools import partial
from typing import Dict, List

import click
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer

from tqdm import tqdm

from probing_norms.constants import NUM_MIN_CONCEPTS
from probing_norms.data import (
    DATASETS,
    filter_by_things_concepts,
    load_features_metadata,
    load_binder_feature_norms,
    load_binder_feature_norms_median,
    load_mcrae_feature_norms,
    get_feature_to_concepts,
)
from probing_norms.utils import cache_json, read_file
from probing_norms.extract_features_image import (
    FEATURE_EXTRACTORS as FEATURE_EXTRACTORS_IMAGE,
)
from probing_norms.extract_features_text import (
    FEATURE_EXTRACTORS as FEATURE_EXTRACTORS_TEXT,
    MAPPING_TYPES,
)


NORMS_PRIMING = "mcrae"
NORMS_NUM_RUNS = 30


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


FEATURE_TYPE_TO_MODALITY = {
    **{k: "image" for k in FEATURE_EXTRACTORS_IMAGE.keys()},
    **{
        k + "-" + m: "text"
        for k in FEATURE_EXTRACTORS_TEXT.keys()
        for m in MAPPING_TYPES
    },
}


def load_embeddings(dataset_name, feature_type, embeddings_level):
    modality = FEATURE_TYPE_TO_MODALITY[feature_type]
    path = "output/features-{}/{}-{}.npz".format(modality, dataset_name, feature_type)
    output = np.load(path, allow_pickle=True)
    embeddings = output["X"]
    labels = output["y"].astype(np.int32)
    if modality == "image":
        embeddings, labels = AGGREGATE_EMBEDDINGS[embeddings_level](embeddings, labels)
    return embeddings, labels


def predict1(make_classifier, X, y, split):
    idxs_tr = split.tr_idxs
    idxs_te = split.te_idxs

    X_tr, y_tr = X[idxs_tr], y[idxs_tr]
    X_te, y_te = X[idxs_te], y[idxs_te]

    clf = make_classifier()
    clf.fit(X_tr, y_tr)

    y_pr = clf.predict_proba(X_te)[:, 1]

    preds = [
        {
            "i": i.item(),
            # "name": dataset.image_files[i],
            # "label": dataset.labels[i],
            "pred": p.item(),
            "true": t.item(),
        }
        for i, p, t in zip(idxs_te, y_pr, y_te)
    ]
    return {
        "preds": preds,
        "clf": clf,
    }


def predict_splits(make_classifier, X, y, splits):
    return [
        {
            "split": split.metadata,
            **predict1(make_classifier, X, y, split),
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


def get_binary_labels(labels, feature, feature_to_concepts, class_to_label):
    concepts_str = feature_to_concepts[feature]
    concepts_num = [class_to_label[c] for c in concepts_str]
    concepts_num = set(concepts_num)
    binary_labels = [label in concepts_num for label in labels]
    return np.array(binary_labels).astype(int)


def get_train_test_split_k_fold(
    labels,
    features,
    *,
    feature_to_concepts,
    class_to_label,
    n_splits=5,
    n_repeats=2,
):
    def get_f(feature):
        binary_labels = get_binary_labels(
            labels,
            feature,
            feature_to_concepts,
            class_to_label,
        )
        rskf = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=42,
        )
        return [
            Split(
                *idxss,
                metadata={"feature": feature},
            )
            for idxss in rskf.split(binary_labels, binary_labels)
        ]

    return {feature: get_f(feature) for feature in features}


GET_TRAIN_TEST_SPLIT = {
    "iid-fixed": get_train_test_split_iid_fixed,
    "leave-one-concept-out": get_train_test_split_leave_one_out,
    "repeated-k-fold": get_train_test_split_k_fold,
}


def sample_features(feature_to_concepts):
    features_selected = [
        feature
        for feature, concepts in feature_to_concepts.items()
        if len(concepts) >= NUM_MIN_CONCEPTS
    ]

    random.seed(42)
    features_selected_1 = random.sample(features_selected, 64)
    features_selected_2 = random.sample(features_selected, 256 - 64)
    features_selected = sorted(set(features_selected_1 + features_selected_2))
    return features_selected


class NormsLoader:
    def __call__(self):
        raise NotImplementedError

    def get_suffix(self) -> str:
        raise NotImplementedError

    def load_concepts(self):
        return read_file("data/concepts-things.txt")


class GPT3NormsLoader(NormsLoader):
    def __init__(self, norms_model):
        self.norms_model = norms_model

    def __call__(self):
        feature_to_concepts, feature_to_id = load_features_metadata(
            priming=NORMS_PRIMING,
            model=self.norms_model,
            num_runs=NORMS_NUM_RUNS,
        )
        features_selected = sample_features(feature_to_concepts)
        return feature_to_concepts, feature_to_id, features_selected

    def get_suffix(self):
        return "{}_{}_{}".format(NORMS_PRIMING, self.norms_model, NORMS_NUM_RUNS)


class McRaeNormsLoader(NormsLoader):
    def __init__(self):
        self.model = "mcrae"

    def __call__(self, *, num_min_concepts=NUM_MIN_CONCEPTS):
        concept_feature = load_mcrae_feature_norms()
        feature_to_concepts = get_feature_to_concepts(concept_feature)

        features = sorted(feature_to_concepts.keys())
        feature_to_id = {feature: i for i, feature in enumerate(features)}

        features_selected = [
            feature
            for feature, concepts in feature_to_concepts.items()
            if len(concepts) >= num_min_concepts
        ]
        return feature_to_concepts, feature_to_id, features_selected

    def get_suffix(self):
        return str(self.model)


class McRaeMappedNormsLoader(NormsLoader):
    def __init__(self):
        self.model = "mcrae-to-gpt35"

    def __call__(self, *, num_min_concepts=10):
        with open("data/mcrae++.json".format(self.model)) as f:
            data = json.load(f)

        feature_to_concepts = {d["norm"]: d["concepts"] for d in data}
        feature_to_id = {d["norm"]: i for i, d in enumerate(data)}
        features_selected = [
            norm
            for norm, concepts in feature_to_concepts.items()
            if len(concepts) >= num_min_concepts
        ]
        return feature_to_concepts, feature_to_id, features_selected

    def get_suffix(self):
        return str(self.model)


class BinderNormsLoader(NormsLoader):
    def __init__(self, thresh):
        self.model = "binder"
        self.thresh = thresh
        if isinstance(thresh, str):
            assert thresh == "median"
            self.load_feature_norms = load_binder_feature_norms_median
        elif isinstance(thresh, int):
            self.load_feature_norms = partial(load_binder_feature_norms, thresh=thresh)
        else:
            assert False

    def load_concepts(self):
        df = pd.read_excel("data/binder-norms.xlsx")
        df = filter_by_things_concepts(df)
        return df["Word"].tolist()

    def __call__(self):
        concept_feature = self.load_feature_norms()
        feature_to_concepts = get_feature_to_concepts(concept_feature)

        features = sorted(feature_to_concepts.keys())
        feature_to_id = {feature: i for i, feature in enumerate(features)}

        features_selected = [
            norm for norm, concepts in feature_to_concepts.items() if len(concepts) >= 5
        ]
        # features_selected = features

        return feature_to_concepts, feature_to_id, features_selected

    def get_suffix(self):
        return "{}-{}".format(self.model, self.thresh)


NORMS_LOADERS = {
    "generated-gpt35": partial(GPT3NormsLoader, norms_model="chatgpt-gpt3.5-turbo"),
    "mcrae": McRaeNormsLoader,
    "mcrae-mapped": McRaeMappedNormsLoader,
    "binder-3": partial(BinderNormsLoader, thresh=3),
    "binder-4": partial(BinderNormsLoader, thresh=4),
    "binder-5": partial(BinderNormsLoader, thresh=5),
    "binder-median": partial(BinderNormsLoader, thresh="median"),
}


CLASSIFIERS = {
    "linear-probe": partial(
        LogisticRegression,
        penalty=None,
        max_iter=1_000,
    ),
    "linear-probe-std": lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1_000),
    ),
    "knn-3": lambda: make_pipeline(
        Normalizer(norm="l2"),
        KNeighborsClassifier(n_neighbors=3),
    ),
    "mlp": lambda: make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(1024, 512)),
    ),
}


@click.command()
@click.option(
    "-c",
    "--classifier-type",
    "classifier_type",
    type=click.Choice(CLASSIFIERS),
    required=True,
)
@click.option(
    "--embeddings-level",
    "embeddings_level",
    type=click.Choice(AGGREGATE_EMBEDDINGS),
    required=True,
)
@click.option(
    "-f",
    "--feature-type",
    "feature_type",
    type=click.Choice(FEATURE_TYPE_TO_MODALITY),
    required=True,
)
@click.option(
    "--norms-type",
    "norms_type",
    type=click.Choice(NORMS_LOADERS),
    required=True,
)
@click.option(
    "--split-type",
    "split_type",
    type=click.Choice(GET_TRAIN_TEST_SPLIT),
    required=True,
)
def main(classifier_type, embeddings_level, feature_type, norms_type, split_type):
    dataset_name = "things"
    dataset = DATASETS[dataset_name]()

    norm_loader = NORMS_LOADERS[norms_type]()
    feature_to_concepts, feature_to_id, features_selected = norm_loader()
    selected_concepts = norm_loader.load_concepts()
    selected_labels = [dataset.class_to_label[c] for c in selected_concepts]

    embeddings, labels = load_embeddings(dataset_name, feature_type, embeddings_level)
    idxs = np.isin(labels, selected_labels)
    embeddings = embeddings[idxs]
    labels = labels[idxs]

    FOLDER = "output/{}-predictions/{}/{}".format(
        classifier_type, embeddings_level, split_type
    )
    os.makedirs(FOLDER, exist_ok=True)

    def get_path(feature):
        feature_norm = norm_loader.get_suffix()
        feature_id = feature_to_id[feature]
        return "{}/{}-{}-{}-{}".format(
            FOLDER,
            dataset_name,
            feature_type,
            feature_norm,
            feature_id,
        )

    splits = GET_TRAIN_TEST_SPLIT[split_type](
        labels,
        features_selected,
        feature_to_concepts=feature_to_concepts,
        class_to_label=dataset.class_to_label,
    )

    make_classifier = CLASSIFIERS[classifier_type]

    def save_preds(path, results):
        data = [{k: r[k] for k in ("split", "preds")} for r in results]
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
 
    def save_clfs(path, results):
        data = [{k: r[k] for k in ("split", "clf")} for r in results]
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def cache_clf_and_preds(path, func, *args):
        path_json = path + ".json"
        if os.path.exists(path_json):
            return
        else:
            results = func(*args)
            save_preds(path_json, results)
            # Avoid saving classifiers for KNN since it stores the whole data!
            if classifier_type != "knn-3":
                save_clfs(path, results)

    def process_feature(feature):
        binary_labels = get_binary_labels(
            labels,
            feature,
            feature_to_concepts,
            dataset.class_to_label,
        )
        cache_clf_and_preds(
            get_path(feature),
            predict_splits,
            make_classifier,
            embeddings,
            binary_labels,
            splits[feature],
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
