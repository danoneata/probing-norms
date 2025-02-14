import pdb
import os

from functools import partial
from itertools import groupby
from pathlib import Path

import pandas as pd

from PIL import Image
from torch.utils.data import Dataset

from torchvision.datasets import ImageNet
from toolz import first, identity, second

from probing_norms.utils import read_file, reverse_dict


DIR_GPT3_NORMS = "/home/doneata/work/semantic-features-gpt-3"
DIR_GPT3_NORMS = os.environ.get("DIR_GPT3_NORMS", DIR_GPT3_NORMS)
DIR_GPT3_NORMS = Path(DIR_GPT3_NORMS)

DIR_THINGS = Path("/mnt/private-share/speechDatabases/THINGS")
DIR_IMAGENET = Path("/mnt/private-share/speechDatabases/imagenet12")


FEATURE_NORMS_OPTIONS = {
    "priming": ["mcrae", "cslb"],
    "model": ["chatgpt-gpt3.5-turbo", "gpt3-davinci"],
    "num_runs": [10, 30],
}


def load_gpt3_feature_norms(
    *,
    root=DIR_GPT3_NORMS,
    priming="mcrae",
    model="chatgpt-gpt3.5-turbo",
    num_runs=30,
):
    assert priming in FEATURE_NORMS_OPTIONS["priming"]
    assert model in FEATURE_NORMS_OPTIONS["model"]
    assert num_runs in FEATURE_NORMS_OPTIONS["num_runs"]
    path = (
        root
        / "data"
        / "gpt_3_feature_norm"
        / (priming + "_priming")
        / model
        / "all_things_concepts"
        / str(num_runs)
        / "decoded_answers.csv"
    )
    cols = ["concept_id", "decoded_feature"]
    df = pd.read_csv(path)
    df = df[cols]
    df = df.drop_duplicates()
    concept_feature = df.values.tolist()
    return concept_feature


def load_mcrae_feature_norms():
    cols = ["Concept", "Feature"]
    df = pd.read_csv("data/norms/mcrae/CONCS_FEATS_concstats_brm.txt", sep="\t")
    df = df[cols]
    df = df.drop_duplicates()
    concept_feature = df.values.tolist()
    return concept_feature


def filter_by_things_concepts(df):
    concepts_things = read_file("data/concepts-things.txt")
    supercats = [
        "living object",
        "artifact",
        "natural object",
    ]

    idxs = df["Super Category"].isin(supercats)
    df = df[idxs]

    idxs = df["Word"].isin(concepts_things)
    df = df[idxs]

    return df


def load_binder_dense():
    df = pd.read_excel("data/binder-norms.xlsx")
    df = filter_by_things_concepts(df)

    cols = df.columns[5: 70]
    df = df.set_index("Word")
    df = df[cols]

    df = df.unstack()
    df = df.reset_index()
    df = df.rename(columns={"level_0": "Feature", 0: "Value"})

    return df


def load_binder_feature_norms_median():
    df = load_binder_dense()
    median = df.groupby("Feature")["Value"].median()
    df["Median"] = df["Feature"].map(median)

    idxs = df["Value"] >= df["Median"]
    cols = ["Word", "Feature"]

    df = df[idxs]
    df = df.drop_duplicates()
    concept_feature = df[cols].values.tolist()
    return concept_feature


def load_binder_feature_norms(thresh):
    assert 0 < thresh < 6

    df = load_binder_dense()
    idxs = df["Value"] >= thresh
    df = df[idxs]

    cols = ["Word", "Feature"]
    df = df.drop_duplicates()
    concept_feature = df[cols].values.tolist()
    return concept_feature


def get_feature_to_concepts(concept_feature):
    concept_feature = sorted(concept_feature, key=second)
    feature_groups = groupby(concept_feature, key=second)
    return {feature: list(map(first, group)) for feature, group in feature_groups}


def load_features_metadata(
    *,
    priming="mcrae",
    model="chatgpt-gpt3.5-turbo",
    num_runs=30,
):
    concept_feature = load_gpt3_feature_norms(
        priming=priming,
        model=model,
        num_runs=num_runs,
    )
    feature_to_concepts = get_feature_to_concepts(concept_feature)
    features = sorted(feature_to_concepts.keys())
    feature_to_id = {feature: i for i, feature in enumerate(features)}
    return feature_to_concepts, feature_to_id


class THINGS(Dataset):
    def __init__(self, root=DIR_THINGS, transform=identity):
        self.root = root
        self.transform = transform
        image_files, labels, label_to_class = self.prepare_metadata(root)
        self.image_files = image_files
        self.labels = labels
        self.label_to_class = label_to_class
        self.class_to_label = reverse_dict(label_to_class)

    def get_image_path(self, image_name):
        _, class_name, file_name = image_name.split("/")
        path = self.root / "object_images" / class_name / file_name
        return str(path)

    def prepare_metadata(self, root):
        def get_class_name(path):
            _, class_name, _ = path.split("/")
            return class_name

        metadata_path = str(root / "01_image-level" / "image-paths.csv")
        image_files = read_file(metadata_path)

        classes = [get_class_name(path) for path in image_files]
        label_to_class = dict(enumerate(sorted(set(classes))))
        class_to_label = reverse_dict(label_to_class)

        labels = [class_to_label[c] for c in classes]
        return image_files, labels, label_to_class

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, i):
        image_file = self.image_files[i]
        image_path = self.get_image_path(image_file)
        image = Image.open(image_path)
        image = self.transform(image)
        label = self.labels[i]
        return {
            "name": image_file,
            "image": image,
            "label": label,
        }


def load_things_concept_mapping(type_="word-and-category"):
    MAP_TYPE = {
        "word": 0,
        "category": 1,
        "word-and-category": 2,
    }

    def parse(line):
        line = line.strip()
        key, *rest = line.split(",")
        value = rest[MAP_TYPE[type_]]
        if not value:
            value = rest[0]
        return key, value


    assert type_ in MAP_TYPE
    path = "data/things/words.csv"
    return dict(read_file(str(path), parse)[1:])


DATASETS = {
    "imagenet12": partial(ImageNet, root=DIR_IMAGENET, split="val"),
    "things": partial(THINGS, root=DIR_THINGS),
}
