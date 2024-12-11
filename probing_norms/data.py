import pdb

from functools import partial
from pathlib import Path

import pandas as pd

from PIL import Image
from torch.utils.data import Dataset

from torchvision.datasets import ImageNet
from toolz import identity

from probing_norms.utils import read_file, reverse_dict


DIR_GPT3_NORMS = Path("/home/doneata/work/semantic-features-gpt-3")
DIR_THINGS = Path("/mnt/private-share/speechDatabases/THINGS")
DIR_IMAGENET = Path("/mnt/private-share/speechDatabases/imagenet12")


def load_gpt3_feature_norms(
    root=DIR_GPT3_NORMS,
    priming="mcrae",
    model="chatgpt-gpt3.5-turbo",
    num_runs=30,
):
    assert priming in "mcrae cslb".split()
    assert model in "chatgpt-gpt3.5-turbo gpt3-davinci".split()
    assert num_runs in [10, 30]
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


class THINGS(Dataset):
    def __init__(self, root=DIR_THINGS, transform=identity):
        self.root = root
        self.transform = transform
        image_files, labels, label_to_class = self.prepare_metadata(root)
        self.image_files = image_files
        self.labels = labels
        self.label_to_class = label_to_class

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


DATASETS = {
    "imagenet12": partial(ImageNet, root=DIR_IMAGENET, split="val"),
    "things": partial(THINGS, root=DIR_THINGS),
}
