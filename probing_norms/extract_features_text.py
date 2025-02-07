import pdb

import click

from tqdm import tqdm

import fasttext
import numpy as np
import torch

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM

from probing_norms.data import DATASETS, load_things_concept_mapping
from probing_norms.utils import read_file


class FastText:
    def __init__(self):
        model_path = hf_hub_download(repo_id="facebook/fasttext-en-vectors", filename="model.bin")
        self.model = fasttext.load_model(model_path)
        self.dim = self.model.get_dimension()

    def __call__(self, text):
        # return self.model.get_word_vector(text)
        return self.model.get_sentence_vector(text)


class Gemma:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
        self.model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
        self.model.eval()
        self.dim = self.model.config.hidden_size

    def __call__(self, text):
        with torch.no_grad():
            input_ids = self.tokenizer(text, return_tensors="pt")
            output = self.model.model.embed_tokens(input_ids["input_ids"])
        return output.mean(dim=(0, 1)).numpy()


FEATURE_EXTRACTORS = {
    "fasttext": FastText,
    "gemma-2b": Gemma,
}


@click.command()
@click.option("-d", "--dataset", "dataset_name", type=str, required=True)
@click.option("-f", "--feature-type", "feature_type", type=str, required=True)
def main(dataset_name, feature_type):
    assert dataset_name == "things"
    dataset = DATASETS[dataset_name]()
    concepts = read_file("data/concepts-things.txt")
    concept_mapping = load_things_concept_mapping()

    num_concepts = len(concepts)

    feature_extractor = FEATURE_EXTRACTORS[feature_type]()
    features = np.zeros((num_concepts, feature_extractor.dim))
    labels = np.zeros(num_concepts)

    for i, concept in enumerate(tqdm(concepts)):
        concept1 = concept_mapping[concept]
        features[i] = feature_extractor(concept1)
        labels[i] = dataset.class_to_label[concept]

    path_np = f"output/features-text/{dataset_name}-{feature_type}.npz"
    np.savez(path_np, features=features, labels=labels)


if __name__ == "__main__":
    main()
