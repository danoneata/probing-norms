import pdb
import click

from functools import partial
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
        model_path = hf_hub_download(
            repo_id="facebook/fasttext-en-vectors",
            filename="model.bin",
        )
        self.model = fasttext.load_model(model_path)
        self.dim = self.model.get_dimension()

    def __call__(self, text):
        # return self.model.get_word_vector(text)
        return self.model.get_sentence_vector(text)


class Gemma:
    def __init__(self, layer="first"):
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
        self.model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
        self.model.eval()
        self.dim = self.model.config.hidden_size
        GET_EMBEDDINGS = {
            "first": self.get_embeddings_first,
            "last": self.get_embeddings_last,
        }
        self.get_embeddings = GET_EMBEDDINGS[layer]

    def get_embeddings_first(self, inp):
        out = self.model.model.embed_tokens(inp["input_ids"])
        return out.mean(dim=[0, 1]).numpy()

    def get_embeddings_last(self, inp):
        out = self.model.model(**inp).last_hidden_state
        return out.mean(dim=[0, 1]).numpy()

    def __call__(self, text):
        with torch.no_grad():
            input_ids = self.tokenizer(text, return_tensors="pt")
            # Remove BOS token.
            input_ids["input_ids"] = input_ids["input_ids"][:, 1:]
            return self.get_embeddings(input_ids)


FEATURE_EXTRACTORS = {
    "fasttext": FastText,
    "gemma-2b": Gemma,
    "gemma-2b-last": partial(Gemma, layer="last"),
}

MAPPING_TYPES = ["word", "word-and-category"]


@click.command()
@click.option("-d", "--dataset", "dataset_name", type=str, required=True)
@click.option("-f", "--feature-type", "feature_type", type=str, required=True)
@click.option("-m", "--mapping-type", "mapping_type", type=str, required=True)
def main(dataset_name, feature_type, mapping_type):
    assert dataset_name == "things"
    dataset = DATASETS[dataset_name]()
    concepts = read_file("data/concepts-things.txt")
    concept_mapping = load_things_concept_mapping(mapping_type)

    feature_extractor = FEATURE_EXTRACTORS[feature_type]()

    num_concepts = len(concepts)
    feature_dim = feature_extractor.dim

    X = np.zeros((num_concepts, feature_dim))
    y = np.zeros(num_concepts)

    for i, concept in enumerate(tqdm(concepts)):
        concept1 = concept_mapping[concept]
        # if concept != concept1: print(concept, concept1)
        X[i] = feature_extractor(concept1)
        y[i] = dataset.class_to_label[concept]

    path_np = f"output/features-text/{dataset_name}-{feature_type}-{mapping_type}.npz"
    np.savez(path_np, X=X, y=y)


if __name__ == "__main__":
    main()
