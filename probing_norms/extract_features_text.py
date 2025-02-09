import csv
import json
import pdb

from functools import partial
from tqdm import tqdm

import click
import fasttext
import numpy as np
import pandas as pd
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
    def __init__(self, type_="first"):
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
        self.model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
        self.model.eval()
        self.dim = self.model.config.hidden_size
        GET_EMBEDDINGS = {
            "first": self.get_embeddings_first,
            "last": self.get_embeddings_last,
            "context-last": self.get_embeddings_last,
        }
        self.get_embeddings = GET_EMBEDDINGS[type_]

    def get_embeddings_first(self, inp):
        out = self.model.model.embed_tokens(inp["input_ids"])
        return out.mean(dim=[0, 1]).numpy()

    def get_embeddings_last(self, inp):
        out = self.model.model(**inp).last_hidden_state
        return out.mean(dim=[0, 1]).numpy()

    def __call__(self, text, context=None):
        with torch.no_grad():
            pdb.set_trace()
            input_ids = self.tokenizer(text, return_tensors="pt")
            # Remove BOS token.
            input_ids["input_ids"] = input_ids["input_ids"][:, 1:]
            return self.get_embeddings(input_ids)


class Glove:
    def __init__(self, n_tokens, dim=300):
        assert dim in {50, 100, 200, 300}
        assert n_tokens in {"6B", "840B"}
        path = f"data/glove/glove.{n_tokens}.{dim}d.txt"
        self.dim = dim
        self.words = pd.read_table(
            path,
            sep=" ",
            index_col=0,
            header=None,
            quoting=csv.QUOTE_NONE,
        )
        self.CONCEPTS_TO_SPLIT = {
            "backscratcher": "back scratcher",
            "bathmat": "bath mat",
            "bunkbed": "bunk bed",
            "cornhusk": "corn husk",
            "cufflink": "cuff link",
            "dogfood": "dog food",
            "doorhandle": "door handle",
            "doorknocker": "door knocker",
            "fencepost": "fence post",
            "footbath": "foot bath",
            "hotplate": "hot plate",
            "icemaker": "ice maker",
            "iceskate": "ice skate",
            "kneepad": "knee pad",
            "oilcan": "oil can",
            "roadsweeper": "road sweeper",
            "saltshaker": "salt shaker",
            "ticktacktoe": "tick tack toe",
        }

    def get_subwords(self, word):
        return self.CONCEPTS_TO_SPLIT[word].split()

    def __call__(self, text):
        def get1(word):
            return self.words.loc[word].values

        words = text.split()
        try:
            embs = [get1(word) for word in words]
        except KeyError:
            embs = [
                get1(subword)
                for word in words
                for subword in self.get_subwords(word)
            ]
        return np.mean(embs, axis=0)


FEATURE_EXTRACTORS = {
    "fasttext": FastText,
    "gemma-2b": Gemma,
    "gemma-2b-last": partial(Gemma, type_="last"),
    "gemma-2b-context-last": partial(Gemma, type_="context-last"),
    "glove-6b-300d": partial(Glove, n_tokens="6B"),
    "glove-840b-300d": partial(Glove, n_tokens="840B"),
}

MAPPING_TYPES = ["word", "word-and-category"]


def load_context():
    def remove_starting_number(sentence):
        num, *words = sentence.split()
        num = num.replace(".", "")
        assert num.isdigit()
        return " ".join(words)

    def parse_line(line):
        data = json.reads(line)
        word = data["id"]
        sentences = data["response"]
        sentences = sentences.split("\n")
        sentences = [remove_starting_number(s) for s in sentences]
        return word, sentences

    path = "data/things/gpt4o_concept_context_sentences.jsonl"
    return read_file(path, parse_line)


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

    contexts = load_context()

    for i, concept in enumerate(tqdm(concepts)):
        concept1 = concept_mapping[concept]
        # if concept != concept1: print(concept, concept1)
        if concept.endswith("1"):
            pdb.set_trace()
        word, sentences = contexts[i]
        assert word == concept1
        X[i] = feature_extractor(concept1, context=sentences)
        y[i] = dataset.class_to_label[concept]

    path_np = f"output/features-text/{dataset_name}-{feature_type}-{mapping_type}.npz"
    np.savez(path_np, X=X, y=y)


if __name__ == "__main__":
    main()
