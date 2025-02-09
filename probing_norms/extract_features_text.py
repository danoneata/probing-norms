import csv
import json
import pdb

from itertools import combinations
from functools import partial
from typing import List, Optional, Tuple

import click
import fasttext
import inflect
import numpy as np
import pandas as pd
import torch

from huggingface_hub import hf_hub_download
from toolz import compose
from tqdm import tqdm
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

    def __call__(self, text, **_):
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

    def __call__(self, text, **_):
        with torch.no_grad():
            input_ids = self.tokenizer(" " + text, return_tensors="pt")
            # Remove BOS token.
            # input_ids["input_ids"] = input_ids["input_ids"][:, 1:]
            return self.get_embeddings(input_ids)


class GemmaContextual(Gemma):
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
        self.contexts = self.load_context()
        self.inflect_engine = inflect.engine()

    def load_context(self):
        def remove_starting_number(sentence):
            sentence = sentence.strip()
            num, *words = sentence.split()
            num = num.replace(".", "")
            try:
                assert num.isdigit()
            except AssertionError:
                pdb.set_trace()
            return " ".join(words)

        def parse_line(line):
            data = json.loads(line)
            id_ = data["id"]
            sentences = data["response"]
            sentences = sentences.split("\n")
            sentences = [remove_starting_number(s) for s in sentences]
            return id_, sentences

        path = "data/things/gpt4o_concept_context_sentences.jsonl"
        return dict(read_file(path, parse_line))

    def generate_word_variants(self, word):
        """Generate multiple variants of a word as it can appear in various forms in the context sentences.
        For example, "aardvark" can appear in "Aardvarks are nice animals."
        
        """

        def all_combinations(xs):
            n = len(xs)
            return [comb for i in range(1, n + 1) for comb in combinations(xs, i)]

        def prepend_space(word):
            return " " + word

        def pluralize(word):
            SPECIAL = {
                "antenna": "antennae",
                "flamingo": "flamingos",
            }
            try:
                return SPECIAL[word]
            except KeyError:
                return self.inflect_engine.plural(word)

        def capitalize(word):
            SPECIAL = {
                "cd player": "CD player",
                "sim card": "SIM card",
                "sim cards": "SIM cards",
            }
            try:
                return SPECIAL[word]
            except KeyError:
                return word.capitalize()

        transformations = [prepend_space, capitalize, pluralize]
        return [word] + [compose(*ts)(word) for ts in all_combinations(transformations)]

    @staticmethod
    def find_first_variant(variants, sentence):
        def find1(query, sentence):
            n = len(query)
            for s in range(len(sentence)):
                e = s + n
                if query == sentence[s:e]:
                    return s, e
            return None

        for word in variants:
            result = find1(word, sentence)
            if result is not None:
                return {
                    "range": result,
                    "word": word,
                }
        return None

    def __call__(self, word, *, word_id):
        def get_emb(sentence):
            input_ids = self.tokenizer(sentence, return_tensors="pt")
            result = self.find_first_variant(variants, sentence)
            if result is not None:
                s, e = result["range"]
                sentence_embeddings = self.get_embeddings(input_ids)
                sentence_embeddings[s:e].mean(dim=0)
            else:
                return None

        variants = self.generate_word_variants(word)
        with torch.no_grad():
            embs = [get_emb(sent) for sent in self.contexts[word_id]]
            embs = [emb for emb in embs if emb is not None]
            print(word_id, len(embs))
            pdb.set_trace()
            emb = np.mean(embs, axis=0)
            return emb.numpy()


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

    def __call__(self, text, **_):
        def get1(word):
            return self.words.loc[word].values

        words = text.split()
        try:
            embs = [get1(word) for word in words]
        except KeyError:
            embs = [
                get1(subword) for word in words for subword in self.get_subwords(word)
            ]
        return np.mean(embs, axis=0)


FEATURE_EXTRACTORS = {
    "fasttext": FastText,
    "gemma-2b": Gemma,
    "gemma-2b-last": partial(Gemma, layer="last"),
    "gemma-2b-context-last": partial(GemmaContextual, layer="last"),
    "glove-6b-300d": partial(Glove, n_tokens="6B"),
    "glove-840b-300d": partial(Glove, n_tokens="840B"),
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
        # if concept.endswith("1"):
        #     pdb.set_trace()
        # word, sentences = contexts[i]
        # try:
        #     assert word == concept1
        # except AssertionError:
        #     pdb.set_trace()
        X[i] = feature_extractor(concept1, word_id=concept)
        y[i] = dataset.class_to_label[concept]

    path_np = f"output/features-text/{dataset_name}-{feature_type}-{mapping_type}.npz"
    np.savez(path_np, X=X, y=y)


if __name__ == "__main__":
    main()
