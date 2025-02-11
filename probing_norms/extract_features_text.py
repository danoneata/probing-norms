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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, CLIPModel

from probing_norms.data import DATASETS, load_things_concept_mapping
from probing_norms.utils import implies, read_file


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
    def __init__(self):
        self.model_class = AutoModelForCausalLM

    def get_backbone(self, model):
        return model.model

    def get_tokens(self, word, tokenizer):
        tokens = tokenizer.encode(word)
        bos, *rest = tokens
        assert bos == tokenizer.bos_token_id
        return rest


class GPT2:
    def __init__(self):
        self.model_class = AutoModelForCausalLM

    def get_backbone(self, model):
        return model.transformer

    def get_tokens(self, word, tokenizer):
        return tokenizer.encode(word)


class DeBERTa:
    def __init__(self):
        self.model_class = AutoModel

    def get_backbone(self, model):
        return model

    def get_tokens(self, word, tokenizer):
        tokens = tokenizer.encode(word)
        bos, *rest, eos = tokens
        assert bos == tokenizer.vocab["[CLS]"]
        assert eos == tokenizer.vocab["[SEP]"]
        return rest


BASE_HF_MODELS = {
    "gemma": Gemma,
    "gpt2": GPT2,
    "deberta": DeBERTa,
}


class HFModel:
    def __init__(self, model_type, model_id, layer):
        self.base_model = BASE_HF_MODELS[model_type]()
        self.device = "cuda"

        model_class = self.base_model.model_class
        self.model = model_class.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.model.eval()
        self.model.to(self.device)
        self.dim = self.model.config.hidden_size
        self.model = self.base_model.get_backbone(self.model)

        GET_EMBEDDINGS = {
            "zero": self.get_embeddings_first,
            "last": self.get_embeddings_last,
            "layer-1": partial(self.get_embeddings_layer, layer=1),
            "layer-2": partial(self.get_embeddings_layer, layer=2),
        }
        self.get_embeddings = GET_EMBEDDINGS[layer]

    def get_embeddings_layer(self, inp, layer):
        output = self.model(**inp, output_hidden_states=True)
        return output.hidden_states[layer]

    def get_embeddings_first(self, inp):
        return self.model.embed_tokens(inp["input_ids"])

    def get_embeddings_last(self, inp):
        return self.model(**inp).last_hidden_state

    def __call__(self, text, **_):
        with torch.no_grad():
            input_ids = self.tokenizer(" " + text, return_tensors="pt")
            # Remove BOS token.
            # input_ids["input_ids"] = input_ids["input_ids"][:, 1:]
            embs = self.get_embeddings(input_ids)
            return embs.mean(dim=[0, 1]).numpy()


class HFModelContextual(HFModel):
    def __init__(self, model_type, model_id, layer, context_type):
        super().__init__(model_type, model_id, layer)
        self.contexts = self.load_context(context_type)
        self.inflect_engine = inflect.engine()

    @staticmethod
    def load_context(type_):
        def remove_starting_number(sentence):
            sentence = sentence.strip()
            num, *words = sentence.split()
            num = num.replace(".", "")
            assert num.isdigit()
            return " ".join(words)

        def parse_line(line):
            data = json.loads(line)
            id_ = data["id"]
            sentences = data["response"]
            sentences = sentences.split("\n")
            sentences = [s for s in sentences if s]
            sentences = [remove_starting_number(s) for s in sentences]
            return id_, sentences

        assert type_ in {
            "gpt4o_concept",
            "gpt4o_50_concept",
            "gpt4o_50_constrained_concept",
        }
        path = f"data/things/{type_}_context_sentences.jsonl"
        return dict(read_file(path, parse_line))

    def generate_word_variants(self, word):
        """Generate multiple variants of a word as it can appear in various forms in the context sentences.
        For example, "aardvark" can appear as "Aardvarks": "Aardvarks are nice animals."

        """

        def all_combinations(xs):
            n = len(xs)
            return [comb for i in range(1, n + 1) for comb in combinations(xs, i)]

        def prepend_space(word):
            return " " + word

        def pluralize(word):
            SPECIAL = {
                "banjo": "banjos",
                "antenna": "antennae",
                "flamingo": "flamingos",
                "hovercraft": "hovercrafts",
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

        def do(word):
            transformations = [prepend_space, capitalize, pluralize]
            return [word] + [
                compose(*ts)(word) for ts in all_combinations(transformations)
            ]

        OTHER_FORMS = {
            "christmas tree": ["Christmas Tree"],
            "eclair": ["éclair"],
            "souffle": ["soufflé"],
        }
        return [
            variant for w in [word] + OTHER_FORMS.get(word, []) for variant in do(w)
        ]

    @staticmethod
    def find_location(variants, sentence):
        """Finds the location of a concept variant ("Aardvark", "aardvarks", etc.) in the sentence.
        Stops at the first match.

        """

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
        variants = self.generate_word_variants(word)
        tokens_variants = [
            self.base_model.get_tokens(word, self.tokenizer) for word in variants
        ]

        def get_emb(sentence):
            tokens_sentence = self.tokenizer.encode(sentence)
            result = self.find_location(tokens_variants, tokens_sentence)
            if result is not None:
                s, e = result["range"]
                input_ids = self.tokenizer(sentence, return_tensors="pt")
                input_ids = input_ids.to(self.device)
                sentence_embeddings = self.get_embeddings(input_ids)
                assert sentence_embeddings.shape[1] == len(tokens_sentence)
                return sentence_embeddings[0, s:e].mean(dim=0).cpu().numpy()
            else:
                return None

        with torch.no_grad():
            embs = [get_emb(sent) for sent in self.contexts[word_id]]
            embs = [emb for emb in embs if emb is not None]
            # print(word_id, len(embs))
            tqdm.write(f"{word_id}: {len(embs)}")
            return np.mean(embs, axis=0)


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


class CLIP:
    def __init__(self) -> None:
        model_id = "openai/clip-vit-large-patch14"
        self.model = CLIPModel.from_pretrained(model_id).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.dim = self.model.config.projection_dim

    def __call__(self, text, **_):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            output = self.model.get_text_features(**inputs)
            return output[0].cpu().numpy()


FEATURE_EXTRACTORS = {
    "fasttext": FastText,
    "glove-6b-300d": partial(Glove, n_tokens="6B"),
    "glove-840b-300d": partial(Glove, n_tokens="840B"),
    "clip": CLIP,
    "gemma-2b": partial(
        HFModel,
        model_type="gemma",
        model_id="google/gemma-2b",
        layer="zero",
    ),
    "gemma-2b-last": partial(
        HFModel,
        model_type="gemma",
        model_id="google/gemma-2b",
        layer="last",
    ),
    "gemma-2b-contextual-last": partial(
        HFModelContextual,
        model_type="gemma",
        model_id="google/gemma-2b",
        layer="last",
        context_type="gpt4o_concept",
    ),
    "gemma-2b-contextual-layer-1": partial(
        HFModelContextual,
        model_type="gemma",
        model_id="google/gemma-2b",
        layer="layer-1",
        context_type="gpt4o_concept",
    ),
    "gemma-2b-contextual-layer-2": partial(
        HFModelContextual,
        model_type="gemma",
        model_id="google/gemma-2b",
        layer="layer-2",
        context_type="gpt4o_concept",
    ),
    "gemma-2b-contextual-50-last": partial(
        HFModelContextual,
        model_type="gemma",
        model_id="google/gemma-2b",
        layer="last",
        context_type="gpt4o_50_concept",
    ),
    "gemma-2b-contextual-50-constrained-last": partial(
        HFModelContextual,
        model_type="gemma",
        model_id="google/gemma-2b",
        layer="last",
        context_type="gpt4o_50_constrained_concept",
    ),
    "llama-3.1-8b-contextual-last": partial(
        HFModelContextual,
        model_id="meta-llama/Llama-3.1-8b",
        layer="last",
        context_type="gpt4o_concept",
    ),
    "gpt2-contextual-last": partial(
        HFModelContextual,
        model_type="gpt2",
        model_id="openai-community/gpt2",
        layer="last",
        context_type="gpt4o_concept",
    ),
    "deberta-v3-contextual-last": partial(
        HFModelContextual,
        model_type="deberta",
        model_id="microsoft/deberta-v3-base",
        layer="last",
        context_type="gpt4o_concept",
    ),
}

MAPPING_TYPES = ["word", "word-and-category"]


@click.command()
@click.option("-d", "--dataset", "dataset_name", type=str, required=True)
@click.option("-f", "--feature-type", "feature_type", type=str, required=True)
@click.option("-m", "--mapping-type", "mapping_type", type=str, required=True)
def main(dataset_name, feature_type, mapping_type):
    assert dataset_name == "things"
    assert implies(
        feature_type == "gemma-2b-contextual-last",
        mapping_type == "word",
    ), "The contextual Gemma embedding supports only original concept words."

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
        X[i] = feature_extractor(concept1, word_id=concept)
        y[i] = dataset.class_to_label[concept]

    path_np = f"output/features-text/{dataset_name}-{feature_type}-{mapping_type}.npz"
    np.savez(path_np, X=X, y=y)


if __name__ == "__main__":
    main()
