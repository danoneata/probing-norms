import pdb

from itertools import combinations
from typing import List, Optional

import inflect

from toolz import compose

from probing_norms.extract_features_text import (
    AutoTokenizer,
    GemmaContextual,
    load_things_concept_mapping,
    read_file,
)


def find(query: List[int], sentence: List[int]) -> Optional[int]:
    n = len(query)
    for i in range(len(sentence)):
        if query == sentence[i : i + n]:
            return i
    return None


concepts = read_file("data/concepts-things.txt")
concept_mapping = load_things_concept_mapping("word")

contexts = GemmaContextual.load_context("gpt4o_50_concept")
contexts_words = list(contexts.keys())
concept_words = [concept_mapping[concept] for concept in concepts]
print(set(contexts_words) - set(concepts))
print(set(concepts) - set(contexts_words))

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
inv_vocab = {v: k for k, v in tokenizer.get_vocab().items()}


def all_combinations(xs):
    n = len(xs)
    return [comb for i in range(1, n + 1) for comb in combinations(xs, i)]


inflect_engine = inflect.engine()


def generate_word_variants(word):
    def pluralize(word):
        SPECIAL = {
            "antenna": "antennae",
            "banjo": "banjos",
            "flamingo": "flamingos",
            "hovercraft": "hovercrafts",
        }
        try:
            return SPECIAL[word]
        except KeyError:
            return inflect_engine.plural(word)


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

    def generate(word):
        transformations = [
            # lambda w: w,
            lambda w: " " + w,
            capitalize,
            pluralize,
        ]
        return [word] + [compose(*ts)(word) for ts in all_combinations(transformations)]

    SPECIAL = {
        "christmas tree": "Christmas Tree",
        "eclair": "éclair",
        "souffle": "soufflé",
    }

    variants = generate(word) 
    if word in SPECIAL:
        variants1 = generate(SPECIAL[word])
        variants = variants + variants1
    return variants



def find_first_variant(variants, sentence):
    for word in variants:
        result = find(word, sentence)
        if result is not None:
            return {
                "index": result,
                "variant": word,
            }
    return None


for word, sentences in contexts.items():
    word1 = concept_mapping[word]
    words = generate_word_variants(word1)
    tokens_variants = [tokenizer.encode(word)[1:] for word in words]
    # print(word1)
    # print(tokens)
    # for token in tokens_word:
    #     print("{:7d} → {}".format(token, inv_vocab[token]))
    # print()
    for i, sentence in enumerate(sentences):
        tokens_sent = tokenizer.encode(sentence)
        # print(sentence)
        # print(tokens)
        result = find_first_variant(tokens_variants, tokens_sent)
        if result is None:
            print(word, i, " → ", sentence)
            # for token in tokens_sent:
            #     print("{:7d} → {}".format(token, inv_vocab[token]))
            # print()
            # pdb.set_trace()
    # print("---")
