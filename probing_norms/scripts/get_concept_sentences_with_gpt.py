import os
import pdb
import random
import json
import sys
import argparse
from collections import defaultdict

from openai import OpenAI

from probing_norms.data import load_things_concept_mapping
from probing_norms.utils import cache_json, read_json, read_file


QUESTION = "Write {} short sentences about {}, in the sense of {}. You must use {} as a noun in each sentence. Return a list of numbered sentences 1 -- {}."
QUESTION_CONSTRAINED = "Write {} short sentences about {}, in the sense of {}. You must use {} as a noun in each sentence. Try to avoid using any of the following phrases in any of the sentences {}. Return a list of numbered sentences 1 -- {}."


def conceptNormLoader(file="data/mcrae++.json"):
    jsondata = json.load(open(file))  # DE: does this create a memory leak?
    cNDict = defaultdict(list)
    for norm in jsondata:
        for concept in norm['concepts']:
            cNDict[concept].append(normalize_norm(norm['norm']))
    return cNDict

        
def normalize_norm(text):
    SEP = "_-_"
    if SEP in text:
        prefix, text = text.split(SEP)
        assert prefix in {"beh", "inbeh", "eg", "has_units", "worn_by_men"}, prefix
        if prefix == "eg":
            text = "example: " + text
        elif prefix in {"has_units", "worn_by_men"}:
            text = text + " " + prefix
        else:
            pass

    if text.startswith("is_"):
        text = text.replace("is_", "")

    if text.startswith("a_") or text.startswith("an_"):
        text = text.replace("a_", "")
        text = text.replace("an_", "")

    text = text.replace("_", " ").strip()
    return text


def get_constraints(concept, mappingDict):
    return ", ".join(mappingDict[concept])


def get_constrained_prompt(question, word, count, sense, constraints):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are asked to write short sentences about a word (to follow).",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question.format(count, word, sense, word, count, constraints),
                },
            ],
        },
    ]

def get_prompt(question, word, count, sense):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are asked to write short sentences about a word (to follow).",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question.format(count, word, sense, word, count),
                },
            ],
        },
    ]


deployment = "gpt-4o"
subscription_key = read_json("openai-api-key.json")["OPENAI_API_KEY"]
org = read_json("openai-api-key.json")["ORG"]
proj = read_json("openai-api-key.json")["PROJECT"]

client = OpenAI(
    organization=org,
    project=proj,
    api_key=subscription_key,
)


def do1(id, word, count, sense):
    question = QUESTION
    prompt = get_prompt(question, word, count, sense)

    response = client.chat.completions.create(
        model=deployment,
        messages=prompt,
        # max_tokens=800,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False,
    )

    print(prompt)
    print(word)
    print(response.choices[0].message.content)
    print(response.usage.prompt_tokens)
    print(response.usage.completion_tokens)
    print(response.usage.total_tokens)
    print()

    return {
        "id": id,
        "response": response.choices[0].message.content,
    }


def do2(id, word, count, sense, constraints):
    question = QUESTION_CONSTRAINED
    prompt = get_constrained_prompt(question, word, count, sense, constraints)

    response = client.chat.completions.create(
        model=deployment,
        messages=prompt,
        # max_tokens=800,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False,
    )

    print(prompt)
    print(word)
    print(response.choices[0].message.content)
    print(response.usage.prompt_tokens)
    print(response.usage.completion_tokens)
    print(response.usage.total_tokens)
    print()

    return {
        "id": id,
        "response": response.choices[0].message.content,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Norm Prompter")
    parser.add_argument("--concepts", type=str)
    parser.add_argument("--target_concept", type=str)
    parser.add_argument("--target_concepts", type=str)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--constrained", action="store_true", default=False)

    args = parser.parse_args()
    concepts_file = args.concepts
    concepts = read_json(concepts_file)
    print(f"Read {len(concepts)} concepts from {concepts_file}")
    if args.target_concepts != None:
        targets = read_file(args.target_concepts)
        print(f"Read {len(targets)} concepts from {args.target_concepts}")
    if args.constrained:
        conceptNormsDict = conceptNormLoader()


    random.seed(1337)

    for c in concepts:
        if c['id'] in targets:
            if args.constrained:
                fname = '{}-{}-constrained-concept_sentences.jsonl'.format(c['id'], args.count)
            else:
                fname = '{}-{}-concept_sentences.jsonl'.format(c['id'], args.count)
            with open(fname, 'w', buffering=1) as f:
                if args.constrained:
                    response = do2(c['id'], c['concept'], args.count, c['definition'], get_constraints(c['id'], conceptNormsDict))
                else:
                    response = do1(c['id'], c['concept'], args.count, c['definition'])
                f.write(json.dumps(response) + "\n")
