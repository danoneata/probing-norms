import os
import pdb
import random
import json
import sys
import ipdb
import argparse

from openai import OpenAI

from probing_norms.data import load_things_concept_mapping
from probing_norms.utils import cache_json, read_json, read_file


QUESTION = 'Is {} a common trait of {}, in the sense of {}. Please answer the request in JSON format with the following structure: {{"concept": CONCEPT, "attribute": ATTRIBUTE, "valid": ANSWER}}.'


def get_prompt(question, norm_language, concept, definition):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are asked to make judgements about whether attributes are valid or invalid in describing a concept (to follow).",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question.format(norm_language, concept, definition),
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


def make_useful(norm_string):
    '''
    'a_bird,is a bird,39' -> list
    '''
    return norm_string.split(",")


def get_norm(norm_data):
    '''
       norm: ["a_bird", "is a bird", "39"]
       We want the 0th element of the list
    '''
    return norm_data[0]


def get_norm_language(norm_data):
    '''
       norm: ["a_bird", "is a bird", "39"]
       We want the 1st element of the list
    '''
    return norm_data[1]


def clean_response(response):
    '''BUG: WHY WHY WHY'''
    clean = response.replace("```","")
    clean = clean.replace("json","")
    clean = clean.replace("\n","")
    return json.loads(clean)


def do1(id, concept, definition, norm, norm_language):
    question = QUESTION
    prompt = get_prompt(question, norm_language, concept, definition)

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

    return response.choices[0].message.content


def retrieve_concept(concept, concepts):
    for cc in concepts:
        if cc['id'] == concept:
            return cc
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Norm Prompter")
    parser.add_argument("--norms", type=str)
    parser.add_argument("--concepts", type=str)
    parser.add_argument("--target_norm", type=str)
    args = parser.parse_args()

    norms_file = args.norms
    norms = read_json(norms_file)  # we don't want the CSV headers
    print(f"Read {len(norms)} norms from {norms_file}")

    concepts_file = args.concepts
    concepts = read_json(concepts_file)  # we don't want the CSV headers
    print(f"Read {len(concepts)} norms from {concepts_file}")

    random.seed(1337)

    output_file = "concept_attributes.jsonl"
    if args.target_norm != None:
        output_file = "{}.jsonl".format(args.target_norm)
        total_counter = 0

    with open(output_file, 'w', buffering=1) as f:
        for nn in norms:
            norm = nn['norm']
            norm_language = nn['norm (language)']
            if norm_language != args.target_norm:
                continue
            mcrae_concepts = nn['concepts-mcrae']
            for cc in mcrae_concepts:
                mcraeplus = retrieve_concept(cc, concepts)
                if mcraeplus == None:
                    continue
                else:
                    total_counter += 1
                
                response = do1(mcraeplus['id'], mcraeplus['concept'], mcraeplus['definition'], norm, norm_language)
                f.write(json.dumps(response) + "\n")
            f.write(str(total_counter)+"\n")
            f.write(str(len(mcrae_concepts)))

# python probing_norms/scripts/get_norms_with_gpt.py --concepts data/things/concepts-and-categories.json --norms data/mcrae-norms-grouped-with-concepts.json --target_norm "tastes good"
