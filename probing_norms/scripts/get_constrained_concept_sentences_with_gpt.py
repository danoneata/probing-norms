import os
import pdb
import random
import json
import sys
from collections import defaultdict

from openai import OpenAI

from probing_norms.data import load_things_concept_mapping
from probing_norms.utils import cache_json, read_json, read_file


QUESTION = "Write 50 short sentences about {}. You must use {} as a noun in each sentence. Try to avoid using the following phrases in any of the sentences: {}."


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


def get_prompt(question, word, word_with_category_parenthesis, excludes):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are asked to write 50 short sentences about a word (to follow). Answer the request by returning a list of numbered sentences, 1--50. ",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question.format(word, word_with_category_parenthesis, excludes),
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


def do1(id, word, word_with_category_parenthesis, excludes):
    question = QUESTION
    prompt = get_prompt(question, word, word_with_category_parenthesis, excludes)

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


def make_useful(concept_string):
    '''
    'aardvark,aardvark,animal,aardvark (animal)' -> list
    '''
    return concept_string.split(",")


def get_id(concept):
    '''
       concept: ["aluminum_foil", "aluminum foil", "food packaging", "aluminum foil (food packaging)"]
       We want the second element of the list
    '''
    return concept[0]

def get_word(concept):
    '''
       concept: ["aluminum_foil", "aluminum foil", "food packaging", "aluminum foil (food packaging)"]
       We want the second element of the list
    '''
    return concept[1]


def get_word_with_category_parenthesis(concept):
    '''
       Not every concept has a category parenthesis version
       ['aardvark', 'aardvark', '', '']
       We want to return it iff it exists, otherwise we just want the word
    '''
    if concept[-1] == "":
        return get_word(concept)
    else:
        return concept[-1]


def get_constraints(concept, mappingDict):
    return ", ".join(mappingDict[concept])


if __name__ == "__main__":
    concepts_file = sys.argv[1]
    concepts = read_file(concepts_file)
    concepts = sorted(concepts)
    conceptNormsDict = conceptNormLoader()
    print(f"Read {len(concepts)} concepts from {concepts_file}")

    random.seed(1337)

    output = {}

    with open('constrained_concept_sentences.jsonl', 'w', buffering=1) as f:
        for c in concepts:
            cc = make_useful(c)
            response = do1(get_id(cc), get_word(cc), get_word_with_category_parenthesis(cc), get_constraints(get_id(cc), conceptNormsDict))
            f.write(json.dumps(response) + "\n")
