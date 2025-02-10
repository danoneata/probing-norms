import os
import pdb
import random
import json
import sys

from openai import OpenAI

from probing_norms.data import load_things_concept_mapping
from probing_norms.utils import cache_json, read_json, read_file
from probing_norms.predict import McRaeNormsLoader, McRaeMappedNormsLoader


QUESTION = "Write 50 short sentences about {}. You must use {} as a noun in each sentence."

def get_prompt(question, word, word_with_category_parenthesis):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are asked to write 10 short sentences about a word (to follow). Answer the request by returning a list of numbered sentences, 1--50. ",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question.format(word, word_with_category_parenthesis),
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


def do1(id, word, word_with_category_parenthesis):
    question = QUESTION
    prompt = get_prompt(question, word, word_with_category_parenthesis)

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


if __name__ == "__main__":
    concepts_file = sys.argv[1]
    concepts = read_file(concepts_file)
    concepts = sorted(concepts)
    print(f"Read {len(concepts)} concepts from {concepts_file}")

    random.seed(1337)

    output = {}

    with open('concept_sentences.jsonl', 'w', buffering=1) as f:
        for c in concepts:
            cc = make_useful(c)
            response = do1(get_id(cc), get_word(cc), get_word_with_category_parenthesis(cc))
            f.write(json.dumps(response) + "\n")
