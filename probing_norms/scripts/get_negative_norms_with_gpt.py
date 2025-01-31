import os
import pdb
import random

from openai import AzureOpenAI

from probing_norms.data import DIR_GPT3_NORMS, load_mcrae_feature_norms
from probing_norms.utils import read_json, read_file


concept_feature = load_mcrae_feature_norms()
features = list(set([f for c, f in concept_feature]))


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

    if text.startswith("a_") or text.startswith("an_"):
        text = "is_" + text

    text = text.replace("_", " ").strip()
    return text


def parse(line):
    line = line.strip()
    word, *_, word_with_category = line.split(",")
    if not word_with_category:
        word_with_category = word.replace("_", " ")
    return word, word_with_category


path = DIR_GPT3_NORMS / "data" / "things" / "words.csv"
concept_mapping = dict(read_file(str(path), parse)[1:])


def normalize_concept(text):
    return concept_mapping[text]


concepts = read_file("data/concepts-things.txt")
concpets_ss = random.sample(concepts, 128)


QUESTIONS = {
    "pos": "(positive) Which of the given concepts is certain to have the attribute {}?",
    "neg": "(negative) Which of the given concepts is certain *not* to have the attribute {}?",
    "maybe": "(uncertain) Which of the given concepts may or may not have the attribute {}?",
}


def get_prompt(question, norm):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are asked to evaluate whether specific attributes can be applied to a list of general-level concepts (to follow). Answer the questions just by listing the attributes, comma separated. Here is the list of concepts: {}".format(
                        ", ".join(map(normalize_concept, concpets_ss))
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question.format(normalize_norm(norm)),
                },
            ],
        },
    ]


endpoint = os.getenv("ENDPOINT_URL", "https://dano.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
subscription_key = read_json("config/azure-openai-api-key.json")["AZURE_OPENAI_API_KEY"]

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2024-05-01-preview",
)


def do1(feature, question_type):
        question = QUESTIONS[question_type]
        prompt = get_prompt(question, feature)

        response = client.chat.completions.create(
            model=deployment,
            messages=prompt,
            max_tokens=800,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False,
        )

        print(response.choices[0].message.content)
        # print(response.usage.prompt_tokens)
        # print(response.usage.completion_tokens)
        print(response.usage.total_tokens)
        print()


results = [
    {
        "feature": feature,
        "question-type": t,
        "response": do1(feature, t),
    }
    for feature in features[:10]
    for t in "pos neg maybe".split()
]

print(results)