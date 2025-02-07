import os
import pdb
import random

from openai import AzureOpenAI

from probing_norms.data import load_things_concept_mapping
from probing_norms.utils import cache_json, read_json, read_file
from probing_norms.predict import McRaeNormsLoader, McRaeMappedNormsLoader


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


concepts = read_file("data/concepts-things.txt")
concepts = sorted(concepts)

concept_mapping = load_things_concept_mapping()


def normalize_concept(text):
    return concept_mapping[text]


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
                    "text": "You are asked to evaluate whether specific attributes can be applied to a list of general-level concepts (to follow). Answer the questions just by listing the attributes, comma separated. Here is the list of concepts (one per line):\n{}".format(
                        "\n".join(map(normalize_concept, concepts))
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


def get_path(feature_id, question_type):
    return "output/gpt4o-annots/{}-{}.json".format(feature_id, question_type)


def do1(feature, question_type):
    question = QUESTIONS[question_type]
    prompt = get_prompt(question, feature)

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
    print(feature, question_type)
    print(response.choices[0].message.content)
    print(response.usage.prompt_tokens)
    print(response.usage.completion_tokens)
    print(response.usage.total_tokens)
    print()

    return {
        "feature": feature,
        "question-type": question_type,
        "response": response.choices[0].message.content,
    }



if __name__ == "__main__":
    norms_loader = McRaeMappedNormsLoader()
    _, feature_to_id, features = norms_loader()

    random.seed(1337)
    features = random.sample(features, 100)
    # print(features)
    # pdb.set_trace()

    for feature in features:
        for t in "pos maybe neg".split():
            cache_json(get_path(feature_to_id[feature], t), do1, feature, t)