import os
import pdb
import random

from openai import AzureOpenAI

# from probing_norms.data import DATASETS
# from probing_norms.predict import McRaeMappedNormsLoader
from probing_norms.utils import read_json, read_file


# dataset_name = "things"
# dataset = DATASETS[dataset_name]()

# norm_loader = McRaeMappedNormsLoader()
# feature_to_concepts, feature_to_id, features_selected = norm_loader()

concepts = read_file("data/concepts-things.txt")
concpets_ss = random.sample(concepts, 128)

endpoint = os.getenv("ENDPOINT_URL", "https://dano.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
subscription_key = read_json("config/azure-openai-api-key.json")["AZURE_OPENAI_API_KEY"]

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2024-05-01-preview",
)


QUESTIONS = {
    "pos": "Which of the given concepts are you certain can be described by the attribute {}?",
    "neg": "Which of the given concepts are you certain cannot be described by the attribute {}?",
    "maybe": "Which of the given concepts are unsure can be described by the attribute {}?",
}


def get_prompt(question, norm):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are asked to provide information on whether general-level concepts are related to various attributes. Answer the questions just by listing the attributes, comma separated. Here is the list of concepts that we are interested in: {}".format(", ".join(concpets_ss)),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question.format(norm.replace("_", " ")),
                },
            ],
        },
    ]

feature = "a_boat"
for t in "pos neg maybe".split():
    question = QUESTIONS[t]
    prompt = get_prompt(question, feature)
    print(prompt)

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

    print(response.to_json())
    print(response.usage.prompt_tokens)
    print(response.usage.completion_tokens)
    print(response.usage.total_tokens)