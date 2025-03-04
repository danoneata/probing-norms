import os
import pdb
import random
import json
import sys
import ipdb
import argparse
from pathlib import Path

from openai import OpenAI

from probing_norms.data import load_things_concept_mapping
from probing_norms.utils import cache_json, read_json, read_file


QUESTION = "Is {} a common trait of {}, in the sense of {}?"
SYSTEM_BATCH = """You are asked to decide whether an attribute is a common trait of a concept (to follow). Please answer the request in JSON format with the following structure: {'concept': CONCEPT, 'attribute': ATTRIBUTE, 'valid': ANSWER}"""


deployment = "gpt-4o"
subscription_key = read_json("openai-api-key.json")["OPENAI_API_KEY"]
org = read_json("openai-api-key.json")["ORG"]
proj = read_json("openai-api-key.json")["PROJECT"]
client = OpenAI(
    organization=org,
    project=proj,
    api_key=subscription_key,
)


def create_batch_api_input(target_norm, norms, concepts, submit=False):

    batchf = """{{"custom_id": "request-{}", "method": "POST", "url": "/v1/chat/completions", "body": {{"model": "gpt-4o", "messages": [{{"role": "system", "content": "{}"}},{{"role": "user", "content": "{}"}}], "max_tokens": 100}}"""

    alignf = """{{"custom_id": "request-{}", "concept_id": "{}", "norm": "{}"}}"""
    
    for nn in norms:
        norm = nn['norm']
        norm_language = nn['norm (language)']
        if args.target_norm == "" or norm == args.target_norm:
            i = 0
            norm = norm.replace("/", "ORRR")
            output_file = "data/api-gpt4o/inputs/{}.jsonl".format(norm)
            output_handle = open(output_file, "w")
            align_file = "data/api-gpt4o/align/{}.jsonl".format(norm)
            align_handle = open(align_file, "w")
            norm = norm.replace("ORRR", "/")
            for cc in concepts:
                USER = QUESTION.format(norm_language, cc['concept'], cc['definition'])
                jsonline = batchf.format(i, SYSTEM_BATCH, USER)
                output_handle.write(jsonline+"}\n")
                alignline = alignf.format(i, cc['id'], norm)
                align_handle.write(alignline+"\n")
                i += 1
            output_handle.close()
            align_handle.close()
            if submit:
                batch_input_file = client.files.create(
                    file=open(output_file, "rb"),
                    purpose="batch"
                )
                batch_input_file_id = batch_input_file.id
                with open("file_ids.txt", "a") as f:
                    f.write("{}: {}".format(output_file, batch_input_file_id))
            
                req = client.batches.create(
                    input_file_id=batch_input_file_id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={
                        "description": "ACL 2025 data collection job"
                    }
                )
                with open("batch_ids.txt", "a") as f:
                    f.write("{}: {}\n".format(output_file, req.id))
        else:
            continue


def retrieve_batch_api_output():
    filenames = open("batch_ids.txt", "r").readlines()
    for f_pair in filenames:
        f_pair = f_pair.replace("\n", "")
        f_pair = f_pair.split(":")
        """ BUG BUG BUG """
        write_path = f_pair[0]
        the_path = Path(write_path)
        new_path = "{}/{}".format(the_path.parts[0], the_path.parts[1])
        new_path += "/outputs/"
        new_path += "output-{}{}".format(the_path.stem, the_path.suffix)
        """ BUG BUG BUG """
        api_response = client.batches.retrieve(f_pair[1][1:])
        if api_response.status == "completed":
            file_response = client.files.content(api_response.output_file_id)
            with open(new_path, "w") as f:
                f.write(file_response.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Norm Prompter")
    parser.add_argument("--norms", type=str)
    parser.add_argument("--concepts", type=str)
    parser.add_argument("--target_norm", type=str, default="")
    parser.add_argument("--batch_api", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--retrieve", action="store_true")
    args = parser.parse_args()

    norms_file = args.norms
    norms = read_json(norms_file)  # we don't want the CSV headers
    print(f"Read {len(norms)} norms from {norms_file}")

    concepts_file = args.concepts
    concepts = read_json(concepts_file)  # we don't want the CSV headers
    print(f"Read {len(concepts)} norms from {concepts_file}")

    random.seed(1337)

    if args.batch_api:
        create_batch_api_input(args.target_norm, norms, concepts, args.submit)

    if args.retrieve:
        retrieve_batch_api_output()

# python probing_norms/scripts/get_norms_with_gpt.py --concepts data/things/concepts-and-categories.json --norms data/mcrae-norms-grouped-with-concepts.json --target_norm "tastes good" --eval_run
