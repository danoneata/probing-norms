import pdb
import torch
import numpy as np

from probing_norms.utils import read_json

data = read_json("data/things/concepts-and-categories.json")
# data = [datum for datum in data if datum["definition"] == ""]
data = sorted(data, key=lambda x: len(x["definition"]))
print(data[:10])