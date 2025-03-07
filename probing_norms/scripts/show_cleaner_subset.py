import pdb
from probing_norms.utils import cache_np
from probing_norms.predict import McRaeNormsLoader
from probing_norms.scripts.map_norms_to_mcrae import normalize_text, SentenceModel
from sklearn.cluster import DBSCAN
import numpy as np


loader = McRaeNormsLoader()
feature_to_concepts, _, norms = loader(num_min_concepts=0)
norms1 = list(map(normalize_text, norms))

# for f, cs in feature_to_concepts.items():
#     if len(cs) == 5:
#         print(f, cs)

path = "/tmp/sims-mcrae-norms.npy"
model = SentenceModel()
similarities = cache_np(path, model.get_similarities, norms1, norms1)

print(len(norms))

# with open("output/mcrae-norms-subset.txt", "w") as f:
count = 0
for i, (norm, sims) in enumerate(zip(norms, similarities)):
    idxs = sims >= 0.9
    start = i + 1
    if idxs[start:].sum() == 0:
        continue
    count += 1
    common = [norm, ", ".join(feature_to_concepts[norm])]
    for j, idx in enumerate(idxs[start:], start=start):
        if idx:
            row = common + [
                norms[j],
                sims[j],
                ", ".join(feature_to_concepts[norms[j]])
            ]
            print("\t".join(map(str, row)))