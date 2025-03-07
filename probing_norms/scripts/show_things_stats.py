import pdb
import numpy as np
from collections import Counter
from probing_norms.data import THINGS

dataset = THINGS()
counts = list(Counter(dataset.labels).values())
print(np.mean(counts))
print(np.median(counts))
