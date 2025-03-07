import random
import numpy as np
from sklearn.metrics import recall_score

num_concepts = 1854

def do(true_p, pred_p):
    pred = [random.random() <= pred_p for _ in range(num_concepts)]
    true = [random.random() <= true_p for _ in range(num_concepts)]
    return recall_score(true, pred)

for p in np.arange(0, 1.1, 0.1):
    print(do(p, p))