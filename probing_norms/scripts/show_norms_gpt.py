import random
from probing_norms.scripts.show_results_per_feature_norm import load_features_metadata

feature_to_concepts, _ = load_features_metadata("chatgpt-gpt3.5-turbo")

features_selected = list(feature_to_concepts.keys())
random.seed(42)
f1 = random.sample(features_selected, 64)
f2 = random.sample(features_selected, 256 - 64)
features_selected = set(f1 + f2)

with open("output/feature-norms-gpt.tsv", "w") as f:
    for feature, concepts in feature_to_concepts.items():
        if feature in features_selected:
            f.write("{}\t{}\t{}\n".format(feature, len(concepts), ", ".join(concepts)))