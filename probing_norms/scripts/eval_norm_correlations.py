import csv
import json
from collections import defaultdict
import pandas as pd

# from https://github.com/ViCCo-Group/THINGS-data/blob/main/MRI/data/Categories_final_20200131.tsv
THINGS_SUPERCATEGORIES='../../data/things/Categories_final_20200131.tsv'
# first row in the file is (here for reference):
SUPER_CATS="animal	bird	body part	clothing	clothing accessory	container	dessert	drink	electronic device	food	fruit	furniture	home decor	insect	kitchen appliance	kitchen tool	medical equipment	musical instrument	office supply	part of car	plant	sports equipment	tool	toy	vegetable	vehicle	weapon	candy	hardware	mammal	sea animal	fastener	jewelry	watercraft	condiment	garden tool	lighting	personal hygiene item	scientific equipment	arts and crafts supply	home appliance	seafood	breakfast food	footwear	safety equipment	school supply	headwear	protective clothing	construction equipment	game	farm animal	outerwear	women's clothing"

MODEL_SCORE_FILE='../../../results-pernorm-dinov2gemma.json'
MODEL_LIST=['dino-v2', 'gemma-2b-contextual-last-word']

def get_supercategories(file=THINGS_SUPERCATEGORIES):
    """Reads the supercategories file and returns a dictionary of supercategories to concepts.

    The file is formatted: word | definition | supercategories... [with 1 for assigned supercategory]
    Not all concepts have supercategories (1): 407 are missing, we ignore these words.
    """
    
    supercategories = defaultdict(set)
    missing = 0
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        supercat_set = header[2:]
        for row in reader:
            word = row[0]
            try:
                supercat_id = (row[2:]).index('1')
            except ValueError:
                print(f"Error: {row[0]} {''.join(row[2:])}")
                missing+=1
                continue
            supercategories[supercat_set[supercat_id]].add(word)

        # Future: there are a few very small supercategories (headwear, garden tool) that might need to be removed
        # they cause problems for overlap correlation more than jaccard and intersection.
        print(f"Found supercategories for {sum([len(supercategories[x]) for x in supercategories])} words")
        print(f"Missing supercategories for {missing} words")
        print(f"Found {len(supercategories)} supercategories")
        #print("supercategories: ", supercategories.keys())
        # for supercat in supercategories:
        #     print(f"supercategory: {supercat} {len(supercategories[supercat])} words")

    return supercategories

def get_features_from_results(file=MODEL_SCORE_FILE):
    """Uses the result.json file to get the concepts associated with each feature.
    
    Returns: dict of feature to set of concepts, and the set of all concepts
    """
    concept_set = set()
    with open(file, 'r') as f:
        results = json.load(f)
    feature_to_concepts = {}
    for x in results:
        feature = x['feature']
        concepts = set(x['concepts'])
        feature_to_concepts[feature] = concepts
        concept_set.update(concepts)
    return feature_to_concepts, concept_set

def get_scores_from_results(model_name, file=MODEL_SCORE_FILE):
    """Returns the scores for each feature for a given model."""
    with open(file, 'r') as f:
        results = json.load(f)
    feature_to_scores = {}
    for x in results:
        if x['model'] == model_name:
            feature = x['feature']
            score = x['score-f1-selectivity']
            feature_to_scores[feature] = score
    return feature_to_scores

def get_best_supercategory_matches():
    """Matches each feature to a supercategory based on the feature's category set.
    We use the supercategory with the highest intersection with the feature's category set,
    which is later normalized by the size of the category set.
    """

    supercategories = get_supercategories()
    feature_to_concepts, concept_set = get_features_from_results()

    supercat_names = list(supercategories.keys())

    """Possible ways of evaluating best set overlap: 
        - intersection / feature extension (concept set) size: best_intersection
        - intersection / union: best_jacard
        - intersection / min: best_overlap_coeff
    Jacard and especially overlap coefficient get distracted by small supersets, which is not what we want:
    the question here is which supercategory is the biggest overlap for a given feature.
    """

    best_match_supercat = {}
    for feature in feature_to_concepts:
        feature_size = len(feature_to_concepts[feature])
        intersections = []
        # jacards = []  # intersection over union 
        # overlap_coeffs = []  # intersection over min
        for supercat in supercategories:
            supercat_concepts = supercategories[supercat]
            feature_concepts = feature_to_concepts[feature]
            intersection = len(feature_concepts & supercat_concepts)
            intersections.append(intersection)
            # jacard = len(common_concepts) / len(feature_concepts | supercat_concepts)
            # overlap_coeff = len(common_concepts) / min(len(feature_concepts), len(supercat_concepts))
            # jacards.append(jacard)
            # overlap_coeffs.append(overlap_coeff)
        best_intersection = max(intersections)
        best_intersection_idx = intersections.index(best_intersection)
        best_intersection_cat = supercat_names[best_intersection_idx]
        best_intersection_norm = best_intersection / feature_size
        # best_jacard = max(jacards)
        # best_jacard_idx = jacards.index(best_jacard)
        # best_jacard_cat = supercat_names[best_jacard_idx]
        # best_overlap_coeff = max(overlap_coeffs)
        # best_overlap_coeff_idx = overlap_coeffs.index(best_overlap_coeff)
        # best_overlap_coeff_cat = supercat_names[best_overlap_coeff_idx]

        # if best_jacard_cat != best_intersection_cat:
        #     print(f"Feature: {feature} best_intersection: {best_intersection_norm} {best_intersection_cat} best_jacard: {best_jacard} {best_jacard_cat} best_overlap_coeff: {best_overlap_coeff} {best_overlap_coeff_cat}")
        #     print(f"         concepts {feature_to_concepts[feature]}") 

        best_match_supercat[feature] = (best_intersection_cat, best_intersection_norm)
    
    return best_match_supercat


def main():
    best_match_supercat = get_best_supercategory_matches()
    bms_records = [(k, v[0], v[1]) for k, v in best_match_supercat.items()]
    df_features = pd.DataFrame.from_records(bms_records, columns=['feature', 'supercategory', 'intersection_norm'])

    for model in MODEL_LIST:
        model_scores = get_scores_from_results(model)
        df_scores = pd.DataFrame.from_records(list(model_scores.items()), columns=['feature', 'score'])
        df = pd.merge(df_features, df_scores, on='feature')
        # Pearson correlation between intersection_norm and score
        print(model, df[['intersection_norm', 'score']].corr())

if __name__ == '__main__':
    main()