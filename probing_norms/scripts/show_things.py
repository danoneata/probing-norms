import pdb
from pathlib import Path
from torchvision.datasets import ImageNet

import streamlit as st
import pandas as pd

df_concept_mapping = pd.read_csv("data/concept-mapping.csv")
mcrae_to_things = dict(df_concept_mapping[["mcrae_concept", "things_concept_id"]].values)


ROOT_THINGS = Path("/mnt/private-share/speechDatabases/THINGS")
df = pd.read_csv("data/norms/mcrae/CONCS_FEATS_concstats_brm.txt", sep="\t")
concepts = df["Concept"].unique()
features1 = df["Feature"].unique()

df_gpt_norms = pd.read_csv("/home/doneata/work/semantic-features-gpt-3/data/gpt_3_feature_norm/mcrae_priming/gpt4/all_things_concepts/30/decoded_answers.csv")
features2 = df_gpt_norms["decoded_feature"].unique()
features2 = [f.replace(" ", "_") for f in features2]

common = set(features1) & set(features2)
diff1 = set(features1) - set(features2)
diff2 = set(features2) - set(features1)

print(len(features1))
print(len(features2))
print(len(common))
print(len(diff1))
print(len(diff2))

pdb.set_trace()

# armour --> armor
# bat_(animal) --> bat1
# bat_(baseball) --> bat2
# baton --> baton{1,2,3,4}?

missing_concepts = []

for concept_mcrae in concepts:
    concept = mcrae_to_things.get(concept_mcrae, concept_mcrae)
    img_folder = ROOT_THINGS / "object_images" / concept
    # st.markdown(concept)
    try:
        img_path, *_ = list(img_folder.iterdir())
        # st.image(str(img_path))
    except:
        # st.write(f"Could not find images for `{concept}`")
        missing_concepts.append(concept)

st.write("Missing concepts:")
st.write(len(missing_concepts))
st.write(len(concepts))
st.write(missing_concepts)
