import pdb

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from probing_norms.data import load_binder_feature_norms, load_binder_dense
from probing_norms.utils import read_file


df = load_binder_dense()
words = sorted(set(df["Word"].tolist()))
col_order = df.groupby("Feature")["Value"].median().sort_values(ascending=False).index.tolist()
fig = sns.displot(df, x="Value", kind="hist", col="Feature", col_wrap=6, bins=12, col_order=col_order, height=2)
for ax in fig.axes.flat:
    _, feature = ax.get_title().split(' = ')
    idxs = df["Feature"] == feature
    median_val = df[idxs]["Value"].median()
    ax.axvline(median_val, color='red', linestyle='--')
st.pyplot(fig)

# concept_feature = load_binder_feature_norms(5)
# pdb.set_trace()

# concepts_map = {
#     "cab": "taxi",
#     "camera": "camera1",
#     "chicken": "chicken2",
#     "mouse": "mouse1",
#     "shelves": "shelf",
# }

# df = pd.read_excel("data/binder-norms.xlsx")
# cols = df.columns[5: 70]
# supercats = [
#     "living object",
#     "artifact",
#     "natural object",
# ]
# idxs = df["Super Category"].isin(supercats)
# concepts1 = df[idxs]["Word"].tolist()
# concepts2 = read_file("data/concepts-things.txt")
# concepts_inter = set(concepts1) & set(concepts2)
# concepts_missing = set(concepts1) - set(concepts2)
# print(sorted(concepts_inter))
# print(len(concepts_inter))
# print()
# print(sorted(concepts_missing))
# print(len(concepts_missing))
# print()
# pdb.set_trace()