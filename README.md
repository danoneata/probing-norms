This repository contains code for the paper:

> Oneata, Dan, Desmond Elliott, and Stella Frank.
> [Seeing What Tastes Good: Revisiting Multimodal Distributional Semantics in the Billion Parameter Era.](https://arxiv.org/abs/2506.03994)
> Findings ACL, 2025.

## Data

The paper introduces the dataset McRae × ᴛʜɪɴɢs, which is obtained by labelling the concepts in the ᴛʜɪɴɢs dataset with the McRae attributes.
This data is available in the [`data/mcrae-x-things.json`](data/mcrae-x-things.json) file, which contains a list of positive concept–attribute pairs.
For the type corresponding to each attribute, see the [`data/mcrae-x-things-taxonomy.json`](data/mcrae-x-things-taxonomy.json) file.

## Setup

The code uses PyTorch (for feature extraction), which we recommend installing via conda:

```bash
conda create -n probing-norms python=3.12
conda activate probing-norms
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Then you can install this code as a library with:

```bash
pip install -e .
```

## Evaluate a single model

To evaluate a single model (let's say the Swin-V2 model, `swin-v2-ssl`), you have to follow the next steps.

**Feature extraction.**
Extract the features for the ᴛʜɪɴɢs concepts:
```bash
python probing_norms/extract_features_image.py -d things -f swin-v2-ssl
```
For a language model, you need to use the `probing_norms/extract_features_text.py` script instead.

**Model training.**
Train linear probes and predict for both the McRae × ᴛʜɪɴɢs and the Binder datasets:
```bash
python probing_norms/predict.py --feature-type swin-v2-ssl --norms-type mcrae-x-things --split-type repeated-k-fold --embeddings-level concept --classifier-type linear-probe
python probing_norms/predict.py --feature-type swin-v2-ssl --norms-type binder-dense --split-type repeated-k-fold-simple --embeddings-level concept --classifier-type linear-regression
```
For McRae × ᴛʜɪɴɢs the probe is a linear classifier, while for Binder it is a linear regression.
The split also differs between the two datasets: for McRae × ᴛʜɪɴɢs we use a stratified repeated k-fold split, while for Binder we don't have to use stratification.

**Results generation.**
To evaluate in terms of the corresponding metrics (F₁ selectivity for McRae × ᴛʜɪɴɢs and RMSE for Binder), run the following command:
```bash
python probing_norms/get_results.py paper-table-main-acl-camera-ready:swin-v2-ssl
```

## Replicating the results in the paper

TODO