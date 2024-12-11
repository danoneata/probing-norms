Set-up:
```bash
pip install -e .
```

Extract image features:
```bash
python probing_norms/extract_features_image.py
```

Train linear probes and predict:
```bash
python probing_norms/predict.py
```

Visualize results:
```bash
streamlit run probing_norms/scripts/show_results_per_feature_norm.py
```