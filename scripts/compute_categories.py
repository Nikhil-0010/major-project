# scripts/compute_categories.py
import json
import pandas as pd
from pathlib import Path

DATA = Path("../artifacts/data/train.csv")   # or ../data/heart_uci_clean.csv if you haven't made train csv
OUT = Path("../artifacts/preprocessor/categories_list.json")

df = pd.read_csv(DATA)
# list of categorical feature names — must match your transforms.py lists:
categorical_features = ['sex','cp','fbs','restecg','exang','slope','thal']

categories_list = []
for c in categorical_features:
    # sort for determinism and convert bool->str if mixed type
    vals = df[c].dropna().unique().tolist()
    # convert booleans to strings for consistent dtypes if needed
    cleaned = [str(v) if isinstance(v, bool) else v for v in vals]
    categories_list.append(sorted(map(str, cleaned)))  # ensure strings where needed

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf8") as f:
    json.dump({"categorical_features": categorical_features, "categories_list": categories_list}, f, indent=2)
print("Saved:", OUT)
