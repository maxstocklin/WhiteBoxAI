import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap
import time
from data_loader import decode_features

from data_loader import load_data, preprocess
from interpret_utils import get_feature_path_ranges, get_used_features
from llm_utils import (
    build_interpretation_prompt,
    generate_streaming_chunks,
    sampler_with_temperature_topk_topp,
    load as load_mlx,
)

# === Config ===
DATA_PATH = "data/adult.data"
OUT_DIR = Path("logs/explanations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === Load everything once ===
df = load_data(DATA_PATH)
X, y, encoders = preprocess(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train model once ===
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# === Load LLM once ===
model_mlx, tokenizer = load_mlx("mlx-community/Mistral-7B-Instruct-v0.2-4bit")
sampler = sampler_with_temperature_topk_topp(temperature=0.7, top_k=40, top_p=0.9)

categorical_columns = list(encoders.keys())
X_test_human = decode_features(X_test, encoders)

# Generate explanations for samples 30-60
for idx in tqdm(range(60, 100), desc="Generating explanations"):
    
# for idx in tqdm(range(30), desc="Generating explanations"):
    output_path = OUT_DIR / f"sample_{idx}.json"
    if output_path.exists():
        continue

    # Throttle execution slightly to reduce CPU strain
    time.sleep(1)

    sample_df = X_test.iloc[[idx]]
    sample = sample_df.iloc[0]
    sample_human = X_test_human.iloc[idx]
    true_label = int(y_test[idx])
    pred_label = int(model.predict(sample_df)[0])
    pred_proba = float(model.predict_proba(sample_df)[0][1])

    # === Extract ranges and SHAP
    feature_ranges = get_feature_path_ranges(model, sample_df, encoders, categorical_columns)
    used_features = list(feature_ranges.keys())

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(sample_df)[0]
    shap_dict = dict(zip(X.columns, shap_vals))

    # === Build summary
    summary_df = get_used_features(
        feature_ranges,
        used_features,
        X_train,
        y_train,
        sample,
        true_label,
        shap_dict,
        encoders,
        categorical_columns
    )

    prompt = build_interpretation_prompt(idx, pred_label, pred_proba, summary_df)

    explanation_chunks = []
    for chunk in generate_streaming_chunks(
        model_mlx,
        tokenizer,
        prompt,
        max_tokens=800,
        sampler=sampler
    ):
        explanation_chunks.append(chunk)
    explanation = "".join(explanation_chunks)

    output = {
        "sample_index": idx,
        "true_label": true_label,
        "predicted_label": pred_label,
        "predicted_proba": pred_proba,
        "used_features": used_features,
        "summary_df": summary_df.to_dict(orient="records"),
        "llm_prompt": prompt,
        "llm_response": explanation
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)