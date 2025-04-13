import pandas as pd
import numpy as np
import shap
from pathlib import Path
from sklearn.model_selection import train_test_split
from data_loader import load_data, preprocess, decode_features
from model_utils import load_model_and_explainer
from llm_utils import build_correlation_prompt, generate_streaming_chunks, sampler_with_temperature_topk_topp, load

# === Paths ===
LOG_DIR = Path("logs/interactions_explorer")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# === Load Data/Model ===
df = load_data("data/adult.data")
X, y, encoders = preprocess(df)
model, _ = load_model_and_explainer(X, y)
X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
X_sample = X_train.sample(500, random_state=42)
X_sample_human = decode_features(X_sample, encoders)
feature_names = list(X.columns)

# === SHAP interaction values ===
explainer = shap.TreeExplainer(model)
interaction_values = explainer.shap_interaction_values(X_sample)

def get_top_interactions(interaction_values, feature_idx):
    mean_interactions = np.abs(interaction_values[:, feature_idx, :]).mean(axis=0)
    top_idx = np.argsort(-mean_interactions)
    return [i for i in top_idx if i != feature_idx][:3]

def get_feature_bins(feature_name, values, bins=6):
    try:
        return pd.cut(values, bins=bins).astype(str)
    except Exception:
        return values.astype(str)

def get_binned_interactions(X_sample, interaction_values, feature_a, feature_b, bins=6):
    fa_idx = X_sample.columns.get_loc(feature_a)
    fb_idx = X_sample.columns.get_loc(feature_b)
    interaction_vals = interaction_values[:, fa_idx, fb_idx]
    bin_labels = get_feature_bins(feature_b, X_sample[feature_b])
    df = pd.DataFrame({
        f"{feature_b} Bin": bin_labels,
        "Interaction": interaction_vals
    })
    df[f"{feature_b} Bin"] = df[f"{feature_b} Bin"].astype(str)
    df = df.groupby(f"{feature_b} Bin").mean().reset_index()
    return df

# === Load LLM ===
model_mlx, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.2-4bit")
sampler = sampler_with_temperature_topk_topp(temperature=0.7, top_k=40, top_p=0.9)

# === Loop over all features ===
for idx, main_feature in enumerate(feature_names):
    top_3 = get_top_interactions(interaction_values, idx)
    for paired_idx in top_3:
        paired_feature = feature_names[paired_idx]

        binned_cache = LOG_DIR / f"binned_{main_feature}_{paired_feature}.json"
        llm_cache = LOG_DIR / f"llm_{main_feature}_{paired_feature}.txt"

        if binned_cache.exists() and llm_cache.exists():
            continue  # Already cached

        try:
            binned_df = get_binned_interactions(X_sample_human, interaction_values, main_feature, paired_feature)
            prompt = build_correlation_prompt(
                main_feature=main_feature,
                paired_feature=paired_feature,
                binned_df=binned_df
            )

            explanation_chunks = []
            for chunk in generate_streaming_chunks(
                model_mlx, tokenizer, prompt, max_tokens=1000, sampler=sampler
            ):
                explanation_chunks.append(chunk)
            explanation = "".join(explanation_chunks)

            binned_cache.write_text(binned_df.to_json(orient="records"), encoding="utf-8")
            llm_cache.write_text(explanation, encoding="utf-8")
            print(f"✅ Saved: {main_feature} × {paired_feature}")
        except Exception as e:
            print(f"❌ Failed: {main_feature} × {paired_feature} → {e}")