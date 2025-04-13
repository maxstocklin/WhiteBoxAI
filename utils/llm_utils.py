# from mlx_lm import load, generate
import streamlit as st
import mlx.core as mx
import numpy as np
from mlx_lm import stream_generate

# Load Mistral model once
@st.cache_resource
def load_mistral():
    return load("mlx-community/Mistral-7B-Instruct-v0.2-4bit")

model, tokenizer = load_mistral()

def build_interpretation_prompt(sample_index, pred_label, pred_proba, df_summary):
    instruction = f"""
You are an expert in machine learning interpretability.

A binary classifier has made a prediction for sample #{sample_index}:
- ✅ **Predicted label**: {pred_label}
- 📈 **Confidence** (probability of label 1): {pred_proba:.2%}

Each feature below was used in the decision. You are provided with:
- The actual feature value
- A Z-score relative to others in the same class
- The decision tree threshold range the feature was involved in
- The SHAP value for that feature (measuring its impact on increasing or decreasing the probability of label 1)

**Important rule:**
- ➕ Positive SHAP value = pushed prediction toward label **1**
- ➖ Negative SHAP value = pushed prediction toward label **0**
- SHAP does **not** mean pushing toward the predicted class — just toward class 1 or 0.

---

Now analyze this prediction:
1. Give a short overview of how confident the model is and what class it leaned toward.
2. List the top 3 most influential features with short analysis.
3. Summarize your interpretation in bullet points.
4. Finish with your own 1–2 sentence assessment of whether this prediction seems reasonable.

---

**Feature Summary Table:**
"""

    table_lines = []
    for _, row in df_summary.iterrows():
        # Fix Z-score formatting safely
        z_raw = row.get("Z-Score (vs label)", "—")
        z = f"{z_raw:.2f}" if isinstance(z_raw, (int, float)) else str(z_raw)

        shap = row.get("SHAP", "—")
        shap_str = f"{shap:.3f}" if isinstance(shap, (int, float)) else str(shap)

        feature_name = row["Feature"].replace("_", " ").title()
        line = (
            f"- **{feature_name}**: value = {row['Value']}, "
            f"Z = {z}, "
            f"Tree condition = {row['Tree Range']}, "
            f"SHAP = {shap_str}"
        )
        table_lines.append(line)

    table = "\n".join(table_lines)

    prompt = f"<s>[INST]\n{instruction}\n{table}\n[/INST]"
    return prompt

def build_correlation_prompt(main_feature, paired_feature, binned_df):
    instruction = f"""
You are an expert in interpreting feature interaction effects from machine learning models.

We are analyzing how **{main_feature.replace('_', ' ').title()}** interacts with **{paired_feature.replace('_', ' ').title()}** to affect the model's predictions.

The table below shows how the interaction effect changes across different bins of **{main_feature.replace('_', ' ').title()}**.
Each row contains the average interaction strength between **{main_feature.replace('_', ' ').title()}** and **{paired_feature.replace('_', ' ').title()}** within that bin. 

Your task:
1. Describe how the interaction changes depending on the value of **{main_feature.replace('_', ' ').title()}**.
2. Identify any threshold or non-linear effects (e.g. “only matters above 40”, or “drops sharply after age 50”).
3. End with a 1–2 sentence assessment of what this means for model behavior and transparency.

---

**Top Interactions:**
"""

    table_lines = []
    for _, row in binned_df.iterrows():
        bin_label = str(row.get(main_feature, row.index if main_feature in row.index.names else "N/A"))
        interaction = row["Interaction"]
        table_lines.append(f"- {main_feature} = {bin_label}: interaction = {interaction:.4f}")

    return f"<s>[INST]\n{instruction}\n" + "\n".join(table_lines) + "\n[/INST]"

# === Softmax helper ===
def softmax(x):
    x = x - mx.max(x, axis=-1, keepdims=True)
    e_x = mx.exp(x)
    return e_x / mx.sum(e_x, axis=-1, keepdims=True)

# === Custom sampler with temperature / top-k / top-p ===
def sampler_with_temperature_topk_topp(temperature=1.0, top_k=None, top_p=None):
    def sampler(logits: mx.array) -> mx.array:
        if temperature != 1.0:
            logits = logits / temperature

        probs = softmax(logits)
        probs_np = np.array(probs).flatten()

        if top_k is not None and top_k > 0:
            top_k_indices = np.argpartition(-probs_np, top_k)[:top_k]
            mask = np.zeros_like(probs_np, dtype=bool)
            mask[top_k_indices] = True
            probs_np = np.where(mask, probs_np, 0)

        if top_p is not None and top_p < 1.0:
            sorted_indices = np.argsort(-probs_np)
            sorted_probs = probs_np[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff = cumulative_probs > top_p
            if np.any(cutoff):
                cutoff_index = np.argmax(cutoff)
                mask = np.zeros_like(probs_np, dtype=bool)
                mask[sorted_indices[:cutoff_index + 1]] = True
                probs_np = np.where(mask, probs_np, 0)

        probs_np /= probs_np.sum()
        token = np.random.choice(len(probs_np), p=probs_np)
        return mx.array([token])

    return sampler

# === Streaming generator with hallucination check ===
def generate_streaming_chunks(model, tokenizer, prompt, **kwargs):
    hold_chunk = None
    blocked_starts = {"user", "you", "you", "user", "input"}

    for response in stream_generate(model, tokenizer, prompt, **kwargs):
        chunk = response.text
        if hold_chunk:
            if chunk.strip() == ":":
                break
            else:
                yield hold_chunk
                yield chunk
                hold_chunk = None
            continue
        if chunk.strip().lower() in blocked_starts:
            hold_chunk = chunk
            continue
        yield chunk
