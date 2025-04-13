import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import shap, json
from pathlib import Path

st.set_page_config(layout="wide")

from utils.data_loader import load_data, preprocess, decode_features
from utils.model_utils import load_model_and_explainer
from utils.interpret_utils import get_feature_path_ranges, get_used_features, get_confidence_report
from utils.llm_utils import (
    build_interpretation_prompt,
    generate_streaming_chunks,
    sampler_with_temperature_topk_topp,
    load
)

st.title("🔍 Sample Interpretation")

# === Load data and model ===
df = load_data("data/adult.data")
X, y, encoders = preprocess(df)
categorical_columns = df.select_dtypes(include="object").columns.tolist()

model, explainer = load_model_and_explainer(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_human = decode_features(X_test, encoders)

# === Get sample index from query param
params = st.query_params
sample_index_url = int(params.get("id", 0))

sample_index = st.sidebar.number_input("Select sample ID", min_value=0, max_value=len(X_test)-1, value=sample_index_url, step=1)

if sample_index >= len(X_test):
    st.error(f"❌ Invalid sample index: {sample_index}. Max allowed is {len(X_test) - 1}.")
    st.stop()

st.markdown(f"Interpreting sample ID: `{sample_index}`")

# === Use encoded for modeling, decoded for display
sample_df_encoded = X_test.iloc[[sample_index]]
sample_df_human = X_test_human.iloc[[sample_index]]
sample_encoded = sample_df_encoded.iloc[0]
sample_human = sample_df_human.iloc[0]

true_label = y_test[sample_index]
pred_label = model.predict(sample_df_encoded)[0]
pred_proba = model.predict_proba(sample_df_encoded)[0][1]

confidence_report, confidence_level = get_confidence_report(
    model, 
    X_train, 
    y_train, 
    sample_df_encoded, 
    pred_label
)

with st.expander("🔎 Model Confidence Report", expanded=True):
    st.markdown(f"**Overall Confidence Level:** {confidence_level}")
    st.dataframe(confidence_report, use_container_width=True)

# === Cache path
cache_path = Path(f"logs/explanations/sample_{sample_index}.json")

with st.expander("📂 Feature-Level Explanation", expanded=False):
    feature_ranges = get_feature_path_ranges(model, sample_df_encoded, encoders, categorical_columns)

    used_features = list(feature_ranges.keys())

    shap_vals = explainer.shap_values(sample_df_encoded)[0]
    shap_dict = dict(zip(X.columns, shap_vals))
    summary_df = get_used_features(
        feature_ranges,
        used_features,
        X_train,
        y_train,
        sample_encoded,
        true_label,
        shap_dict,
        encoders,
        categorical_columns
    )
    st.dataframe(summary_df, use_container_width=True)

with st.expander("🧠 Assistant Assessment by Mistral", expanded=False):

    # === Cached or generate?
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)

        st.success(f"✅ Loaded cached interpretation for sample {sample_index}")
        summary_df = pd.DataFrame(cached["summary_df"])
        prompt = cached["llm_prompt"]
        explanation = cached["llm_response"]
        pred_label = cached["predicted_label"]
        pred_proba = cached["predicted_proba"]
        true_label = cached["true_label"]

    else:
        st.warning("🔄 No cache found. Running LLM interpretation...")

        prompt = build_interpretation_prompt(sample_index, pred_label, pred_proba, summary_df)

        try:
            model_mlx, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.2-4bit")
        except Exception as e:
            st.warning(f"⚠️ LLM interpretation currently not available for sample {sample_index}. Coming soon!")
            st.stop()

        sampler = sampler_with_temperature_topk_topp(temperature=0.7, top_k=40, top_p=0.9)

        explanation_chunks = []
        for chunk in generate_streaming_chunks(model_mlx, tokenizer, prompt, max_tokens=5000, sampler=sampler):
            explanation_chunks.append(chunk)
        explanation = "".join(explanation_chunks)

        output = {
            "sample_index": sample_index,
            "true_label": int(true_label),
            "predicted_label": int(pred_label),
            "predicted_proba": float(pred_proba),
            "used_features": used_features,
            "summary_df": summary_df.to_dict(orient="records"),
            "llm_prompt": prompt,
            "llm_response": explanation
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        st.success(f"✅ Interpretation complete and saved to cache")


    st.markdown(explanation)





# === 1. Sticky footer CSS and animation definition (declare only once)
st.markdown("""
<style>
@keyframes correctFlip {
    0% { background-color: #c62828; }
    100% { background-color: #2e7d32; }
}

.sticky-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    color: white;
    padding: 15px 30px;
    font-size: 18px;
    font-weight: 600;
    text-align: center;
    border-top: 2px solid #222;
    z-index: 9999;
    box-shadow: 0 -2px 10px rgba(0,0,0,0.3);
    border-radius: 10px 10px 0 0;
}
</style>
""", unsafe_allow_html=True)

# === Display section ===
decoded_label = encoders["income"].inverse_transform([pred_label])[0]
decoded_true = encoders["income"].inverse_transform([true_label])[0]

if confidence_level == "🟢 High":
    footer_color = "#2e7d32"  # green
if confidence_level == "🟡 Medium":
    footer_color = "#ff9800"  # orange
if confidence_level == "🔴 Low":
    footer_color = "#c62828"  # red

st.markdown(f"""
<div class="sticky-footer" id="footer" style="background-color: {footer_color};">
    🎯 <b>True Label:</b> {true_label} |
    <b>Predicted Label:</b> {pred_label} ({pred_proba:.2%}) | 
    <b>Confidence Level:</b> {confidence_level}
</div>
""", unsafe_allow_html=True)
