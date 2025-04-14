import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from utils.data_loader import load_data, preprocess, decode_features
from utils.model_utils import load_model_and_explainer
from sklearn.model_selection import train_test_split
# from utils.simulator_utils import find_minimal_flip
from utils.counterfactuals import load_dice_explainer, generate_counterfactuals

st.set_page_config(layout="wide")
st.title("🎚 Real-Time Simulator")

# === Load data and model ===
df = load_data("data/adult.data")
X, y, encoders = preprocess(df)
model, explainer = load_model_and_explainer(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_human = decode_features(X_test, encoders)

categorical_columns = list(encoders.keys())

# === Sample selector ===
sample_index = st.sidebar.number_input("Select sample ID", min_value=0, max_value=len(X_test)-1, value=0)

original_sample = X_test_human.iloc[sample_index]

# 🧠 Reset slider state if new sample selected
if "last_sample_index" not in st.session_state:
    st.session_state.last_sample_index = sample_index

if sample_index != st.session_state.last_sample_index:
    for col in X.select_dtypes(include=[np.number]).columns:
        if col in st.session_state:
            del st.session_state[col]
    st.session_state.last_sample_index = sample_index

original_encoded = X_test.iloc[sample_index]
original_human = X_test_human.iloc[sample_index]

true_label = y_test[sample_index]

with st.expander("🔁 Flip-the-Prediction Generator", expanded=True):
    st.subheader("What Small Change Could Flip the Prediction?")
    st.markdown("""
This tool explores **realistic, minimal changes** to the input that would cause the model to make a **different decision**.  
It helps reveal what factors are **most critical** in shaping the prediction — and how flexible the outcome is.

_Powered by DiCE._
""")

    with st.spinner("Loading counterfactual engine..."):
        cf_explainer = load_dice_explainer(model, X_train, y_train)

    # Show previously stored CF
    cf_df = st.session_state.get("generated_cf", None)
    if cf_df is not None:
        st.markdown("#### 🔁 Close Valid Counterfactual")
        # st.dataframe(cf_df)

        for i, row in cf_df.iterrows():
            # st.markdown(f"**CF #{i+1}:**")
            changes = []
            for col in X.columns:
                orig_val = original_encoded[col]
                new_val = row[col]
                if new_val != orig_val:
                    if col in categorical_columns:
                        orig_decoded = encoders[col].inverse_transform([orig_val])[0]
                        new_decoded = encoders[col].inverse_transform([new_val])[0]
                    else:
                        orig_decoded = orig_val
                        new_decoded = new_val
                    changes.append(f"- **{col.replace('_', ' ').title()}**: {orig_decoded} → {new_decoded}")
            st.markdown("\n".join(changes) if changes else "✅ No changes needed to flip prediction.")

    if st.button("Generate Counterfactuals"):
        try:
            for _ in range(10):
                cf_df = generate_counterfactuals(
                    original_encoded,
                    cf_explainer,
                    total=1,
                    features_to_vary="all",
                    proximity_weight=0.5,
                    diversity_weight=0.1,
                    desired_class="opposite"
                )
                if cf_df is not None and not cf_df.empty:
                    st.session_state["generated_cf"] = cf_df
                    break

            cf_df = st.session_state.get("generated_cf", None)

            valid_cf = None
            for _, row in cf_df.iterrows():
                if any(row[col] != original_encoded[col] for col in X.columns):
                    valid_cf = row
                    break

            if valid_cf is not None:
                # st.markdown("#### 🔁 Close Valid Counterfactual")
                changes = []
                for col in X.columns:
                    orig_val = original_encoded[col]
                    new_val = valid_cf[col]
                    if new_val != orig_val:
                        if col in categorical_columns:
                            orig_decoded = encoders[col].inverse_transform([orig_val])[0]
                            new_decoded = encoders[col].inverse_transform([new_val])[0]
                        else:
                            orig_decoded = orig_val
                            new_decoded = new_val
                        changes.append(f"- **{col.replace('_', ' ').title()}**: {orig_decoded} → {new_decoded}")
                # st.markdown("\n".join(changes))
                st.rerun()
            else:
                raise ValueError("No valid counterfactuals found")

        except Exception:
            st.warning("⚠️ No realistic counterfactuals could be found under the current constraints.")
            st.markdown("""
                This means the model is very confident about this prediction,
                and even when we tried small, realistic changes to key features, 
                it still predicted the same outcome.
            """)





with st.expander("🔧 Feature Control Lab"):
    st.subheader("Adjust Inputs Features to Simulate a New Scenario")
    st.markdown("""
This tool lets you **manually adjust input values** to explore how the model reacts.  
Use this panel to answer:  
_"What if this person had a different education level, or worked more hours?"_
""")

    col1, col2 = st.columns(2)
    with col1:
        if "constraint_mode" not in st.session_state:
            st.session_state["constraint_mode"] = "Realistic Constraints"

        constraint_mode = st.pills(
            "Constraints Mode", 
            options=["Realistic Constraints", "Full Range"],
            key="constraint_mode"
        )
    with col2:
        if st.button("🔄 Revert to Original Values"):
            for col in X.columns:
                num_key = f"{col}_{sample_index}"
                cat_key = f"{col}_cat_{sample_index}"
                if num_key in st.session_state:
                    del st.session_state[num_key]
                if cat_key in st.session_state:
                    del st.session_state[cat_key]
                st.rerun()
    num_col, cat_col = st.columns(2)
    modified_encoded = original_encoded.copy()
    with num_col:
        st.markdown("### 🔢 Numerical Features")
        for col in X.select_dtypes(include=[np.number]).columns:
            if col in categorical_columns:
                continue  # skip encoded categorical columns

            orig_val = int(original_encoded[col])
            key = f"{col}_{sample_index}"
            current_val = st.session_state.get(key, orig_val)

            if constraint_mode == "Realistic Constraints":
                if col == "age":
                    min_val, max_val = max(0, orig_val - 5), min(100, orig_val + 5)
                elif col == "hours_per_week":
                    min_val, max_val = max(1, orig_val - 10), min(100, orig_val + 10)
                elif col == "education_num":
                    min_val, max_val = max(1, orig_val - 2), min(16, orig_val + 2)
                elif col == "capital_gain":
                    min_val, max_val = 0, orig_val + 5000
                elif col == "capital_loss":
                    min_val, max_val = 0, orig_val + 1000
                elif col == "fnlwgt":
                    min_val, max_val = int(orig_val * 0.9), int(orig_val * 1.1)
                else:
                    min_val, max_val = int(X[col].min()), int(X[col].max())
            else:
                min_val, max_val = int(X[col].min()), int(X[col].max())

            modified_encoded[col] = st.slider(
                f"{col.replace('_', ' ').title()} (original: {orig_val})",
                min_value=min_val,
                max_value=max_val,
                value=current_val,
                key=key
            )
    with cat_col:
        st.markdown("### 🏷️ Categorical Features")
        for col in categorical_columns:
            if col not in original_encoded:
                continue
            decoder = encoders[col]
            orig_val_encoded = original_encoded[col]
            orig_val = decoder.inverse_transform([orig_val_encoded])[0]
            options = list(decoder.classes_)
            key = f"{col}_cat_{sample_index}"
            current_val = st.session_state.get(key, orig_val)

            selected = st.selectbox(
                f"{col.replace('_', ' ').title()} (original: {orig_val})",
                options,
                index=options.index(current_val),
                key=key
            )
            modified_encoded[col] = decoder.transform([selected])[0]
    # Updated prediction
    pred_label = model.predict(pd.DataFrame([modified_encoded]))[0]
    pred_proba = model.predict_proba(pd.DataFrame([modified_encoded]))[0][1]

    # st.markdown("### 🤖 New Prediction")
    # st.write(f"**Predicted label:** `{pred_label}`")
    # st.write(f"**Probability of label 1:** {pred_proba:.2%}")
    
# === SHAP delta ===
shap_orig = explainer.shap_values(pd.DataFrame([original_encoded]))[0]
shap_new = explainer.shap_values(pd.DataFrame([modified_encoded]))[0]
delta_shap = shap_new - shap_orig

delta_df = pd.DataFrame({
    "Feature": [col.replace("_", " ").title() for col in X.columns],
    "Δ SHAP": delta_shap
}).sort_values("Δ SHAP", key=abs, ascending=False)

with st.expander("🔍 Feature Impact Tracker"):
    st.subheader("How Did Your Changes Influence the Prediction?")
    st.markdown("""
This section tracks how your edits impacted the model's decision:  
- See which features had the biggest impact change
- Review the exact values you modified
- Detect if the outcome flipped — and why
""")

    fig, ax = plt.subplots(figsize=(6, 3))
    delta_df.set_index("Feature")["Δ SHAP"].plot(kind="barh", color="skyblue", ax=ax)
    ax.axvline(0, color='black', linestyle='--')
    st.pyplot(fig)

    # === Counterfactual Generator ===



    # === Delta View ===
    st.subheader("Feature Differences")

    diffs = []
    for col in X.columns:
        original_val = original_encoded[col]
        new_val = modified_encoded[col]
        if original_val != new_val:
            if col in categorical_columns:
                orig_decoded = encoders[col].inverse_transform([original_val])[0]
                new_decoded = encoders[col].inverse_transform([new_val])[0]
                diffs.append((col.replace("_", " ").title(), orig_decoded, new_decoded))
            else:
                diffs.append((col.replace("_", " ").title(), original_val, new_val))

    if diffs:
        delta_feat_df = pd.DataFrame(diffs, columns=["Feature", "Original", "Modified"])
        st.dataframe(delta_feat_df, use_container_width=True)
    else:
        st.info("No feature has been changed yet.")




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


original_pred_label = model.predict(pd.DataFrame([original_encoded]))[0]
original_proba = model.predict_proba(pd.DataFrame([original_encoded]))[0][1]

animate_flip = (original_pred_label != true_label) and (pred_label == true_label)



if pred_label == true_label:
    # Correct prediction → green
    footer_color = "#2e7d32"  # green
if pred_label != true_label:
    # Incorrect prediction → red
    footer_color = "#c62828"  # red
if pred_label == original_pred_label and abs(pred_proba - original_proba) > 0.1:
    # Confidence changed but label same → yellow
    footer_color = "#f9a825"  # yellow


st.markdown(f"""
<div class="sticky-footer" id="footer" style="background-color: {footer_color};">
    🎯 <b>True Label:</b> {true_label} |
    🤖 <b>Original:</b> {original_pred_label} ({original_proba:.2%}) →
    <b>New:</b> {pred_label} ({pred_proba:.2%})
</div>
""", unsafe_allow_html=True)


if animate_flip:
    st.markdown("""
    <script>
    setTimeout(() => {
        const footer = document.getElementById("footer");
        footer.style.animation = "correctFlip 1.5s ease-in-out forwards";
    }, 100); // short delay to let red show first
    </script>
    """, unsafe_allow_html=True)
