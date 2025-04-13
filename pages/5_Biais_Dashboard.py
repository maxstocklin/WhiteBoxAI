import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from utils.data_loader import load_data, preprocess, decode_features
from utils.model_utils import load_model_and_explainer
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")

st.title("⚖️ Biais Dashboard")

# === Load data, model, and SHAP explainer ===
df = load_data("data/adult.data")
X, y, encoders = preprocess(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_human = decode_features(X_test, encoders)
model, explainer = load_model_and_explainer(X, y)

X_test_human["Label"] = y_test

with st.expander("🧪 Data Integrity Checks"):
    st.markdown("#### 📊 Category vs. Label Crosstab")
    label_map = {0: "<=50K", 1: ">50K"}
    X_test_human["Label_Name"] = X_test_human["Label"].map(label_map)

    for cat_col in ["sex", "race", "workclass", "education", "marital_status", "occupation", "relationship", "native_country"]:
        st.markdown(f"##### `{cat_col}`")
        crosstab = pd.crosstab(X_test_human[cat_col], X_test_human["Label_Name"])
        # Add percentage of >50K column
        if ">50K" in crosstab.columns:
            crosstab["% >50K"] = ((crosstab[">50K"] / crosstab.sum(axis=1)) * 100).round(1).astype(str) + "%"
        else:
            crosstab["% >50K"] = "0.0%"
        st.dataframe(crosstab)

        numeric_crosstab = crosstab.select_dtypes(include=[np.number])
        rare_mask = numeric_crosstab.sum(axis=1) < 10
        rare_values = crosstab[rare_mask]

        if not rare_values.empty:
            st.warning(f"⚠️ Rare categories in `{cat_col}` (fewer than 10 examples):")
            st.dataframe(rare_values)

        single_class_mask = (numeric_crosstab == 0).any(axis=1)
        biased_values = crosstab[single_class_mask]

        if not biased_values.empty:
            st.error(f"🚨 Some categories in `{cat_col}` only appear with one label:")
            st.dataframe(biased_values)



st.markdown("### 📊 Fairness Metrics")
sensitive_attr = st.pills("Choose sensitive feature", options=["sex", "race"])

if sensitive_attr:
    # === SHAP disparity by group ===
    with st.expander("📊 SHAP Value Disparities"):

        with st.spinner("Calculating SHAP values..."):
            shap_vals = explainer.shap_values(X_test)
            shap_df = pd.DataFrame(shap_vals, columns=X.columns)
            shap_df[sensitive_attr] = X_test_human[sensitive_attr].values

            group_avg = shap_df.groupby(sensitive_attr).mean().T

        fig, ax = plt.subplots(figsize=(10, 6))
        group_avg.plot(kind="barh", ax=ax)
        ax.set_title(f"Average SHAP Value by Feature and {sensitive_attr.title()}")
        st.pyplot(fig)


    # === Prediction distribution by group ===
    with st.expander("🔍 Prediction Distribution"):

        preds = model.predict(X_test)
        X_test_human["Prediction"] = preds

        label_map = {0: "<=50K", 1: ">50K"}
        X_test_human["Prediction_Label"] = X_test_human["Prediction"].map(label_map)
        X_test_human["Label_Name"] = X_test_human["Label"].map(label_map)

        group_pred_dist = X_test_human.groupby([sensitive_attr, "Prediction_Label"]).size().unstack().fillna(0)
        group_pred_percent = group_pred_dist.div(group_pred_dist.sum(axis=1), axis=0) * 100
        group_pred_combined = group_pred_dist.copy()

        # for col in group_pred_percent.columns:
        #     group_pred_combined[f"% {col}"] = group_pred_percent[col].round(1).astype(str) + "%"

        # st.dataframe(group_pred_combined)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        group_pred_dist.plot(kind="bar", stacked=True, ax=ax2)
        ax2.set_title(f"Prediction Distribution by {sensitive_attr.title()}")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)



    with st.expander("📐 Fairness Metrics: Prediction vs. Ground Truth"):

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🏷️ Actual Disparity")
            group_label_rate = X_test_human.groupby(sensitive_attr)["Label"].mean()
            group_label_rate_display = group_label_rate.apply(lambda x: f"{x:.2%}")
            label_diff = group_label_rate.max() - group_label_rate.min()

            for group, rate in group_label_rate_display.items():
                st.write(f"- **{group}**: {rate}")
            st.markdown(f"**Label Disparity:** `{label_diff:.2%}`")
            if label_diff > 0.2:
                st.error("⚠️ High disparity in actual labels!")
            elif label_diff > 0.1:
                st.warning("⚠️ Moderate disparity in actual labels.")
            else:
                st.success("✅ Low disparity in actual labels.")

        with col2:
            st.markdown("#### 🔮 Model Predictions")
            preds = model.predict(X_test)
            X_test_human["Prediction"] = preds
            group_pred_rate = X_test_human.groupby(sensitive_attr)["Prediction"].mean()
            group_pred_rate_display = group_pred_rate.apply(lambda x: f"{x:.2%}")
            dp_diff = group_pred_rate.max() - group_pred_rate.min()

            for group, rate in group_pred_rate_display.items():
                st.write(f"- **{group}**: {rate}")
            st.markdown(f"**Demographic Parity Difference:** `{dp_diff:.2%}`")
            if dp_diff > 0.2:
                st.error("⚠️ High disparity detected!")
            elif dp_diff > 0.1:
                st.warning("⚠️ Moderate disparity detected.")
            else:
                st.success("✅ Low disparity.")

        # === Comparison Summary: Model vs. Ground Truth ===
        st.subheader("📉 Disparity Comparison Summary")

        if dp_diff > label_diff * 1.2:
            st.error("⚠️ The model's prediction disparity is higher than the label disparity — potential bias introduced.")
        elif dp_diff < label_diff:
            st.info("ℹ️ The model has reduced the disparity compared to the ground truth — possible bias mitigation.")
        else:
            st.success("✅ The model's disparity matches the ground truth.")
