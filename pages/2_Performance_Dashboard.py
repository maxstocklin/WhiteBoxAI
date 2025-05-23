import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score

)
import shap

st.set_page_config(layout="wide")

from utils.model_utils import load_model_and_explainer
from utils.shap_utils import create_interaction_plot, get_feature_bins
from utils.data_loader import load_data, preprocess, decode_features
from utils.feature_info import FEATURE_DESCRIPTIONS
from pathlib import Path
import json

st.title("Performance Dashboard")
# st.caption("Review how well the model is doing overall.")

# === Train model ===
@st.cache_resource(show_spinner=True)
def train_model(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

# === Load, preprocess, split ===
df = load_data("data/adult.data")
X, y, encoders = preprocess(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = train_model(X_train, y_train)

# === Evaluation ===
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Classification Report
report = classification_report(y_test, y_pred, output_dict=True, target_names=["<=50K", ">50K"])
accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred) * 100
recall = recall_score(y_test, y_pred) * 100
f1 = f1_score(y_test, y_pred) * 100

with st.expander("📊 Classification Report", expanded=True):
    st.subheader("Performance-wise Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 **Accuracy**")
        st.markdown("_How often the model was right overall._")
        st.markdown(f"<h2 style='color:#4CAF50'>{accuracy:.2f}%</h2>", unsafe_allow_html=True)

        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("### 📏 **Precision**")
        st.markdown("_Of the ones we said were positive, how many really were?_")
        st.markdown(f"<h2 style='color:#2196F3'>{precision:.2f}%</h2>", unsafe_allow_html=True)

    with col2:
        st.markdown("### 🔁 **Recall**")
        st.markdown("_Of all the actual positives, how many did we find?_")
        st.markdown(f"<h2 style='color:#FFC107'>{recall:.2f}%</h2>", unsafe_allow_html=True)

        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("### ⚖️ **F1 Score**")
        st.markdown("_A balance between precision and recall._")
        st.markdown(f"<h2 style='color:#9C27B0'>{f1:.2f}%</h2>", unsafe_allow_html=True)

    st.markdown("---")
    st.info("✅ **Tip:** Use **precision** when false alarms are bad (e.g., fraud detection), and **recall** when missing real cases is worse (e.g., disease diagnosis).")

with st.expander("📈 Confusion Matrix", expanded=False):
    # Confusion Matrix
    st.subheader("See how often the model predicts correctly vs. mistakenly — split by actual and predicted labels.")

    st.markdown("""The confusion matrix shows how well the model distinguishes between the two classes:""")

    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["<=50K", ">50K"],
                yticklabels=["<=50K", ">50K"],
                ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

# ROC Curve
if y_proba is not None:
    with st.expander("📐 ROC Curve & AUC Score"):
        st.subheader("Measures how well the model separates the two classes across thresholds.")

        st.markdown("""
    The ROC curve shows how well the model separates the two classes.  
    AUC (Area Under the Curve) summarizes this into a single score — the closer to **1.0**, the better.
    """)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        fig_roc, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig_roc)
        st.metric("ROC AUC", f"{auc:.3f}")

with st.expander("🔗 Feature Correlation Matrix"):
    st.subheader("Check relationships between features, or between features and the target.")
    st.markdown("""
This heatmap shows how features are correlated with each other and with the label.  
- High correlation between features may suggest redundancy (multicollinearity).  
- High correlation with the label may indicate strong predictors — or risks like bias or data leakage.
""")

    X_corr = X_train.copy()
    X_corr["Label"] = y_train
    corr_matrix = X_corr.corr()

    # mode = st.pills("Correlation type", ["Raw feature values", "SHAP values"], help="Features highly correlated with the label are likely to be strong predictors — but may require scrutiny for fairness or leakage.")

    # if mode == "Raw feature values":
    #     X_corr = X_train.copy()
    #     X_corr["Label"] = y_train
    #     corr_matrix = X_corr.corr()

    # else:
    #     explainer = shap.TreeExplainer(model)
    #     shap_vals = explainer.shap_values(X_train)
    #     X_corr = pd.DataFrame(shap_vals, columns=X_train.columns).copy()
    #     X_corr["Label"] = y_train
    #     corr_matrix = X_corr.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, square=True, ax=ax, vmin=-1, vmax=1)
    st.pyplot(fig)

# === SHAP ===
with st.expander("🌍 Overall Feature Impact", expanded=False):
    st.subheader("Which Features Influence the Model Most?")
    st.markdown("""
This chart shows the overall importance of each feature across all predictions.  
""")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    fig_shap, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig_shap)


# === SHAP Force Plots ===
with st.expander("📍 Local Feature Impact", expanded=False):
    st.subheader("What Drove a Specific Prediction?")
    st.markdown("""
This view breaks down a single prediction to show which features pushed the prediction score higher or lower.  
""")

    sample_idx = st.slider("Select Sample Index", 0, len(X_test) - 1, 0)

    shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=shap_values[sample_idx],
        features=X_test.iloc[sample_idx],
        feature_names=X_test.columns,
        matplotlib=True
    )

    fig = plt.gcf()
    st.pyplot(fig)




with st.expander("🧬 Feature Interaction Explorer"):
    st.subheader("How Features Work *Together* to Influence Predictions")
    st.markdown("""
While most explanations focus on individual features, this view explores how **two features combined** impact the model.  
""")

    LOG_DIR = Path("logs/interactions_explorer")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    @st.cache_data
    def compute_interactions(_model, X_sample):
        explainer = shap.TreeExplainer(_model)
        return explainer.shap_interaction_values(X_sample)

    @st.cache_data
    def get_top_interactions(interaction_values, feature_names, feature_idx):
        mean_interactions = np.abs(interaction_values[:, feature_idx, :]).mean(axis=0)
        top_idx = np.argsort(-mean_interactions)
        top_pairs = [(feature_names[i], mean_interactions[i]) for i in top_idx if i != feature_idx][:3]
        return top_pairs

    @st.cache_data
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

    # === Load data/model
    df = load_data("data/adult.data")
    X, y, encoders = preprocess(df)
    model, _ = load_model_and_explainer(X, y)
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    X_sample = X_train.sample(500, random_state=42)
    X_sample_human = decode_features(X_sample, encoders)

    # === Feature selection
    display_names = [name.replace("_", " ").title() for name in X.columns]
    selected_display = st.selectbox("Select a main feature", display_names, index=display_names.index("Education"))
    selected_feature = X.columns[display_names.index(selected_display)]

    # === SHAP interaction values
    interaction_values = compute_interactions(model, X_sample)
    top_pairs = get_top_interactions(interaction_values, list(X.columns), X.columns.get_loc(selected_feature))
    top_feature_names_raw = [pair[0] for pair in top_pairs]

    for selected_pair in top_feature_names_raw:
        binned_cache = LOG_DIR / f"binned_{selected_feature}_{selected_pair}.json"
        llm_cache = LOG_DIR / f"llm_{selected_feature}_{selected_pair}.txt"


        if binned_cache.exists() and llm_cache.exists():
            try:
                binned_df = pd.read_json(binned_cache, orient="records")
                with open(llm_cache, "r", encoding="utf-8") as f:
                    explanation = f.read()
            except Exception as e:
                st.warning(f"Cache file error: {e}")
                binned_df = get_binned_interactions(X_sample_human, interaction_values, selected_feature, selected_pair)
                explanation = ""
        else:
            # ''' Added for Streamlit compatibility'''

            st.warning(f"⚠️ LLM interpretation not available at the moment — skipping explanation. ({e})")
            binned_cache.write_text(binned_df.to_json(orient="records"), encoding="utf-8")
            llm_cache.write_text("⚠️ LLM interpretation not available at the moment.", encoding="utf-8")
            explanation = "⚠️ LLM interpretation not available at the moment."

            # ''' Commented out for Streamlit compatibility'''
            # binned_df = get_binned_interactions(X_sample_human, interaction_values, selected_feature, selected_pair)

            # from utils.llm_utils import (
            #     build_correlation_prompt,
            #     generate_streaming_chunks,
            #     sampler_with_temperature_topk_topp,
            #     load
            # )
            # try:
            #     model_mlx, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.2-4bit")
            # except Exception as e:
                # st.warning(f"⚠️ LLM interpretation not available at the moment — skipping explanation. ({e})")
                # binned_cache.write_text(binned_df.to_json(orient="records"), encoding="utf-8")
                # llm_cache.write_text("⚠️ LLM interpretation not available at the moment.", encoding="utf-8")
                # explanation = "⚠️ LLM interpretation not available at the moment."
            # else:
            #     sampler = sampler_with_temperature_topk_topp(temperature=0.7, top_k=40, top_p=0.9)

            #     explanation_chunks = []
            #     with st.spinner("Generating LLM insight..."):
            #         for chunk in generate_streaming_chunks(
            #             model_mlx, tokenizer, prompt, max_tokens=1000, sampler=sampler
            #         ):
            #             explanation_chunks.append(chunk)
            #     explanation = "".join(explanation_chunks)

            #     binned_cache.write_text(binned_df.to_json(orient="records"), encoding="utf-8")
            #     llm_cache.write_text(explanation, encoding="utf-8")

        if not binned_df.empty:
            st.subheader(
                f"Interaction of {selected_feature.title().replace('_', ' ')} with {selected_pair.title().replace('_', ' ')}", 
                help=f"{selected_feature.title().replace('_', ' ')} = {FEATURE_DESCRIPTIONS[selected_feature]} | {selected_pair.title().replace('_', ' ')} = {FEATURE_DESCRIPTIONS[selected_pair]}")
            st.markdown(f"""
            The chart below shows how the **combined effect** of **{selected_feature.replace('_', ' ').title()}** and  
            **{selected_pair.replace('_', ' ').title()}** influences model predictions across different groups of **{selected_pair.replace('_', ' ').title()}**.

            _Powered by Mistral._
            """)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(binned_df.iloc[:, 0].astype(str), binned_df["Interaction"], color="skyblue")
            ax.axhline(0, color="gray", linestyle="--")
            ax.set_title(f"{selected_feature} × {selected_pair} Interaction by {selected_pair} Bins")
            ax.set_ylabel("Mean SHAP Interaction")
            ax.set_xlabel(f"{selected_pair} Bins")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            def describe_interaction(feature, pair, df):
                trend = df["Interaction"].values
                bins = df.iloc[:, 0].astype(str).values

                if len(trend) < 2:
                    return "Insufficient data to describe the trend."

                direction = "increases" if trend[-1] > trend[0] else "decreases"
                return f"As **{pair}** increases, the influence of **{feature}** on the prediction {direction}."

            summary = describe_interaction(selected_feature, selected_pair, binned_df)

            st.subheader("LLM Insights")
            st.markdown(explanation)
