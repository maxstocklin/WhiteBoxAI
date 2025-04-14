import streamlit as st

st.set_page_config(
    page_title="WhiteBox AI",
    page_icon="📊",
    layout="wide",
)

selection = ""
# Page router
if selection == "📄 Data Overview":
    st.switch_page("pages/1_Data_Overview.py")

elif selection == "🏋️ Performance Dashboard":
    st.switch_page("pages/2_Performance_Dashboard.py")

elif selection == "🔍 XAI Prediction Breakdown":
    st.switch_page("pages/3_XAI_Prediction_Breakdown.py")

elif selection == "🧪 Real-Time Simulator":
    st.switch_page("pages/4_Real-Time_Simulator.py")

elif selection == "⚖️ Fairness & Bias Analyisis":
    st.switch_page("pages/5_Fairness_&_Bias_Analyisis.py")


st.title("Welcome to WhiteBox AI")

st.markdown("""
WhiteBox AI is an interactive tool designed to help you **understand how machine learning models make decisions**.

We go beyond accuracy. WhiteBox AI adds **explainability layers** to every prediction, helping:

- Data scientists explore model behavior  
- Domain experts understand feature influence  
- Decision makers gain confidence in AI-driven outcomes
""")
st.markdown("")

st.info("""
In this demo, we use **XGBoost** — one of the most powerful algorithms for tabular data — to solve a simple but meaningful task:

> 🎯 **Can we predict whether someone earns more than $50K/year based on their personal and professional profile?**
""")
st.markdown("""

---

### What You Can Explore

- `Data Overview`: Browse individuals and explore input features.
- `Performance Dashboard`: Evaluate model quality and global feature importance.
- `Prediction Breakdown`: See what drove a specific prediction — with confidence, reasoning, and natural-language summary.
- `Real-Time Simulator`: Adjust inputs and watch predictions and explanations change in real time.
- `Fairness & Bias Analysis`: Investigate group disparities and SHAP differences by race, sex, and more.

---

Use the sidebar to navigate between pages and explore **how, why, and when** the model makes each decision.
""")

with st.expander("📁 About the Data"):
    st.markdown("""
This demo uses the **UCI Adult Income dataset**, a classic benchmark for classification tasks.
Each sample represents an individual, described by attributes such as:
- Age, education, occupation
- Marital status, working hours, capital gains/losses
- Sex, race, and country of origin

The goal is to predict:  
> **Does this person earn >$50K per year?**
""")

with st.expander("⚙️ About the Algorithm"):
    st.markdown("""
We use **XGBoost**, a high-performance gradient boosting model widely adopted in industry for structured data.

Why XGBoost?
- 🚀 Fast and accurate
- 🌲 Tree-based, great for interpretability
- 🔍 Well-suited for SHAP explanations

All predictions are enriched with **confidence scores**, **SHAP-based reasoning**, and **fairness metrics**.
""")
