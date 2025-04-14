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


st.title("👋 Welcome to WhiteBox AI")

st.markdown("""
This tool is designed to help **everyone** — data scientists, domain experts, and decision makers — understand how a machine learning model makes its predictions. It’s built on top of **XGBoost**, one of the most powerful algorithms for tabular data, and enriched with explainability features.

You’ll find a suite of pages, each tailored to a specific exploration task:

- 📄 **Data Overview**: Browse test samples with filtering and direct links to interpretation.
- 🏋️‍♂️ **Performance Dashboard**: Evaluate model performance and explore global feature importance.
- 🔬 **XAI Prediction Breakdown**: Deep-dive into a single prediction with a multi-layered confidence report, detailed reasoning report, and a natural language summary.
- 🎚 **Real-Time Simulator**: Adjust input features and see how predictions shift. Try counterfactuals, feature tweaking, and SHAP comparisons.
- ⚖️ **Fairness & Bias Analyisis**: Investigate bias across sensitive attributes like sex and race. Includes fairness metrics and SHAP attribution by group.

Use the sidebar to navigate between these pages and uncover what drives each prediction.
""")

with st.expander("📂 About the Data"):
    st.markdown("""
This app uses the classic **Adult Census Income Dataset** (also known as the "Census Income" or `adult.data` dataset).

- **Source**: UCI Machine Learning Repository  
- **Task**: Predict whether an individual's income exceeds $50K/year  
- **Features**: Includes demographic and work-related attributes (e.g., age, education, occupation, sex, hours-per-week, etc.)  
- **Target**: Binary label — `>50K` or `<=50K`  
- **Preprocessing**: Categorical features are label-encoded, missing values (`?`) are handled during cleaning

This dataset is well-suited for testing explainability techniques due to its mix of numeric and categorical features.
""")

with st.expander("⚙️ About the Algorithm"):
    st.markdown("""
**XGBoost (Extreme Gradient Boosting)** is a high-performance implementation of gradient boosted decision trees. It's widely used in Kaggle competitions and industry for structured data tasks.

- **Boosting**: Combines weak learners (decision trees) iteratively, improving errors at each stage  
- **Loss Optimization**: Uses gradient descent on a loss function (e.g. log-loss)  
- **Regularization**: Includes L1/L2 penalties to reduce overfitting  
- **Parallelizable**: Efficient training and inference  

XGBoost is particularly interpretable thanks to:
- Tree path tracing
- SHAP values for feature attribution
- Leaf-level voting and path support analysis

This app leverages all these strengths, plus adds LLM-generated reasoning and audit layers.
""")