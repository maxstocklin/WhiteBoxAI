import streamlit as st

st.set_page_config(
    page_title="XGBoost Interpretability Explorer",
    page_icon="📊",
    layout="wide",
)

selection = ""
# Page router
if selection == "📄 Dataset Viewer":
    st.switch_page("pages/1_Dataset_Viewer.py")

elif selection == "🏋️ Train / Test / Visualize":
    st.switch_page("pages/2_Train_Test_Visualize.py")

elif selection == "🔍 Interpretation":
    st.switch_page("pages/3_Interpretation.py")

elif selection == "🧪 Simulator":
    st.switch_page("pages/4_Simulator.py")

elif selection == "⚖️ Biais Dashboard":
    st.switch_page("pages/5_Biais_Dashboard.py")


st.title("👋 Welcome to the XGBoost Interpretability App")

st.markdown("""
This tool is designed to help **everyone** — data scientists, domain experts, and decision makers — understand how a machine learning model makes its predictions. It’s built on top of **XGBoost**, one of the most powerful algorithms for tabular data, and enriched with explainability features.

You’ll find a suite of pages, each tailored to a specific exploration task:

- 🏋️‍♂️ **Train / Test / Visualize**: Evaluate model performance and explore global feature importance.
- 📄 **Dataset Viewer**: Browse test samples with filtering and direct links to interpretation.
- 🔬 **Sample Interpretation**: Deep-dive into a single prediction with a multi-layered confidence report, SHAP values, and a natural language summary.
- 🎛️ **Prediction Simulator**: Adjust input features and see how predictions shift. Try counterfactuals, feature tweaking, and SHAP comparisons.
- ⚖️ **Fairness Dashboard**: Investigate bias across sensitive attributes like sex and race. Includes fairness metrics and SHAP attribution by group.

Use the sidebar to navigate between these pages and uncover what drives each prediction.
""")

with st.expander("📂 Dataset — About the Data"):
    st.markdown("""
This app uses the classic **Adult Census Income Dataset** (also known as the "Census Income" or `adult.data` dataset).

- **Source**: UCI Machine Learning Repository  
- **Task**: Predict whether an individual's income exceeds $50K/year  
- **Features**: Includes demographic and work-related attributes (e.g., age, education, occupation, sex, hours-per-week, etc.)  
- **Target**: Binary label — `>50K` or `<=50K`  
- **Preprocessing**: Categorical features are label-encoded, missing values (`?`) are handled during cleaning

This dataset is well-suited for testing explainability techniques due to its mix of numeric and categorical features.
""")

with st.expander("⚙️ Algorithm — About XGBoost"):
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