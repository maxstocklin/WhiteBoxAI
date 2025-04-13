# WhiteBoxAI
## Advanced Explainable AI Platform

A powerful Streamlit-based tool designed to explore and interpret predictions made by an XGBoost model. Built for transparency, auditability, and accessibility to both technical and non-technical users.

---

## 📦 Features


### 🧪 Train / Test / Visualize
This page provides an overview of model performance and global explainability:
- Classification Report: Shows precision, recall, F1-score, and support per class.
- Confusion Matrix: Visual breakdown of true/false positives and negatives.
- ROC Curve & AUC: Measures model discrimination power.
- Feature Correlation Matrix:
  - Toggle between raw features or SHAP values.
  - Understand relationships between features and the label.
- SHAP Summary Plot:
  - Bar chart of feature importance using SHAP values.
- SHAP Force Plot:
  - Visual breakdown of a single prediction.
- Interaction Explorer:
  - Analyze and interpret feature × feature interactions using SHAP interaction values.
  - Includes LLM-generated descriptions per pair.


### 📄 Dataset Viewer
- Full interactive table with test samples (decoded features)
- Rich filtering:
  - 🔍 By label or feature range
  - Separate numeric and categorical filters
- Instant link to interpretation per row
- Built with `st.dataframe` + clickable links for intuitive navigation


### 🔬 Sample Interpretation
- Confidence Report:
  - Evaluates every prediction across **6+ trustworthiness metrics**:
    - 🔮 Prediction Probability
    - 📦 Category Familiarity
    - 🌍 Distance to Training Samples
    - 🧨 Z-Score Anomalies
    - 🧠 Tree Class Consensus (Top-K Trees)
    - 🌐 Tree Path Similarity
    - 🌲 Rare Path Coverage
  - Aggregates into an overall confidence level (🟢 High, 🟡 Medium, 🔴 Low)
- Feature-Level Explanation: 
  - Full breakdown of model decision logic per feature
- LLM Assistant Assessment:
  - Natural language interpretation generated using Mistral
  - Summarizes what influenced the decision and why it makes sense
- Bottom sticky bar: summarizes true vs predicted label and confidence, with animated color transitions when label flips


### 🎛️ Prediction Simulator
- Compare original vs. adjusted predictions side-by-side
- Live edit any feature value — including categorical and numerical fields
- Bottom sticky bar summarizes prediction shift with animated color indicator
- Three main views:
  - 🔁 **Counterfactual Explanations** (powered by DiCE): What minimal change would flip the outcome?
  - 🎚️ **Adjust Sample Inputs**: Manual controls for real-time feature tweaking
  - 🔍 **Top SHAP Differences**: Compare original vs. modified prediction explanations


### ⚖️ Fairness Dashboard
- **Data Integrity Checks**
  - Detects missing values, placeholder symbols like `'?'`, and class imbalance.
- **Fairness Metrics**
  - Select a sensitive feature like `sex` or `race`
  - Analyze how model performance and SHAP attributions vary across groups
  - Explore fairness concerns like disparate impact or demographic parity

---

## 🚀 Setup & Installation

```bash
# Clone the repo
git clone https://github.com/maxstocklin/interpretableML.git
cd interpretableML/xgb-interpretability-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run Home.py
# Note: LLM features require Apple Silicon (Mac) and will not run on Streamlit Cloud

# Optional: Enable counterfactuals via DiCE
pip install -r dice_patch.txt
```

Ensure you have the dataset in `data/adult.data`.

---

## 🤖 LLM Integration

Uses [mlx-community/Mistral-7B-Instruct-v0.2-4bit] locally with:
- Custom sampling (temperature, top-k, top-p)
- Streaming responses
- Caching of responses for each sample in `/logs/explanations/`
> ⚠️ **Note:** LLM-based explanations use `mlx` and `mlx-lm`, which currently require a Mac with Apple Silicon (M1/M2/M3) and cannot run on Streamlit Cloud. On deployed versions, LLM features will be disabled with a friendly warning.

---

## 🛠️ Planned Extensions

We're continuously improving the app to enhance interpretability, transparency, and trust. Here's what's on the roadmap:

- 🤖 RAG Q&A Assistant
  - Let users ask questions like “What matters most for this profile?”, or “What changed the outcome?”
- 🧱 Audit Toolkit
  - Export full explanations (SHAP, tree paths, confidence metrics) for any prediction, with visual + JSON traceability.
- 📊 Data Drift Detection
  - Compare incoming data distributions to training data — flag anomalies, unseen categories, and suspicious shifts.
- 🔁 MLOps Integration
  - Support model versioning, prediction logging, monitoring (e.g. confidence drops), and seamless retraining hooks.

---

## 🧩 Optional: DiCE Patch for Counterfactuals

To enable counterfactual explanations with `dice-ml`, you need a special patch due to its older dependency on `pandas<2.0.0`.

Use this helper file:

```bash
pip install -r dice_patch.txt
```

This file applies:
- `pandas<2.0.0`
- `dice-ml==0.11`
- `pandas==2.2.3` (reinstated after install)

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch
3. Submit a PR with a clear explanation

---

## 🧠 [Demo](https://whitebox.streamlit.app)

https://whitebox.streamlit.app
