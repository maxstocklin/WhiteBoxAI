
from matplotlib.figure import Figure
import pandas as pd
# Instead of rendering, return a figure object for later use in Streamlit
def create_interaction_plot(mean_interaction_df, feature_a, feature_b):
    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()
    ax.bar(mean_interaction_df.index.astype(str), mean_interaction_df["Interaction"], color="skyblue")
    ax.set_title(f"Interaction of {feature_a} with {feature_b}")
    ax.set_ylabel("Mean SHAP Interaction")
    ax.set_xlabel(f"{feature_b} Bins")
    ax.axhline(0, color="gray", linestyle="--")
    fig.tight_layout()
    return fig


def get_feature_bins(feature_name, series):
    if feature_name == "age":
        return pd.cut(series, bins=[0, 25, 35, 45, 55, 65, 100],
                      labels=["<25", "25–35", "35–45", "45–55", "55–65", "65+"])
    elif feature_name == "education_num":
        return pd.cut(series, bins=[0, 8, 10, 12, 14, 16],
                      labels=["<9", "9–10", "11–12", "13–14", "15+"])
    elif feature_name == "hours_per_week":
        return pd.cut(series, bins=[0, 20, 30, 40, 50, 60, 100],
                      labels=["<20", "20–30", "30–40", "40–50", "50–60", "60+"])
    elif feature_name == "capital_gain":
        return pd.cut(series, bins=[-1, 0, 1000, 5000, 10000, 50000, 100000],
                      labels=["0", "1–1K", "1K–5K", "5K–10K", "10K–50K", "50K+"])
    elif feature_name == "capital_loss":
        return pd.cut(series, bins=[-1, 0, 500, 1500, 2500, 5000],
                      labels=["0", "1–500", "500–1500", "1500–2500", "2500+"])
    elif feature_name == "fnlwgt":
        return pd.qcut(series, q=5, duplicates='drop')
    else:
        try:
            return pd.qcut(series, q=5, duplicates='drop')
        except:
            return series  # fallback for non-numeric or constant
