import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")

from utils.data_loader import load_data, preprocess

st.title("Data Overview")
# st.caption("Explore samples from the Census Income dataset and filter by label or feature values.")

# === Load and preprocess data ===
df = load_data("data/adult.data")
X, y, encoders = preprocess(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Combine features and labels ===
X_test_human = X_test.copy()
for col in encoders:
    if col in X_test_human.columns:
        X_test_human[col] = encoders[col].inverse_transform(X_test_human[col])
data_view = X_test_human.copy()
data_view["True Label"] = y_test

# === Filters ===
with st.expander("🔍 Filter Options", expanded=False):
    st.subheader("Use filters to narrow down the dataset view.")
    st.markdown("This tool helps focus your analysis on specific groups or conditions.")
    # Create two columns for numeric and categorical filters
    col_num, col_cat = st.columns(2)

    # Determine which columns to filter (exclude the index column)
    filter_cols = [col for col in data_view.columns if col != "index"]
    num_cols = [col for col in filter_cols if pd.api.types.is_numeric_dtype(data_view[col])]
    cat_cols = [col for col in filter_cols if not pd.api.types.is_numeric_dtype(data_view[col])]

    with col_num:
        st.markdown("#### Numeric Filters")
        for col in num_cols:
            if col in ["True Label"]:
                continue
            if data_view[col].dropna().empty:
                st.warning(f"⚠️ No data available for column '{col}' after applying previous filters.")
                continue  # Skip this filter
            min_val = int(data_view[col].min())
            max_val = int(data_view[col].max())
            if min_val == max_val:
                st.info(f"ℹ️ All remaining rows have the same value `{min_val}` for `{col}`. Skipping slider.")
                continue

            user_range = st.slider(
                f"Select range for {col}",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
            )
            data_view = data_view[(data_view[col] >= user_range[0]) & (data_view[col] <= user_range[1])]
        st.markdown("#### True Label")
        class_filter = st.selectbox("Filter by true label", options=["All", 0, 1], index=0)
        if class_filter != "All":
            data_view = data_view[data_view["True Label"] == class_filter]

    with col_cat:
        st.markdown("#### Categorical Filters")
        for col in cat_cols:
            unique_vals = sorted(data_view[col].unique())
            selected_vals = st.multiselect(
                f"Select values for {col}",
                options=unique_vals,
                default=unique_vals,
                help="Filter by selecting one or more values"
            )
            data_view = data_view[data_view[col].isin(selected_vals)]

# === Add full URL link for interpretation ===
data_view = data_view.copy()
data_view.reset_index(inplace=True)
# base_url = "http://localhost:8503/Interpretation"  # adjust if deploying elsewhere
# data_view["Interpret"] = data_view["index"].apply(
#     lambda idx: f"{base_url}?id={idx}"
# )

# === Show table using st.dataframe with clickable links ===
st.subheader("Census Income Dataset")
st.markdown("You are viewing a sample from the U.S. Census dataset. Each row is a person, and the label shows whether they earn >50K or not.")
st.dataframe(
    data_view[["index", "True Label"] + list(X.columns)],
    # column_config={
    #     "Interpret": st.column_config.LinkColumn("🔍 Interpret", help="Click to open sample in Interpretation page")
    # },
    use_container_width=True
)
