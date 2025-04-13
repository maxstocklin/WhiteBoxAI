import pandas as pd
try:
    import dice_ml
    from dice_ml.utils import helpers
    DICE_AVAILABLE = True
except ImportError:
    DICE_AVAILABLE = False

def load_dice_explainer(model, X_train, y_train, outcome_name="income"):
    """
    Wraps XGBoost/sklearn model in a DiCE explainer.
    """
    d = dice_ml.Data(
        dataframe=X_train.copy().assign(**{outcome_name: y_train}),
        continuous_features=X_train.select_dtypes(include=["int", "float"]).columns.tolist(),
        outcome_name=outcome_name
    )

    m = dice_ml.Model(model=model, backend="sklearn")
    explainer = dice_ml.Dice(d, m, method="random")

    return explainer


def generate_counterfactuals(
    sample,
    explainer,
    total=3,
    features_to_vary="all",
    desired_class="opposite",
    proximity_weight=0.5,
    diversity_weight=0.9,
):
    """
    Generate counterfactuals using DiCE.

    Parameters:
        - sample: pd.Series (one row from X)
        - explainer: DiCE explainer
        - total: number of counterfactuals to generate
        - features_to_vary: list or "all"
        - desired_class: "opposite" or 0/1
        - proximity_weight: float, importance of staying close to input
        - diversity_weight: float, importance of diverse outputs

    Returns:
        - cf_df: pd.DataFrame of counterfactuals
    """
    query_instance = pd.DataFrame([sample])

    # === Define realistic constraints ===
    permitted_range = {
        "age": (max(0, sample["age"] - 5), min(100, sample["age"] + 5)),
        "hours_per_week": (max(1, sample["hours_per_week"] - 10), min(100, sample["hours_per_week"] + 10)),
        "education_num": (max(1, sample["education_num"] - 2), min(16, sample["education_num"] + 2)),
        "capital_gain": (0, sample["capital_gain"] + 5000),
        "capital_loss": (0, sample["capital_loss"] + 1000),
        "fnlwgt": (sample["fnlwgt"] * 0.9, sample["fnlwgt"] * 1.1),
    }

    cf = explainer.generate_counterfactuals(
        query_instances=query_instance,
        total_CFs=total,
        desired_class=desired_class,
        features_to_vary=features_to_vary,
        proximity_weight=proximity_weight,
        diversity_weight=diversity_weight,
        permitted_range=permitted_range,
    )

    return cf.cf_examples_list[0].final_cfs_df

def find_minimal_flip(sample, model, feature, X_train, permitted_range=None):
    from pandas import DataFrame

    original_pred = model.predict(DataFrame([sample]))[0]

    # Get candidate values
    unique_vals = sorted(X_train[feature].unique())

    # If numeric and permitted_range is given, filter values accordingly
    if permitted_range and X_train[feature].dtype.kind in 'biufc':
        min_val, max_val = permitted_range
        unique_vals = [v for v in unique_vals if min_val <= v <= max_val]

    for val in unique_vals:
        if val == sample[feature]:
            continue
        modified = sample.copy()
        modified[feature] = val
        new_pred = model.predict(DataFrame([modified]))[0]
        if new_pred != original_pred:
            return {
                "feature": feature,
                "original": sample[feature],
                "changed_to": val,
                "original_pred": original_pred,
                "new_pred": new_pred
            }

    return None
