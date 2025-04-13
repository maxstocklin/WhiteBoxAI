import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

_model = None
_explainer = None

def load_model_and_explainer(X, y):
    global _model, _explainer

    if _model is not None and _explainer is not None:
        return _model, _explainer

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)

    _model = model
    _explainer = explainer
    return model, explainer
