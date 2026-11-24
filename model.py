import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

def get_models():
    return {"XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="logloss", random_state=0),
            "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=0, n_jobs=-1)}

def time_split(data, test_from=2008):
    """Train on earlier editions, test on later ones."""
    return data[data.Year < test_from], data[data.Year >= test_from]

def evaluate(model, Xte, yte):
    p, pred = model.predict_proba(Xte)[:, 1], model.predict(Xte)
    return {"Accuracy": accuracy_score(yte, pred), "Precision": precision_score(yte, pred),
            "Recall": recall_score(yte, pred), "F1": f1_score(yte, pred), "ROC-AUC": roc_auc_score(yte, p)}

def compare(train, test, features, synthetic=None):
    """Fit both models on real data (and real+synthetic if given). Returns metrics table + fitted models."""
    rows, fitted = [], {}
    setups = {"Real": train}
    if synthetic is not None:
        setups["Real + Synthetic"] = pd.concat([train, synthetic], ignore_index=True)
    for setup, tr in setups.items():
        for name, m in get_models().items():
            m.fit(tr[features], tr.target)
            rows.append({"Model": name, "Training data": setup, **evaluate(m, test[features], test.target)})
            fitted[(name, setup)] = m
    return pd.DataFrame(rows), fitted

def conf_matrix(model, test, features):
    return confusion_matrix(test.target, model.predict(test[features]))
