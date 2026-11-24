import shap
import matplotlib.pyplot as plt

def explainer_values(xgb_model, X):
    ex = shap.TreeExplainer(xgb_model)
    return ex, ex(X)

def summary_plot(sv, kind="beeswarm"):
    fig = plt.figure()
    if kind == "bar":
        shap.plots.bar(sv, show=False)
    else:
        shap.plots.beeswarm(sv, show=False)
    plt.tight_layout()
    return plt.gcf()

def waterfall_plot(sv_row):
    plt.figure()
    shap.plots.waterfall(sv_row, show=False)
    plt.tight_layout()
    return plt.gcf()
