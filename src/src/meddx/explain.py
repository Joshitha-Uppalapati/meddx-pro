from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt

def load_val():
    val = pd.read_csv("data/processed/val.csv")
    X, y = val.drop(columns=["target"]), val["target"].astype(int)
    return X, y

def ensure_reports():
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("artifacts").mkdir(parents=True, exist_ok=True)

def main():
    ensure_reports()
    model = load("artifacts/model.joblib")
    X, y = load_val()

    try:
        import shap
        explainer = shap.Explainer(model, X, feature_names=X.columns)
        sample = X.sample(min(200, len(X)), random_state=42)
        shap_values = explainer(sample)

        plt.figure()
        shap.plots.beeswarm(shap_values, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig("reports/shap_beeswarm.png", dpi=200)
        plt.close()

        vals = np.abs(shap_values.values).mean(axis=0)
        order = np.argsort(vals)[::-1]
        topk = 15 if len(vals) >= 15 else len(vals)
        top = [(X.columns[i], float(vals[i])) for i in order[:topk]]
        with open("artifacts/top_features.json","w") as f:
            json.dump({"top_features_mean_abs_shap": top}, f, indent=2)

        shap.plots.bar(shap_values, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig("reports/shap_bar.png", dpi=200)
        plt.close()

    except Exception as e:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=1, scoring="roc_auc")
        imp = sorted([(X.columns[i], float(r.importances_mean[i])) for i in range(len(X.columns))], key=lambda x: -abs(x[1]))[:15]
        with open("artifacts/top_features.json","w") as f:
            json.dump({"permutation_importance": imp, "note": f"SHAP fallback: {type(e).__name__}"}, f, indent=2)
        fig = plt.figure()
        names = [k for k,_ in imp][::-1]
        vals = [abs(v) for _,v in imp][::-1]
        plt.barh(names, vals)
        plt.tight_layout()
        plt.savefig("reports/perm_importance.png", dpi=200)
        plt.close()

    print("Wrote reports/*shap*.png or perm_importance.png and artifacts/top_features.json")

if __name__ == "__main__":
    main()
