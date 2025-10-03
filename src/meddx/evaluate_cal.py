import json, joblib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve

def load_test():
    df = pd.read_csv("data/processed/test.csv")
    X, y = df.drop(columns=["target"]), df["target"].astype(int)
    return X, y

def main():
    Path("reports").mkdir(parents=True, exist_ok=True)
    model = joblib.load("artifacts/model_adv.joblib")
    X, y = load_test()
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[:,1]
    else:
        p = model.decision_function(X)

    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    brier = brier_score_loss(y, p)

    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    x = np.linspace(0,1,100)
    plt.plot(x, x, "--", label="Perfect")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/calibration_curve.png", dpi=200)
    plt.close()

    with open("artifacts/adv_test_metrics.json","w") as f:
        json.dump({"test_auc": float(auc), "test_auprc": float(ap), "test_brier": float(brier)}, f, indent=2)

    print("Wrote reports/calibration_curve.png and artifacts/adv_test_metrics.json")
