import json, joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import matplotlib.pyplot as plt
import mlflow

def _plots(y, p, prefix):
    from sklearn.metrics import roc_curve, precision_recall_curve
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    Path("reports").mkdir(exist_ok=True, parents=True)
    plt.figure(); plt.plot(fpr, tpr); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
    roc_path = f"reports/roc_{prefix}.png"; plt.savefig(roc_path, dpi=200); plt.close()
    plt.figure(); plt.plot(rec, prec); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR")
    pr_path = f"reports/pr_{prefix}.png"; plt.savefig(pr_path, dpi=200); plt.close()
    return roc_path, pr_path

def main():
    test = pd.read_csv("data/processed/test.csv")
    X, y = test.drop(columns=["target"]), test["target"].astype(int)
    model = joblib.load("artifacts/model.joblib")
    p = model.predict_proba(X)[:,1] if hasattr(model,"predict_proba") else model.predict(X)
    metrics = {
        "test_auroc": float(roc_auc_score(y, p)),
        "test_auprc": float(average_precision_score(y, p)),
        "test_brier": float(brier_score_loss(y, p)),
    }
    Path("artifacts").mkdir(exist_ok=True, parents=True)
    with open("artifacts/test_metrics.json","w") as f: json.dump(metrics, f, indent=2)
    roc_path, pr_path = _plots(y, p, "test")
    mlflow.set_tracking_uri("file:mlruns")
    mlflow.set_experiment("meddx")
    with mlflow.start_run(run_name="baseline-eval"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("artifacts/test_metrics.json")
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(pr_path)
        print(json.dumps(metrics, indent=2))
    return 0

if __name__ == "__main__":
    main()
