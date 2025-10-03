import json, joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

def _load_splits():
    tr = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/val.csv")
    Xtr, ytr = tr.drop(columns=["target"]), tr["target"].astype(int)
    Xv, yv = val.drop(columns=["target"]), val["target"].astype(int)
    num = Xtr.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in Xtr.columns if c not in num]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), num),
            ("cat", Pipeline([("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ],
        remainder="drop",
    )
    return Xtr, ytr, Xv, yv, pre

def _plots(y, p, prefix):
    fpr, tpr = [], []
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
    Xtr, ytr, Xv, yv, pre = _load_splits()
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=200, n_jobs=1 if hasattr(LogisticRegression(), "n_jobs") else None))])
    mlflow.set_tracking_uri("file:mlruns")
    mlflow.set_experiment("meddx")
    with mlflow.start_run(run_name="baseline-train"):
        mlflow.log_params({"model":"LogisticRegression","max_iter":200})
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xv)[:,1] if hasattr(model,"predict_proba") else model.predict(Xv)
        metrics = {
            "val_auroc": float(roc_auc_score(yv, p)),
            "val_auprc": float(average_precision_score(yv, p)),
            "val_brier": float(brier_score_loss(yv, p)),
        }
        Path("artifacts").mkdir(exist_ok=True, parents=True)
        joblib.dump(model, "artifacts/model.joblib")
        with open("artifacts/val_metrics.json","w") as f: json.dump(metrics, f, indent=2)
        roc_path, pr_path = _plots(yv, p, "val")
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact("artifacts/model.joblib")
        mlflow.log_artifact("artifacts/val_metrics.json")
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(pr_path)
        print(json.dumps(metrics, indent=2))
        print("Saved artifacts and logged to MLflow at ./mlruns")
    return 0

if __name__ == "__main__":
    main()
