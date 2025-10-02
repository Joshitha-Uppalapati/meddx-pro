from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt
from joblib import dump

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def load_splits(root="data/processed"):
    train = pd.read_csv(Path(root)/"train.csv")
    val   = pd.read_csv(Path(root)/"val.csv")
    X_train, y_train = train.drop(columns=["target"]), train["target"].astype(int)
    X_val,   y_val   = val.drop(columns=["target"]),   val["target"].astype(int)
    return X_train, y_train, X_val, y_val

def train_and_eval():
    X_train, y_train, X_val, y_val = load_splits()
    models = {}

    lr = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=200, n_jobs=1))
    ])
    lr.fit(X_train, y_train)
    models["logreg"] = lr

    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="logloss", n_jobs=2
        )
        xgb.fit(X_train, y_train)
        models["xgboost"] = xgb

    metrics = {}
    for name, mdl in models.items():
        p = mdl.predict_proba(X_val)[:,1] if hasattr(mdl, "predict_proba") else mdl.predict(X_val)
        metrics[name] = {
            "auroc": float(roc_auc_score(y_val, p)),
            "auprc": float(average_precision_score(y_val, p)),
            "brier": float(brier_score_loss(y_val, p))
        }
    best_name = sorted(metrics.items(), key=lambda kv: kv[1]["auroc"], reverse=True)[0][0]
    best_model = models[best_name]

    Path("artifacts").mkdir(parents=True, exist_ok=True)
    dump(best_model, "artifacts/model.joblib")
    with open("artifacts/val_metrics.json","w") as f:
        json.dump({"metrics":metrics, "best_model":best_name}, f, indent=2)

    p_best = best_model.predict_proba(X_val)[:,1] if hasattr(best_model, "predict_proba") else best_model.predict(X_val)
    RocCurveDisplay.from_predictions(y_val, p_best)
    Path("reports").mkdir(exist_ok=True, parents=True)
    plt.title(f"ROC — {best_name}")
    plt.savefig("reports/roc_val.png"); plt.close()
    PrecisionRecallDisplay.from_predictions(y_val, p_best)
    plt.title(f"PR — {best_name}")
    plt.savefig("reports/pr_val.png"); plt.close()

    print(f"Saved artifacts/model.joblib with best={best_name}")
    print(f"Validation metrics: {metrics[best_name]}")
    return best_name, metrics

if __name__ == "__main__":
    train_and_eval()
