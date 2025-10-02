from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from joblib import load

def load_test(root="data/processed"):
    test = pd.read_csv(Path(root)/"test.csv")
    return test.drop(columns=["target"]), test["target"].astype(int)

def evaluate():
    X_test, y_test = load_test()
    model = load("artifacts/model.joblib")
    p = model.predict_proba(X_test)[:,1] if hasattr(model,"predict_proba") else model.predict(X_test)
    metrics = {
        "auroc": float(roc_auc_score(y_test, p)),
        "auprc": float(average_precision_score(y_test, p)),
        "brier": float(brier_score_loss(y_test, p))
    }
    Path("artifacts").mkdir(exist_ok=True, parents=True)
    with open("artifacts/test_metrics.json","w") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)
    return metrics

if __name__ == "__main__":
    evaluate()
