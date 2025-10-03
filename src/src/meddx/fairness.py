from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import load
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate, demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference

def load_test():
    df = pd.read_csv("data/processed/test.csv")
    X, y = df.drop(columns=["target"]), df["target"].astype(int)
    return df, X, y

def bin_age(a):
    bins = [0, 40, 55, 70, 150]
    labels = ["<=40", "41-55", "56-70", "70+"]
    return pd.cut(a, bins=bins, labels=labels, right=True, include_lowest=True)

def ensure_dirs():
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("artifacts").mkdir(parents=True, exist_ok=True)

def main():
    ensure_dirs()
    df, X, y = load_test()
    model = load("artifacts/model.joblib")
    p = model.predict_proba(X)[:,1] if hasattr(model,"predict_proba") else model.predict(X)
    yhat = (p>=0.5).astype(int)

    sex = df["sex"].astype(int)
    ageg = bin_age(df["age"])

    metrics = {
        "selection_rate_by_sex": MetricFrame(metrics=selection_rate, y_true=y, y_pred=yhat, sensitive_features=sex).by_group.to_dict(),
        "tpr_by_sex": MetricFrame(metrics=true_positive_rate, y_true=y, y_pred=yhat, sensitive_features=sex).by_group.to_dict(),
        "fpr_by_sex": MetricFrame(metrics=false_positive_rate, y_true=y, y_pred=yhat, sensitive_features=sex).by_group.to_dict(),
        "dp_diff_sex": float(demographic_parity_difference(y, yhat, sensitive_features=sex)),
        "dp_ratio_sex": float(demographic_parity_ratio(y, yhat, sensitive_features=sex)),
        "eo_diff_sex": float(equalized_odds_difference(y, yhat, sensitive_features=sex)),
        "selection_rate_by_age": MetricFrame(metrics=selection_rate, y_true=y, y_pred=yhat, sensitive_features=ageg).by_group.to_dict(),
        "tpr_by_age": MetricFrame(metrics=true_positive_rate, y_true=y, y_pred=yhat, sensitive_features=ageg).by_group.to_dict(),
        "fpr_by_age": MetricFrame(metrics=false_positive_rate, y_true=y, y_pred=yhat, sensitive_features=ageg).by_group.to_dict(),
    }

    Path("artifacts").mkdir(exist_ok=True, parents=True)
    with open("artifacts/fairness_summary.json","w") as f:
        json.dump(metrics, f, indent=2)

    with open("reports/fairness_report.md","w") as f:
        f.write("# Fairness Report\n\n")
        f.write("## Sensitive attribute: sex (0=female,1=male)\n\n")
        f.write(f"- Demographic parity difference: {metrics['dp_diff_sex']:.4f}\n")
        f.write(f"- Demographic parity ratio: {metrics['dp_ratio_sex']:.4f}\n")
        f.write(f"- Equalized odds difference: {metrics['eo_diff_sex']:.4f}\n\n")
        f.write("### Selection rate by sex\n")
        f.write(str(metrics["selection_rate_by_sex"]) + "\n\n")
        f.write("### TPR by sex\n")
        f.write(str(metrics["tpr_by_sex"]) + "\n\n")
        f.write("### FPR by sex\n")
        f.write(str(metrics["fpr_by_sex"]) + "\n\n")
        f.write("## Sensitive attribute: age groups\n\n")
        f.write("### Selection rate by age\n")
        f.write(str(metrics["selection_rate_by_age"]) + "\n\n")
        f.write("### TPR by age\n")
        f.write(str(metrics["tpr_by_age"]) + "\n\n")
        f.write("### FPR by age\n")
        f.write(str(metrics["fpr_by_age"]) + "\n\n")

    print("Wrote artifacts/fairness_summary.json and reports/fairness_report.md")

if __name__ == "__main__":
    main()
