import json, joblib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    from sklearn.ensemble import GradientBoostingClassifier as _FallbackGB

def _load_splits():
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/val.csv")
    Xtr, ytr = train.drop(columns=["target"]), train["target"].astype(int)
    Xval, yval = val.drop(columns=["target"]), val["target"].astype(int)
    return Xtr, ytr, Xval, yval

def _preproc(X):
    num_cols = X.columns.tolist()
    num = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())])
    return ColumnTransformer([("num", num, num_cols)], remainder="drop")

def _objective(trial, X, y):
    model_name = trial.suggest_categorical("model", ["rf", "xgb" if HAS_XGB else "gb"])
    if model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            max_features=trial.suggest_float("max_features", 0.3, 1.0),
            n_jobs=-1,
            random_state=42,
        )
    elif model_name == "xgb":
        clf = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=100),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
    else:
        clf = _FallbackGB(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 2, 5),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            random_state=42,
        )
    pipe = Pipeline([("prep", _preproc(X)), ("clf", clf)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(pipe, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)
    return scores.mean()

def main():
    Path("artifacts").mkdir(exist_ok=True, parents=True)
    Path("reports").mkdir(exist_ok=True, parents=True)

    Xtr, ytr, Xval, yval = _load_splits()

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: _objective(t, Xtr, ytr), n_trials=30, show_progress_bar=False)

    best_params = study.best_trial.params
    model_tag = best_params.get("model", "rf")

    if model_tag == "rf":
        base = RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=best_params["min_samples_leaf"],
            max_features=best_params["max_features"],
            n_jobs=-1,
            random_state=42,
        )
    elif model_tag == "xgb" and HAS_XGB:
        base = XGBClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            subsample=best_params["subsample"],
            colsample_bytree=best_params["colsample_bytree"],
            reg_lambda=best_params["reg_lambda"],
            reg_alpha=best_params["reg_alpha"],
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
    else:
        base = _FallbackGB(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            random_state=42,
        )

    pipe = Pipeline([("prep", _preproc(Xtr)), ("clf", base)])
    pipe.fit(Xtr, ytr)

    def _prob(m, X):
        return m.predict_proba(X)[:,1] if hasattr(m, "predict_proba") else m.decision_function(X)

    p_val_raw = _prob(pipe, Xval)
    auc_raw = float(roc_auc_score(yval, p_val_raw))
    brier_sig = brier_score_loss(yval, CalibratedClassifierCV(pipe, cv="prefit", method="sigmoid").fit(Xval, yval).predict_proba(Xval)[:,1])
    brier_iso = brier_score_loss(yval, CalibratedClassifierCV(pipe, cv="prefit", method="isotonic").fit(Xval, yval).predict_proba(Xval)[:,1])
    best_method = "isotonic" if brier_iso < brier_sig else "sigmoid"
    calib = CalibratedClassifierCV(pipe, cv="prefit", method=best_method)
    calib.fit(Xval, yval)

    joblib.dump(calib, "artifacts/model_adv.joblib")
    with open("artifacts/adv_metrics.json","w") as f:
        json.dump({"val_auc_raw": auc_raw, "brier_sigmoid": float(brier_sig),
                   "brier_isotonic": float(brier_iso), "chosen_calibration": best_method,
                   "best_params": best_params}, f, indent=2)

    study_df = study.trials_dataframe()
    study_df.to_csv("artifacts/optuna_trials.csv", index=False)
    print("Saved artifacts/model_adv.joblib and artifacts/adv_metrics.json")
