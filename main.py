#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diabetes complications prediction (UCI 130-US hospitals, 1999–2008)
Adaptive SMOTE (no more ratio errors), richer features, fast tuning (RandomizedSearchCV),
optional XGBoost, ROC curves, and Age×Gender risk charts.

Run:
  python main.py --data diabetic_data.csv --out outputs_user
  python main.py --data diabetic_data.csv --out outputs_user --tune
  python main.py --data diabetic_data.csv --out outputs_user --full --n_estimators 300 --try_xgb --tune
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from packaging import version
import sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline  # allows sampling steps
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from scipy.stats import randint as sp_randint, uniform as sp_uniform

# optional XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

RANDOM_SEED = 42

POSSIBLE_MEDS = [
    "metformin","repaglinide","nateglinide","chlorpropamide","glimepiride","acetohexamide",
    "glipizide","glyburide","tolbutamide","pioglitazone","rosiglitazone","acarbose",
    "miglitol","troglitazone","tolazamide","examide","citoglipton","insulin",
    "glyburide-metformin","glipizide-metformin","glimepiride-pioglitazone",
    "metformin-rosiglitazone","metformin-pioglitazone"
]

BASE_FEATURES = [
    "gender", "age", "medical_specialty",
    "time_in_hospital", "num_lab_procedures", "num_medications",
    "num_procedures", "number_inpatient", "number_outpatient", "number_emergency",
    "A1Cresult", "change", "diabetesMed", "readmitted",
    "diag_1", "diag_2", "diag_3"
]

TARGETS = ["has_nephropathy","has_retinopathy","has_neuropathy","has_cardiovascular"]

# ---------- ICD-9 helpers ----------
def to_float_code(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if s == "": return np.nan
    try:
        if s[0] in ("V","E"):
            s_num = "".join(ch for ch in s[1:] if (ch.isdigit() or ch == "."))
            return float(s_num) if s_num else np.nan
        return float(s)
    except Exception:
        return np.nan

def is_nephropathy(code):    return (not np.isnan(code)) and ((580 <= code <= 589) or (abs(code - 250.4) < 1e-6))
def is_retinopathy(code):    return (not np.isnan(code)) and ((362.00 <= code <= 362.07) or (abs(code - 250.5) < 1e-6))
def is_neuropathy(code):     return (not np.isnan(code)) and (abs(code - 356.9) < 1e-6 or abs(code - 337.1) < 1e-6 or abs(code - 250.6) < 1e-6)
def is_cardiovascular(code): return (not np.isnan(code)) and ((390 <= code <= 459) or (abs(code - 250.7) < 1e-6))

def build_flags(df):
    def flags_from_row(row):
        codes = [to_float_code(row.get("diag_1")), to_float_code(row.get("diag_2")), to_float_code(row.get("diag_3"))]
        neph  = any(is_nephropathy(c)    for c in codes)
        reti  = any(is_retinopathy(c)    for c in codes)
        neuro = any(is_neuropathy(c)     for c in codes)
        cardio= any(is_cardiovascular(c) for c in codes)
        anyc  = neph or reti or neuro or cardio
        return pd.Series({
            "has_nephropathy": int(neph),
            "has_retinopathy": int(reti),
            "has_neuropathy": int(neuro),
            "has_cardiovascular": int(cardio),
            "has_any_complication": int(anyc)
        })
    flags = df.apply(flags_from_row, axis=1)
    return pd.concat([df, flags], axis=1)

# ---------- utils ----------
def cap_categories(s: pd.Series, top_n=30, fill="Unknown"):
    s = s.astype("string").fillna(fill)
    counts = s.value_counts()
    keep = set(counts.head(top_n).index)
    return s.where(s.isin(keep), other="Other")

def natural_age_key(a):
    s = str(a).strip()
    if s.startswith("[") and "-" in s:
        try: return int(s.strip("[]").split("-")[0])
        except Exception: return 999
    return 999

def plot_grouped_bar(df_sum, comp_key, title, savepath):
    key = f"mean_pred_{comp_key}"
    sub = df_sum[["age","gender", key]].dropna()
    ages = sorted(sub["age"].unique(), key=natural_age_key)
    genders = [g for g in ["Male","Female","Unknown/Invalid"] if g in sub["gender"].unique()]
    if not genders: genders = sorted(sub["gender"].unique())
    x = np.arange(len(ages)); width = 0.8 / max(1, len(genders))
    plt.figure()
    for i, g in enumerate(genders):
        y_vals = [sub[(sub["age"] == a) & (sub["gender"] == g)][key].mean() for a in ages]
        plt.bar(x + i * width, y_vals, width, label=str(g))
    plt.xticks(x + (len(genders)-1)*width/2, ages, rotation=45, ha="right")
    plt.xlabel("Age group"); plt.ylabel("Predicted probability"); plt.title(title)
    plt.legend(); plt.tight_layout(); plt.savefig(savepath); plt.close()

# ---------- adaptive SMOTE builder ----------
def make_smote_for_split(y_series, for_tuning=False):
    """
    Create a SMOTE configured safely for the given y.
    - Only oversamples when minority truly exists and has >= 2 samples.
    - Uses sampling_strategy='auto' to avoid 'ratio' errors.
    - Sets k_neighbors <= minority_count - 1.
    Returns a SMOTE instance or None (meaning: skip sampling).
    """
    y = y_series.to_numpy()
    # ensure binary labels 0/1 exist
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return None
    minority_count = counts.min()
    majority_count = counts.max()

    # If already balanced or minority too tiny to define neighbors, skip
    if minority_count < 2:
        return None

    # choose k safely
    k = min(5 if not for_tuning else 3, minority_count - 1)
    if k < 1:
        return None

    # 'auto' grows minority to majority; never removes samples
    return SMOTE(
        random_state=RANDOM_SEED,
        k_neighbors=k,
        sampling_strategy='auto'
    )

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/diabetic_data.csv", help="Path to diabetic_data.csv")
    parser.add_argument("--out", default="outputs_user", help="Output folder")
    parser.add_argument("--full", action="store_true", help="Use full dataset (disable quick sample)")
    parser.add_argument("--n_estimators", type=int, default=120, help="RF trees (default 120)")
    parser.add_argument("--tune", action="store_true", help="Run fast RandomizedSearchCV")
    parser.add_argument("--try_xgb", action="store_true", help="Try XGBoost (if installed)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    assert os.path.exists(args.data), f"Could not find {args.data}"

    # 1) Load, clean
    df = pd.read_csv(args.data).replace("?", np.nan)
    med_cols = [c for c in POSSIBLE_MEDS if c in df.columns]
    use_cols = [c for c in BASE_FEATURES if c in df.columns] + med_cols
    df = df[use_cols].copy()

    # 2) Sample for speed unless --full
    if not args.full and len(df) > 20000:
        df = df.sample(20000, random_state=RANDOM_SEED).reset_index(drop=True)

    # 3) Labels
    df = build_flags(df)

    # 4) Categorical hygiene
    if "medical_specialty" in df.columns: df["medical_specialty"] = cap_categories(df["medical_specialty"], top_n=30)
    if "age" in df.columns:               df["age"] = df["age"].astype("string").fillna("Unknown")
    if "gender" in df.columns:            df["gender"] = df["gender"].astype("string").fillna("Unknown/Invalid")
    if "readmitted" in df.columns:        df["readmitted"] = df["readmitted"].astype("string").fillna("NO")
    if "A1Cresult" in df.columns:         df["A1Cresult"] = df["A1Cresult"].astype("string").fillna("None")
    if "change" in df.columns:            df["change"] = df["change"].astype("string").fillna("No")
    if "diabetesMed" in df.columns:       df["diabetesMed"] = df["diabetesMed"].astype("string").fillna("No")

    # 5) Features
    cat_cols = [c for c in ["gender","age","medical_specialty","readmitted","A1Cresult","change","diabetesMed"] + med_cols if c in df.columns]
    num_cols = [c for c in ["time_in_hospital","num_lab_procedures","num_medications","num_procedures","number_inpatient","number_outpatient","number_emergency"] if c in df.columns]
    X = df[cat_cols + num_cols].copy()
    y_any = df["has_any_complication"]

    # 6) Preprocess (OneHotEncoder version-safe)
    enc_kwargs = {"handle_unknown": "ignore"}
    try:
        if version.parse(sklearn.__version__) >= version.parse("1.2"):
            enc_kwargs["sparse_output"] = True
        else:
            enc_kwargs["sparse"] = True
    except Exception:
        enc_kwargs["sparse"] = True

    ohe = OneHotEncoder(**enc_kwargs)
    num_pipe = SkPipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = SkPipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", ohe)])
    preprocess = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")

    # 7) Split
    X_tr, X_te, idx_tr, idx_te = train_test_split(
        X, np.arange(len(X)), test_size=0.2, random_state=RANDOM_SEED, stratify=y_any
    )

    # 8) Base models
    rf_base = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=None, min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_SEED
    )
    dt_base = DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=RANDOM_SEED)
    lr_base = LogisticRegression(max_iter=400, class_weight="balanced", solver="liblinear")

    xgb_base = None
    if args.try_xgb and HAS_XGB:
        xgb_base = XGBClassifier(
            n_estimators=max(250, args.n_estimators), learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            n_jobs=-1, random_state=RANDOM_SEED, eval_metric="logloss"
        )

    # helpers
    def with_pipe(clf, smote):
        """Build a pipeline with optional SMOTE. If smote is None, skip it."""
        if smote is None:
            return SkPipeline([("prep", preprocess), ("clf", clf)])
        return Pipeline([("prep", preprocess), ("smote", smote), ("clf", clf)])

    def rf_distributions():
        return {
            "clf__n_estimators": sp_randint(150, max(250, args.n_estimators) + 1),
            "clf__max_depth": [6, 8, 12, 16, None],
            "clf__min_samples_split": sp_randint(2, 11),
            "clf__min_samples_leaf": sp_randint(1, 5),
            "clf__bootstrap": [True, False],
        }

    def xgb_distributions():
        return {
            "clf__n_estimators": sp_randint(max(200, args.n_estimators), max(450, args.n_estimators) + 1),
            "clf__max_depth": sp_randint(4, 11),
            "clf__learning_rate": sp_uniform(0.03, 0.12),
            "clf__subsample": sp_uniform(0.6, 0.4),
            "clf__colsample_bytree": sp_uniform(0.6, 0.4),
            "clf__reg_lambda": sp_uniform(0.5, 1.5),
        }

    rows, probas, best_models_per_target = [], {}, {}

    # 9) Train/eval loop
    for tgt in TARGETS:
        y = df[tgt].astype(int)
        y_tr, y_te = y.iloc[idx_tr], y.iloc[idx_te]

        if y_te.sum() == 0 or (len(y_te) - y_te.sum()) == 0:
            print(f"Skip {tgt}: no positive/negative examples in test split.")
            continue

        # Build adaptive SMOTE objects for training and tuning
        smote_train = make_smote_for_split(y_tr, for_tuning=False)
        smote_tune  = make_smote_for_split(y_tr, for_tuning=True)

        candidates = {
            "rf":     with_pipe(rf_base, smote_train),
            "dtree":  with_pipe(dt_base, smote_train),
            "logreg": with_pipe(lr_base, smote_train),
        }
        if xgb_base is not None:
            candidates["xgb"] = with_pipe(xgb_base, smote_train)

        tuned = {}
        if args.tune:
            # small stratified subset for tuning (<= 8000 rows)
            n_tune = min(8000, X_tr.shape[0])
            X_tune, _, y_tune, _ = train_test_split(
                X_tr, y_tr, train_size=n_tune, stratify=y_tr, random_state=RANDOM_SEED
            )

            # RF fast tuning
            rf_search = RandomizedSearchCV(
                estimator=with_pipe(RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1), smote_tune),
                param_distributions=rf_distributions(),
                n_iter=24, scoring="f1", cv=3, n_jobs=-1, verbose=1, random_state=RANDOM_SEED
            )
            print(f"[TUNE-FAST] {tgt} → RF on {len(X_tune)} rows…")
            rf_search.fit(X_tune, y_tune)
            tuned["rf"] = rf_search.best_estimator_
            print("  RF best:", rf_search.best_params_)

            if xgb_base is not None:
                xgb_search = RandomizedSearchCV(
                    estimator=with_pipe(XGBClassifier(random_state=RANDOM_SEED, eval_metric="logloss", n_jobs=-1), smote_tune),
                    param_distributions=xgb_distributions(),
                    n_iter=24, scoring="f1", cv=3, n_jobs=-1, verbose=1, random_state=RANDOM_SEED
                )
                print(f"[TUNE-FAST] {tgt} → XGB on {len(X_tune)} rows…")
                xgb_search.fit(X_tune, y_tune)
                tuned["xgb"] = xgb_search.best_estimator_
                print("  XGB best:", xgb_search.best_params_)

        use_models = tuned if args.tune else candidates

        perf_for_target = []
        for key, pipe in use_models.items():
            start = time.time()
            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_te)
            y_proba = pipe.predict_proba(X_te)[:, 1]
            dur = time.time() - start

            report = classification_report(y_te, y_pred, zero_division=0, output_dict=True)
            try:    auc_val = roc_auc_score(y_te, y_proba)
            except: auc_val = float("nan")

            rows.append({
                "target": tgt, "model": key,
                "accuracy": report.get("accuracy", float("nan")),
                "precision": report.get("1", {}).get("precision", float("nan")),
                "recall": report.get("1", {}).get("recall", float("nan")),
                "f1": report.get("1", {}).get("f1-score", float("nan")),
                "roc_auc": auc_val,
                "train_seconds": round(dur, 2)
            })
            probas[(tgt, key)] = (y_te.values, y_proba)
            perf_for_target.append((key, report.get("1", {}).get("f1-score", 0.0), pipe))

        perf_for_target.sort(key=lambda x: x[1], reverse=True)
        best_key, best_f1, best_pipe = perf_for_target[0]
        best_models_per_target[tgt] = (best_key, best_pipe)

    # Save metrics
    metrics_df = pd.DataFrame(rows).sort_values(["target","f1"], ascending=[True, False])
    metrics_path = os.path.join(args.out, "model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[Saved] {metrics_path}")

    # ROC curves
    for tgt in sorted(set(metrics_df["target"])):
        plt.figure()
        for key, label in [("rf","Random Forest"), ("dtree","Decision Tree"), ("logreg","Logistic Regression"), ("xgb","XGBoost")]:
            if (tgt, key) not in probas: continue
            y_true, y_prob = probas[(tgt, key)]
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc:.3f})")
        plt.plot([0,1],[0,1], lw=1, linestyle="--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"ROC — {tgt.replace('has_','').title()}"); plt.legend(loc="lower right")
        plt.tight_layout()
        outp = os.path.join(args.out, f"ROC_{tgt}.png")
        plt.savefig(outp); plt.close()
        print(f"[Saved] {outp}")

    # Refit best model on ALL rows (for age×gender charts)
    from functools import reduce
    for tgt, (best_key, best_pipe) in best_models_per_target.items():
        clf = best_pipe.named_steps["clf"]
        pred_pipe = SkPipeline([("prep", preprocess), ("clf", clf)])
        pred_pipe.fit(X, df[tgt].astype(int))
        df[f"pred_{tgt}"] = pred_pipe.predict_proba(X)[:, 1]

    frames = []
    for tgt in best_models_per_target.keys():
        g = df.groupby(["age","gender"], dropna=False)[f"pred_{tgt}"].mean().reset_index()
        g = g.rename(columns={f"pred_{tgt}": f"mean_pred_{tgt}"})
        frames.append(g)

    if frames:
        age_gender_summary = reduce(lambda L, R: pd.merge(L, R, on=["age","gender"], how="outer"), frames)
        summary_path = os.path.join(args.out, "predicted_risk_by_age_gender_bestmodel.csv")
        age_gender_summary.to_csv(summary_path, index=False)
        print(f"[Saved] {summary_path}")

        for tgt, title in [
            ("has_nephropathy",   "Predicted risk — Nephropathy (best model)"),
            ("has_retinopathy",   "Predicted risk — Retinopathy (best model)"),
            ("has_neuropathy",    "Predicted risk — Neuropathy (best model)"),
            ("has_cardiovascular","Predicted risk — Cardiovascular (best model)")
        ]:
            outp = os.path.join(args.out, f"best_{tgt}_by_age_gender.png")
            plot_grouped_bar(age_gender_summary, tgt, title, outp)
            print(f"[Saved] {outp}")

    print("✅ Done.")

if __name__ == "__main__":
    main()