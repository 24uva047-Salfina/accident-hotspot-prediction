"""Train the district accident-risk model (LightGBM) the honest way.

    python train_model.py

* features : 2020 statistics (scale-free, see features.py)
* target   : risk class of 2021
* checks   : stratified cross-validation + a "same class as last year" baseline
* output   : model/accident_risk_model.pkl  (model + feature list + metrics)
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

try:
    from lightgbm import LGBMClassifier
except ImportError:
    raise SystemExit("LightGBM missing -> run: pip install -r requirements.txt")

from features import (FEATURES, RISK_LABELS, build_features,
                      label_from_score, risk_score)

DATA_PATH = "data/preprocessed_tamil_nadu_accidents.csv"
MODEL_PATH = "model/accident_risk_model.pkl"


def new_model():
    return LGBMClassifier(
        objective="multiclass",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=4,
        min_child_samples=2,      # only 36 districts -> keep leaves tiny
        subsample=0.9, subsample_freq=1,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )


df = pd.read_csv(DATA_PATH)
assert (df["coordinate_status"] == "Valid").all(), "Fix invalid coordinates first"
print(f"Districts: {len(df)}")

# ---------- features from 2020, target from 2021 --------------------------
X = build_features(df, "2020_fatal", "total__2020")
y = label_from_score(risk_score(df["total__2021"]))
print("Target distribution:", {RISK_LABELS[k]: int(v) for k, v in y.value_counts().sort_index().items()})

# ---------- baseline: "district stays in the same class" -------------------
y_persist = label_from_score(risk_score(df["total__2020"]))
base_acc = accuracy_score(y, y_persist)
base_f1 = f1_score(y, y_persist, average="macro")

# ---------- cross-validation (never score on the training rows!) ----------
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=42)
accs, f1s = [], []
cm = np.zeros((3, 3), dtype=int)
for tr, te in cv.split(X, y):
    m = new_model().fit(X.iloc[tr], y.iloc[tr])
    p = m.predict(X.iloc[te])
    accs.append(accuracy_score(y.iloc[te], p))
    f1s.append(f1_score(y.iloc[te], p, average="macro", labels=[0, 1, 2]))
    cm += confusion_matrix(y.iloc[te], p, labels=[0, 1, 2])

print("\n=== RESULTS (cross-validated, unseen districts) ===")
print(f"LightGBM  accuracy : {np.mean(accs)*100:.1f}%  (+/- {np.std(accs)*100:.1f})")
print(f"LightGBM  macro-F1 : {np.mean(f1s):.3f}")
print(f"Baseline  accuracy : {base_acc*100:.1f}%  (same class as last year)")
print(f"Baseline  macro-F1 : {base_f1:.3f}")
print("Confusion matrix (rows = actual, cols = predicted; Low/Medium/High):")
print(cm)

# ---------- final model on all rows, then save ----------------------------
model = new_model().fit(X, y)
imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nFeature importance:\n", imp.to_string())

os.makedirs("model", exist_ok=True)
joblib.dump({
    "model": model,
    "features": FEATURES,
    "classes": RISK_LABELS,
    "metrics": {"cv_accuracy": float(np.mean(accs)), "cv_macro_f1": float(np.mean(f1s)),
                "baseline_accuracy": float(base_acc), "baseline_macro_f1": float(base_f1)},
}, MODEL_PATH)
print(f"\nSaved -> {MODEL_PATH}")
