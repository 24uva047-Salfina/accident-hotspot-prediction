import pandas as pd
import joblib
import os

from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "data/preprocessed_tamil_nadu_accidents.csv"
MODEL_PATH = "model/accident_lightgbm_model.pkl"

print("\n======================================")
print(" TRAINING NEW ACCIDENT RISK MODEL")
print("======================================")

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)


# ============================================================
# 2. FEATURES
# ============================================================

FEATURES = [
    "2020_fatal",
    "2021_fatal",
    "2020_nonfatal",
    "2021_nonfatal",
    "total__2020",
    "total__2021",
    "accident_change",
    "accident_change_percent",
    "latitude",
    "longitude"
]


# ============================================================
# 3. CHECK FEATURES
# ============================================================

missing_features = [
    col for col in FEATURES
    if col not in df.columns
]

if missing_features:

    print("\nERROR: Missing features:")
    print(missing_features)

    raise SystemExit


print("\nAll required features are available.")


# ============================================================
# 4. CREATE RISK LABEL
# ============================================================

# Low    = 0
# Medium = 1
# High   = 2

risk_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

df["target"] = df["risk_level"].map(risk_mapping)


# Check invalid labels
if df["target"].isna().any():

    print("\nERROR: Unknown risk labels found.")

    print(
        df.loc[
            df["target"].isna(),
            "risk_level"
        ]
    )

    raise SystemExit


# ============================================================
# 5. DISPLAY CLASS DISTRIBUTION
# ============================================================

print("\nRisk distribution:")

print(
    df["risk_level"].value_counts()
)

print("\nEncoded distribution:")

print(
    df["target"].value_counts().sort_index()
)


# ============================================================
# 6. PREPARE X AND Y
# ============================================================

X = df[FEATURES].copy()

y = df["target"].astype(int)


# ============================================================
# 7. TRAIN LIGHTGBM
# ============================================================

print("\nTraining LightGBM...")


model = LGBMClassifier(

    objective="multiclass",

    num_class=3,

    n_estimators=150,

    learning_rate=0.03,

    max_depth=3,

    num_leaves=7,

    min_child_samples=3,

    subsample=0.9,

    colsample_bytree=0.9,

    class_weight="balanced",

    random_state=42,

    verbosity=-1
)


model.fit(
    X,
    y
)


# ============================================================
# 8. PREDICTION ON TRAINING DATA
# ============================================================

predictions = model.predict(X)

predictions = predictions.astype(int)


accuracy = accuracy_score(
    y,
    predictions
)


print("\n======================================")
print(" TRAINING RESULT")
print("======================================")

print(
    f"\nTraining Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y,
        predictions,
        target_names=[
            "Low",
            "Medium",
            "High"
        ],
        zero_division=0
    )
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y,
        predictions
    )
)


# ============================================================
# 11. SHOW DISTRICT PREDICTIONS
# ============================================================

result = pd.DataFrame({

    "district":
        df["district"],

    "actual":
        df["risk_level"],

    "predicted_class":
        predictions

})


result["predicted_risk"] = result[
    "predicted_class"
].map({

    0: "Low",
    1: "Medium",
    2: "High"

})


print("\n======================================")
print(" DISTRICT PREDICTIONS")
print("======================================")

print(
    result[
        [
            "district",
            "actual",
            "predicted_risk"
        ]
    ].to_string(index=False)
)


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "feature":
        FEATURES,

    "importance":
        model.feature_importances_

})


importance = importance.sort_values(
    "importance",
    ascending=False
)


print("\n======================================")
print(" FEATURE IMPORTANCE")
print("======================================")

print(
    importance.to_string(index=False)
)


# ============================================================
# 13. SAVE MODEL
# ============================================================

os.makedirs(
    "model",
    exist_ok=True
)


joblib.dump(
    model,
    MODEL_PATH
)


print("\n======================================")
print(" MODEL SAVED SUCCESSFULLY")
print("======================================")

print(
    "\nSaved to:"
)

print(
    MODEL_PATH
)

print(
    "\nModel classes:"
)

print(
    model.classes_
)

print("\nTraining completed successfully!")