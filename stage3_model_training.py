import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# load dataset
# ==========================
DATASET_PATH = "dataset/training_dataset.csv"
df = pd.read_csv(DATASET_PATH)

# ========================
# set target and features
# ========================
TARGET = "auto_label"

FEATURES = [
    "age",
    "glasses",
    "hours_sleep",
    "caffeine_last_6h",
    "task",
    "lighting",

    "baseline_blinks",
    "total_blinks",
    "avg_blinks_per_min",
    "std_blinks_per_min",
    "blink_baseline_ratio"
]

X = df[FEATURES]
y = df[TARGET]

# ===================
# feature types
# ===================
numeric_features = [
    "age",
    "hours_sleep",
    "baseline_blinks",
    "total_blinks",
    "avg_blinks_per_min",
    "std_blinks_per_min",
    "blink_baseline_ratio"
]

categorical_features = [
    "glasses",
    "caffeine_last_6h",
    "task",
    "lighting"
]

# =================
# preprocessing
# =================
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ===================
# train/test split
# ===================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
)

# ==============
# models
# ==============
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "SVM": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42
    )
}

# ========================
# training + evaluation
# ========================
results = []

for name, model in models.items():
    print(f"\n🔹 Training {name}")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1-score": f1
    })

    print("Accuracy:", acc)
    print("F1-score:", f1)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)

    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=pipeline.classes_,
                yticklabels=pipeline.classes_
                )
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# =======================
# results comparison
# =======================
results_df = pd.DataFrame(results)

results_df.set_index("Model")[["Accuracy", "F1-score"]].plot(
    kind="bar",
    figsize=(8, 5),
    ylim=(0, 1),
    rot=0
)

plt.title("Model Comparison – Ocular Fatigue Classification")
plt.ylabel("Score")
plt.grid(axis="y")
plt.tight_layout()
plt.show()