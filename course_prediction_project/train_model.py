"""
train_model.py
--------------
Trains and evaluates multiple ML classifiers to predict the recommended course
for a student based on academic performance, interests, and achievements.

Models trained:
    1. Logistic Regression
    2. Decision Tree Classifier
    3. Random Forest Classifier
    4. Support Vector Machine (SVM)

Evaluation:
    - Accuracy Score
    - Confusion Matrix
    - ROC Curve (One-vs-Rest)
    - AUC Score

Optimization:
    - GridSearchCV on Random Forest
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────
SEED           = 42
DATASET_PATH   = "dataset.csv"
MODEL_PATH     = "model.pkl"
SCALER_PATH    = "scaler.pkl"
ENCODER_PATH   = "label_encoder.pkl"
FIGURES_DIR    = "figures"

os.makedirs(FIGURES_DIR, exist_ok=True)

# Palette for consistent colouring across plots
PALETTE = sns.color_palette("tab10")

# ────────────────────────────────────────────────────────────
# 1. Load Dataset
# ────────────────────────────────────────────────────────────
print("=" * 60)
print("  Career Counselling System – Model Training")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)
print(f"\n[1] Dataset loaded  →  Shape: {df.shape}")
print(df.head(3))

# ────────────────────────────────────────────────────────────
# 2. Preprocessing
# ────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "math_score", "physics_score", "chemistry_score",
    "biology_score", "english_score",
    "programming_interest", "analytical_skills",
    "creativity_level", "communication_skills", "leadership_skills",
    "sports", "olympiad", "projects_done",
]
TARGET_COL = "course"

X = df[FEATURE_COLS].values
y_raw = df[TARGET_COL].values

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y_raw)
CLASSES = le.classes_
N_CLASSES = len(CLASSES)

print(f"\n[2] Classes ({N_CLASSES}): {list(CLASSES)}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler and encoder (needed by Streamlit app)
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)
with open(ENCODER_PATH, "wb") as f:
    pickle.dump(le, f)

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=SEED, stratify=y
)
print(f"\n[3] Split  →  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ────────────────────────────────────────────────────────────
# 3. Model Definitions
# ────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, random_state=SEED
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=SEED, max_depth=10
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, random_state=SEED
    ),
    "SVM": SVC(
        kernel="rbf", probability=True, random_state=SEED
    ),
}

# ────────────────────────────────────────────────────────────
# 4. Training & Evaluation
# ────────────────────────────────────────────────────────────
results = {}  # model_name → dict of metrics
trained_models = {}

print("\n[4] Training models …\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)  # probabilities for ROC

    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)

    results[name] = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "model": model,
    }
    trained_models[name] = model
    print(f"  {name:<25}  Accuracy = {acc:.4f}")

# ────────────────────────────────────────────────────────────
# 5. Confusion Matrices
# ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Confusion Matrices – All Models", fontsize=16, fontweight="bold", y=1.01)

for ax, (name, res) in zip(axes.flat, results.items()):
    sns.heatmap(
        res["confusion_matrix"],
        annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES,
        ax=ax, linewidths=0.5,
    )
    ax.set_title(f"{name}  (Acc = {res['accuracy']:.3f})", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0,  fontsize=8)

plt.tight_layout()
cm_path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[5] Confusion matrices saved → {cm_path}")

# ────────────────────────────────────────────────────────────
# 6. ROC Curves (One-vs-Rest for each model)
# ────────────────────────────────────────────────────────────
y_test_bin = label_binarize(y_test, classes=list(range(N_CLASSES)))

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("ROC Curves (One-vs-Rest) – All Models", fontsize=16, fontweight="bold", y=1.01)

for ax, (name, res) in zip(axes.flat, results.items()):
    macro_auc_vals = []

    for i, cls_name in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], res["y_prob"][:, i])
        roc_auc      = auc(fpr, tpr)
        macro_auc_vals.append(roc_auc)
        ax.plot(fpr, tpr, label=f"{cls_name} (AUC={roc_auc:.2f})", linewidth=1.5)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_title(f"{name}  (macro-AUC={np.mean(macro_auc_vals):.3f})", fontsize=11)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=6, loc="lower right")
    ax.grid(alpha=0.3)

    results[name]["macro_auc"] = np.mean(macro_auc_vals)

plt.tight_layout()
roc_path = os.path.join(FIGURES_DIR, "roc_curves.png")
plt.savefig(roc_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[6] ROC curves saved → {roc_path}")

# ────────────────────────────────────────────────────────────
# 7. Accuracy Comparison Bar Chart
# ────────────────────────────────────────────────────────────
model_names = list(results.keys())
accuracies  = [results[m]["accuracy"] for m in model_names]
auc_scores  = [results[m]["macro_auc"] for m in model_names]

x = np.arange(len(model_names))
width = 0.38

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, accuracies, width, label="Accuracy",  color="#4C72B0", edgecolor="white")
bars2 = ax.bar(x + width/2, auc_scores,  width, label="Macro AUC", color="#55A868", edgecolor="white")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.set_title("Model Comparison – Accuracy & Macro AUC", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
bar_path = os.path.join(FIGURES_DIR, "model_comparison.png")
plt.savefig(bar_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[7] Model comparison chart saved → {bar_path}")

# ────────────────────────────────────────────────────────────
# 8. Summary Table
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Model Performance Summary")
print("=" * 60)
print(f"{'Model':<25} {'Accuracy':>10} {'Macro AUC':>12}")
print("-" * 50)
for name in model_names:
    print(f"  {name:<23} {results[name]['accuracy']:>10.4f} {results[name]['macro_auc']:>12.4f}")

# ────────────────────────────────────────────────────────────
# 9. Best Model (before tuning)
# ────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["accuracy"])
best_acc  = results[best_name]["accuracy"]
print(f"\n[8] Best model (pre-tuning): {best_name}  (Accuracy = {best_acc:.4f})")

# ────────────────────────────────────────────────────────────
# 10. GridSearchCV – Hyperparameter Tuning on Random Forest
# ────────────────────────────────────────────────────────────
print("\n[9] Running GridSearchCV on Random Forest …")

param_grid = {
    "n_estimators":    [100, 200, 300],
    "max_depth":       [None, 10, 20],
    "min_samples_split": [2, 5, 10],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=SEED),
    param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0,
)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
best_params = grid_search.best_params_
tuned_acc = accuracy_score(y_test, best_rf.predict(X_test))

pre_tuned_acc = results["Random Forest"]["accuracy"]

print(f"   Best params    : {best_params}")
print(f"   Accuracy before tuning : {pre_tuned_acc:.4f}")
print(f"   Accuracy after  tuning : {tuned_acc:.4f}")
print(f"   Improvement            : {tuned_acc - pre_tuned_acc:+.4f}")

# ────────────────────────────────────────────────────────────
# 11. Select & Save Final Best Model
# ────────────────────────────────────────────────────────────
# Use tuned RF if it beats the previous best, else keep original best
final_model = best_rf if tuned_acc >= best_acc else trained_models[best_name]
final_acc   = max(tuned_acc, best_acc)
final_label = (
    f"Tuned Random Forest (acc={tuned_acc:.4f})"
    if tuned_acc >= best_acc
    else f"{best_name} (acc={best_acc:.4f})"
)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(final_model, f)

print(f"\n[10] Final model selected : {final_label}")
print(f"     Saved to             : {MODEL_PATH}")

# ────────────────────────────────────────────────────────────
# 12. Feature Importances (Random Forest)
# ────────────────────────────────────────────────────────────
importances = best_rf.feature_importances_
feat_df = pd.DataFrame({
    "Feature":   FEATURE_COLS,
    "Importance": importances,
}).sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(feat_df)))
ax.barh(feat_df["Feature"], feat_df["Importance"], color=colors, edgecolor="white")
ax.set_xlabel("Importance Score")
ax.set_title("Feature Importances (Tuned Random Forest)", fontsize=13, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fi_path = os.path.join(FIGURES_DIR, "feature_importances.png")
plt.savefig(fi_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[11] Feature importances chart saved → {fi_path}")

# ────────────────────────────────────────────────────────────
# 13. Classification Report for the Final Model
# ────────────────────────────────────────────────────────────
y_pred_final = final_model.predict(X_test)
print("\n[12] Classification Report (Final Model):")
print(classification_report(y_test, y_pred_final, target_names=CLASSES))

print("\n✅ Training pipeline complete!")
print("   Artefacts: model.pkl, scaler.pkl, label_encoder.pkl")
print("   Figures  :", FIGURES_DIR)
