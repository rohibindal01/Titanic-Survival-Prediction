"""
Titanic Survival Prediction - Machine Learning Project
Dataset: https://www.kaggle.com/competitions/titanic
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_data(train_path="data/train.csv", test_path="data/test.csv"):
    """Load Titanic dataset from local CSV files."""
    print("📦 Loading data...")
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    print(f"  Train shape : {train_df.shape}")
    print(f"  Test  shape : {test_df.shape}")
    return train_df, test_df


# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────

def run_eda(df: pd.DataFrame, output_dir="outputs"):
    """Generate EDA plots and save them to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    print("\n🔍 Running EDA...")

    # --- Survival rate by class & sex ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(data=df, x="Pclass", hue="Survived", ax=axes[0], palette="Set2")
    axes[0].set_title("Survival by Passenger Class")
    sns.countplot(data=df, x="Sex", hue="Survived", ax=axes[1], palette="Set1")
    axes[1].set_title("Survival by Sex")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/survival_by_class_sex.png", dpi=150)
    plt.close()

    # --- Age distribution ---
    fig, ax = plt.subplots(figsize=(8, 4))
    df["Age"].hist(bins=30, edgecolor="black", ax=ax, color="steelblue")
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/age_distribution.png", dpi=150)
    plt.close()

    # --- Correlation heatmap ---
    fig, ax = plt.subplots(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=150)
    plt.close()

    print(f"  EDA plots saved to '{output_dir}/'")


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features and handle missing values."""
    df = df.copy()

    # Fill missing values
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)

    # New features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)
    df["Title"]      = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df["Title"]      = df["Title"].replace(
        ["Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"], "Rare"
    )
    df["Title"]      = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    # Encode categoricals
    le = LabelEncoder()
    df["Sex"]      = le.fit_transform(df["Sex"])
    df["Embarked"] = le.fit_transform(df["Embarked"])
    df["Title"]    = le.fit_transform(df["Title"])

    # Age & Fare bands
    df["AgeBand"]  = pd.cut(df["Age"],  bins=5, labels=False)
    df["FareBand"] = pd.qcut(df["Fare"], q=4, labels=False)

    return df


# ─────────────────────────────────────────────
# 4. PREPARE X / y
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
    "Embarked", "FamilySize", "IsAlone", "Title", "AgeBand", "FareBand"
]

def get_Xy(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df["Survived"] if "Survived" in df.columns else None
    return X, y


# ─────────────────────────────────────────────
# 5. BUILD & TRAIN MODELS
# ─────────────────────────────────────────────

def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=200, random_state=42))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(n_estimators=200, random_state=42))
        ]),
    }


def train_and_evaluate(X_train, X_val, y_train, y_val, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    models   = build_models()
    results  = {}

    print("\n🚂 Training models...\n")
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
        acc       = accuracy_score(y_val, y_pred)
        auc       = roc_auc_score(y_val, pipeline.predict_proba(X_val)[:, 1])

        results[name] = {
            "pipeline": pipeline,
            "accuracy": acc,
            "roc_auc":  auc,
            "cv_mean":  cv_scores.mean(),
            "cv_std":   cv_scores.std(),
        }

        print(f"  ▸ {name}")
        print(f"      Validation Accuracy : {acc:.4f}")
        print(f"      ROC-AUC             : {auc:.4f}")
        print(f"      CV Accuracy         : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

    # --- Pick best model ---
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best      = results[best_name]
    print(f"🏆 Best model: {best_name}  (AUC = {best['roc_auc']:.4f})\n")

    # --- Plots for best model ---
    _plot_confusion_matrix(best["pipeline"], X_val, y_val, best_name, output_dir)
    _plot_roc_curve(results, X_val, y_val, output_dir)
    if hasattr(best["pipeline"].named_steps["clf"], "feature_importances_"):
        _plot_feature_importance(best["pipeline"], best_name, output_dir)

    # --- Save best model ---
    os.makedirs("model", exist_ok=True)
    joblib.dump(best["pipeline"], "model/best_model.pkl")
    print("💾 Best model saved to 'model/best_model.pkl'")

    return best["pipeline"], results


def _plot_confusion_matrix(pipeline, X_val, y_val, name, output_dir):
    cm = confusion_matrix(y_val, pipeline.predict(X_val))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix – {name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=150)
    plt.close()


def _plot_roc_curve(results, X_val, y_val, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_val, r["pipeline"].predict_proba(X_val)[:, 1])
        ax.plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – All Models")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curves.png", dpi=150)
    plt.close()


def _plot_feature_importance(pipeline, name, output_dir):
    importances = pipeline.named_steps["clf"].feature_importances_
    feat_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis", ax=ax)
    ax.set_title(f"Feature Importances – {name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png", dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# 6. PREDICT & SAVE SUBMISSION
# ─────────────────────────────────────────────

def make_submission(pipeline, test_df_engineered, raw_test_df, output_dir="outputs"):
    X_test, _ = get_Xy(test_df_engineered)
    preds = pipeline.predict(X_test)
    submission = pd.DataFrame({
        "PassengerId": raw_test_df["PassengerId"],
        "Survived":    preds
    })
    path = f"{output_dir}/submission.csv"
    submission.to_csv(path, index=False)
    print(f"\n📄 Submission saved to '{path}'")
    return submission


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  🚢 Titanic Survival Prediction – ML Pipeline")
    print("=" * 55)

    # Load
    train_df, test_df = load_data()

    # EDA
    run_eda(train_df)

    # Feature engineering
    train_eng = engineer_features(train_df)
    test_eng  = engineer_features(test_df)

    # Split
    X, y = get_Xy(train_eng)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train & evaluate
    best_pipeline, results = train_and_evaluate(X_train, X_val, y_train, y_val)

    # Generate submission
    make_submission(best_pipeline, test_eng, test_df)

    print("\n✅ Pipeline complete! Check 'outputs/' for plots and 'model/' for the saved model.")


if __name__ == "__main__":
    main()
