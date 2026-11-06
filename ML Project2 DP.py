import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# 1. Load Dataset

file_path = r"C:\Users\madhu\OneDrive\Documents\ML PROJECT DATASETS\water_pollution_disease.csv"

df = pd.read_csv(file_path, encoding="utf-8-sig")
df.columns = df.columns.str.strip()

print("DATASET INFORMATION")
print("-------------------")
print("Dataset Shape:", df.shape)
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())


# 2. Select Input Features

features = [
    "Contaminant Level (ppm)",
    "pH Level",
    "Turbidity (NTU)",
    "Dissolved Oxygen (mg/L)",
    "Nitrate Level (mg/L)",
    "Lead Concentration (µg/L)",
    "Bacteria Count (CFU/mL)",
    "Access to Clean Water (% of Population)",
    "Sanitation Coverage (% of Population)",
    "Rainfall (mm per year)",
    "Temperature (°C)",
    "Healthcare Access Index (0-100)",
    "GDP per Capita (USD)",
    "Population Density (people per km²)"
]

disease_columns = [
    "Diarrheal Cases per 100,000 people",
    "Cholera Cases per 100,000 people",
    "Typhoid Cases per 100,000 people"
]

X = df[features].copy()

for col in disease_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 3. Create Disease Risk Target

df["Average Disease Rate"] = df[disease_columns].mean(axis=1)

valid_rows = df["Average Disease Rate"].notna()

X = X.loc[valid_rows].copy()
disease_rate = df.loc[valid_rows, "Average Disease Rate"]

threshold = disease_rate.median()

y = (disease_rate >= threshold).astype(int)

print("\nTARGET INFORMATION")
print("------------------")
print("Median Disease Rate:", threshold)
print("0 = Low Risk")
print("1 = High Risk")
print("\nRisk Distribution:")
print(y.value_counts())


# 4. Exploratory Data Analysis

plt.figure(figsize=(8, 5))
sns.histplot(disease_rate, bins=30, kde=True)
plt.title("Distribution of Average Disease Rates")
plt.xlabel("Average Disease Rate")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title("Distribution of Disease Risk Categories")
plt.xlabel("Risk Category (0 = Low, 1 = High)")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(
    X.corr(numeric_only=True),
    cmap="coolwarm",
    annot=False
)
plt.title("Correlation Heatmap of Input Features")
plt.tight_layout()
plt.show()


# 5. Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAINING RECORDS:", len(X_train))
print("TESTING RECORDS:", len(X_test))


# 6. Define Machine Learning Algorithms

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "SVM": SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    )
}


# 7. Train Models and Evaluate Metrics

results = []
trained_models = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    results.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    trained_models[name] = pipeline

    print("\n" + "=" * 55)
    print(name.upper())
    print("=" * 55)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-Score :", round(f1, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Low Risk", "High Risk"],
            zero_division=0
        )
    )


# 8. Compare All Algorithms

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1-Score",
    ascending=False
).reset_index(drop=True)

print("\n\nFINAL MODEL COMPARISON")
print("=" * 75)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# 9. Identify the Best Algorithm

best_name = results_df.loc[0, "Algorithm"]
best_model = trained_models[best_name]

best_accuracy = results_df.loc[0, "Accuracy"]
best_precision = results_df.loc[0, "Precision"]
best_recall = results_df.loc[0, "Recall"]
best_f1 = results_df.loc[0, "F1-Score"]

print("\n\nBEST-PERFORMING ALGORITHM")
print("=" * 55)
print("Best Algorithm :", best_name)
print("Accuracy       :", round(best_accuracy, 4))
print("Precision      :", round(best_precision, 4))
print("Recall         :", round(best_recall, 4))
print("F1-Score       :", round(best_f1, 4))
print("\nSelection Basis: Highest F1-Score")


# 10. Plot Evaluation Metrics

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

results_df.set_index("Algorithm")[metrics].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Performance Comparison of Disease Prediction Algorithms")
plt.ylabel("Score")
plt.xlabel("Machine Learning Algorithm")
plt.ylim(0, 1)
plt.xticks(rotation=30, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# 11. Confusion Matrix for the Best Model

best_predictions = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=[0, 1]
)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Low Risk", "High Risk"],
    yticklabels=["Low Risk", "High Risk"]
)

plt.title("Confusion Matrix - " + best_name)
plt.xlabel("Predicted Risk")
plt.ylabel("Actual Risk")
plt.tight_layout()
plt.show()


# 12. ROC Curve Comparison

plt.figure(figsize=(8, 6))

for name, pipeline in trained_models.items():

    model = pipeline.named_steps["model"]

    if hasattr(model, "predict_proba"):
        scores = pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = pipeline.decision_function(X_test)
    else:
        continue

    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {roc_auc:.3f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()


# 13. Cross-Validation of the Best Algorithm

best_cv_scores = cross_val_score(
    best_model,
    X,
    y,
    cv=5,
    scoring="f1"
)

print("\nCROSS-VALIDATION RESULTS")
print("=" * 55)
print("F1-Scores:", np.round(best_cv_scores, 4))
print("Mean F1-Score:", round(best_cv_scores.mean(), 4))


# 14. Feature Importance, If Supported

best_algorithm = best_model.named_steps["model"]

if hasattr(best_algorithm, "feature_importances_"):

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_algorithm.feature_importances_
    }).sort_values(
        by="Importance",
        ascending=False
    )

    print("\nFEATURE IMPORTANCE")
    print(importance.to_string(index=False))

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance,
        x="Importance",
        y="Feature"
    )

    plt.title("Feature Importance - " + best_name)
    plt.tight_layout()
    plt.show()


# 15. Save the Best Model

joblib.dump(best_model, "best_disease_risk_model.pkl")

print("\nBest model saved as best_disease_risk_model.pkl")


# 16. Final Conclusion

print("\n" + "=" * 65)
print("FINAL CONCLUSION")
print("=" * 65)

print(
    f"Among the five algorithms, {best_name} performed best "
    f"based on the highest F1-Score of {best_f1:.4f}."
)

print(f"Accuracy  : {best_accuracy:.4f}")
print(f"Precision : {best_precision:.4f}")
print(f"Recall    : {best_recall:.4f}")
print(f"F1-Score  : {best_f1:.4f}")

print(
    "\nTherefore, " + best_name +
    " is selected as the best-performing algorithm "
    "for this dataset based on the test-set F1-Score."
)

print(
    "\nNote: The risk labels are created using the median "
    "of the average disease rates. They are project-defined "
    "categories, not medically validated diagnoses."
)