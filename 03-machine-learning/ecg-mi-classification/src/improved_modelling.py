import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# Locate the data file relative to this script
data_path = Path(__file__).resolve().parent.parent / "data" / "combined_patient_data.csv"
df = pd.read_csv(data_path)

features = [
    "RR Interval (s)",
    "HR (bpm)",
    "QRS Duration (s)",
    "QT Interval (s)"
]

X = df[features]
y = df["Label"]
groups = df["Patient_ID"]

# Split by patient rather than by individual observation
splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.25, random_state=42
)

train_idx, test_idx = next(splitter.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

assert set(groups.iloc[train_idx]).isdisjoint(
    set(groups.iloc[test_idx])
), "Patient overlap detected!"

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"\n{name}")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.3f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.3f}")
    print(f"F1 Score:  {f1_score(y_test, predictions, zero_division=0):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, probabilities):.3f}")
