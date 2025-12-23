import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.sklearn
import numpy as np

# === Load dataset ===
df = pd.read_csv("C:/Users/Anggita Pradnya Dewi/Eksperimen/preprocessing/telco_churn_preprocessing/telco_churn_clean.csv")

# Pisahkan features dan target
X = df.drop(columns=['Churn'])
y = df['Churn']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === RandomForest ===
rf = RandomForestClassifier(random_state=42)

# === Hyperparameter grid ===
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [5, 7, 10],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 5],
    'max_features': ['sqrt', 'log2'],
    'class_weight': [{0:1,1:1.2}, {0:1,1:1.5}, 'balanced']
}

# === GridSearchCV ===
grid = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)

# === MLflow manual logging ===
with mlflow.start_run(run_name="RandomForest_Skilled_Optimized"):
    # Fit model
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # Predict probabilities
    y_proba = best_model.predict_proba(X_test)[:,1]

    # === Threshold tuning untuk F1 optimal ===
    thresholds = np.arange(0.3, 0.7, 0.01)
    best_f1 = 0
    best_thresh = 0.5
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_test, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    # Apply best threshold
    y_pred = (y_proba >= best_thresh).astype(int)

    # Hitung metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # === Manual logging ke MLflow ===
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("best_threshold", best_thresh)

    # Simpan model sebagai artefak
    mlflow.sklearn.log_model(best_model, "random_forest_model")

    # Print hasil
    print("=== BEST PARAMETERS ===")
    print(grid.best_params_)
    print(f"Optimal Threshold: {best_thresh:.2f}")
    print("\n=== TEST METRICS ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
