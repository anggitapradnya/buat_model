import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.sklearn

# === Load dataset ===
df = pd.read_csv("C:/Users/Anggita Pradnya Dewi/Eksperimen/preprocessing/telco_churn_preprocessing/telco_churn_clean.csv")

# Misal target column 'Churn' dan sisanya features
X = df.drop(columns=['Churn'])
y = df['Churn']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Enable MLflow autologging ===
mlflow.sklearn.autolog()

# === Define RandomForest dengan hyperparameter manual + balanced class weight ===
model = RandomForestClassifier(
    n_estimators=200,         # jumlah pohon
    max_depth=10,             # batasi kedalaman pohon
    min_samples_split=10,     # minimal sample untuk split
    min_samples_leaf=5,       # minimal sample di leaf
    max_features='sqrt',      # jumlah fitur yang dipakai tiap split
    class_weight='balanced',  # bantu kelas minoritas
    random_state=42,
    n_jobs=-1
)

# === Train model dengan MLflow run ===
with mlflow.start_run(run_name="RandomForest_Manual_Balanced"):
    model.fit(X_train, y_train)

    # === Evaluate model ===
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]  # untuk ROC AUC

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
