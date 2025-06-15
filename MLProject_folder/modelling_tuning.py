# Workflow-CI/MLProject_folder/modelling.py
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Inisialisasi MLflow Autologging
mlflow.sklearn.autolog()

# Muat data
df = pd.read_csv("../data/processed_data.csv")
X = df.drop('Potability', axis=1)
y = df['Potability']
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Mulai Run
with mlflow.start_run(run_name="Automated_CI_Training"):
    # Gunakan hyperparameter terbaik dari Kriteria 2
    best_params = {
        'max_depth': 20,
        'min_samples_leaf': 2,
        'min_samples_split': 2,
        'n_estimators': 300
    }

    # Latih model dengan parameter terbaik
    model = RandomForestClassifier(random_state=42, **best_params)
    model.fit(X_train, y_train)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    print("Model berhasil dilatih ulang dan di-log oleh CI.")