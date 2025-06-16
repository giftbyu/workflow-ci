import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

def main(args):
    mlflow.set_tracking_uri("https://dagshub.com/giftbyu/mlops.mlflow")
    mlflow.set_experiment("WaterPotability_Tuning_CI")
    
    run = mlflow.active_run()
    if run is None:
        run = mlflow.start_run()
    print(f"MLflow Run ID: {run.info.run_id}")
    mlflow.log_param("script_name", "modelling_tuning_smote.py")

    try:
        df = pd.read_csv("MLProject_folder/processed_data.csv")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    X = df.drop("Potability", axis=1)
    y = df["Potability"]
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.random_state, stratify=y)

    smote = SMOTE(random_state=args.random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print("Data shape after SMOTE:", X_train_smote.shape)
    mlflow.log_param("smote_applied", True)

    print("Training RandomForestClassifier with SMOTE...")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=args.random_state,
        n_jobs=-1
    )
    model.fit(X_train_smote, y_train_smote)
    
    mlflow.log_params({
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "random_state": args.random_state
    })

    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    mlflow.log_metrics({
        "test_accuracy": accuracy,
        "test_f1_score": f1,
        "test_precision": precision,
        "test_recall": recall,
        "test_roc_auc": roc_auc
    })

    mlflow.sklearn.log_model(model, "model")
    print("Model logged successfully.")

    mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    main(args)