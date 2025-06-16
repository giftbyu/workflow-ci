import pandas as pd
import numpy as np
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import json

DAGSHUB_REPO_OWNER = 'giftbyu'
DAGSHUB_REPO_NAME = 'mlops'
DAGSHUB_TOKEN = "7598a5b91255c19dead8882bd5b515e96cf02d73"  # Ganti dengan token DagsHub Anda

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Potable', 'Potable'], 
                yticklabels=['Not Potable', 'Potable'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    path = "confusion_matrix.png"
    plt.savefig(path)
    plt.close()
    return path

def plot_roc_curve(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    path = "roc_curve.png"
    plt.savefig(path)
    plt.close()
    return path, roc_auc

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    path = "feature_importance.png"
    plt.savefig(path)
    plt.close()
    return path, importances

def main():
    try:
        # Setup DagsHub dan MLflow
        dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
        mlflow.set_tracking_uri(f'https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow')
        
        # Set autentikasi
        os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_REPO_OWNER
        os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
        
        print("Inisialisasi DagsHub dan MLflow berhasil.")

        # Load data
        df = pd.read_csv('processed_data.csv')
        X = df.drop('Potability', axis=1)
        y = df['Potability']
        feature_names = X.columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        # Inisialisasi model dan grid search
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            estimator=rf, 
            param_grid=param_grid, 
            cv=5, 
            n_jobs=-1, 
            verbose=2, 
            scoring='f1'
        )
        
        # Mulai MLflow run
        with mlflow.start_run(run_name="Tuning_RandomForest") as run:
            run_id = run.info.run_id
            print(f"\n=== MLflow Run ID: {run_id} ===")
            
            # Latih model
            print("Memulai pelatihan dengan GridSearchCV...")
            grid_search.fit(X_train, y_train)
            print("Pelatihan selesai.")
            
            # Dapatkan model terbaik
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            # Log parameter
            print("\nLogging parameter terbaik:", best_params)
            mlflow.log_params(best_params)
            mlflow.log_param("best_estimator", "RandomForestClassifier")
            
            # Log metrik cross-validation
            best_cv_score = grid_search.best_score_
            mlflow.log_metric("best_cross_val_f1", best_cv_score)
            print(f"F1 Score Cross-Validation: {best_cv_score:.4f}")
            
            # Evaluasi pada test set
            y_pred = best_model.predict(X_test)
            y_probs = best_model.predict_proba(X_test)[:, 1]  # Probabilitas kelas positif
            
            # Hitung metrik
            test_accuracy = accuracy_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred)
            test_recall = recall_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred)
            test_roc_auc = roc_auc_score(y_test, y_probs)
            
            # Log metrik test
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("test_precision", test_precision)
            mlflow.log_metric("test_recall", test_recall)
            mlflow.log_metric("test_f1_score", test_f1)
            mlflow.log_metric("test_roc_auc", test_roc_auc)
            
            print("\n=== Hasil Evaluasi Test Set ===")
            print(f"Akurasi: {test_accuracy:.4f}")
            print(f"Presisi: {test_precision:.4f}")
            print(f"Recall: {test_recall:.4f}")
            print(f"F1 Score: {test_f1:.4f}")
            print(f"ROC AUC: {test_roc_auc:.4f}")
            
            # Buat dan log visualisasi
            print("\nMembuat dan logging artefak visual...")
            cm_path = plot_confusion_matrix(y_test, y_pred)
            mlflow.log_artifact(cm_path, "plots")
            
            roc_path, roc_auc = plot_roc_curve(y_test, y_probs)
            mlflow.log_artifact(roc_path, "plots")
            
            fi_path, feature_importances = plot_feature_importance(best_model, feature_names)
            mlflow.log_artifact(fi_path, "plots")
            
            # Log feature importance sebagai metrik
            for i, (name, imp) in enumerate(zip(feature_names, feature_importances)):
                mlflow.log_metric(f"feature_importance_{name}", imp)
            
            # Simpan classification report
            from sklearn.metrics import classification_report
            report = classification_report(y_test, y_pred)
            report_path = "classification_report.txt"
            with open(report_path, "w") as f:
                f.write(report)
            mlflow.log_artifact(report_path, "reports")
            
            # Simpan model dengan pendekatan alternatif
            print("\nLogging model dengan pendekatan alternatif...")
            model_path = "model"
            os.makedirs(model_path, exist_ok=True)
            
            # 1. Simpan model
            model_file = os.path.join(model_path, "model.joblib")
            joblib.dump(best_model, model_file)
            
            # 2. Simpan metadata
            metadata = {
                "model_type": "RandomForestClassifier",
                "input_example": X_train.iloc[:1].to_dict(orient='records')[0],
                "feature_names": feature_names,
                "classes": [0, 1],
                "run_id": run_id
            }
            metadata_file = os.path.join(model_path, "metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=4)
            
            # 3. Log seluruh folder sebagai artefak
            mlflow.log_artifacts(model_path, "model")
            
            # Log environment info
            mlflow.log_dict({
                "python_version": "3.11",
                "dependencies": {
                    "pandas": pd.__version__,
                    "numpy": np.__version__,
                    "scikit-learn": joblib.__version__,
                    "mlflow": mlflow.__version__
                }
            }, "environment.json")
            
            print("\n✅ Proses tuning selesai. Model dan artefak berhasil di-log.")
            print(f"🧪 Lihat hasil di: https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow#/experiments/0/runs/{run_id}")
    
    except Exception as e:
        print(f"\n❌ Terjadi error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()