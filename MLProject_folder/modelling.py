import pandas as pd
import mlflow
import os
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    try:
        mlflow.sklearn.autolog()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, 'processed_data.csv')
        df = pd.read_csv(data_path)
        X = df.drop('Potability', axis=1)
        y = df['Potability']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        with mlflow.start_run(run_name="Baseline_RF_Model_Lokal") as run:
            print("\n=== MEMULAI MODEL BASELINE (LOKAL) ===")
            print(f"MLflow Run ID: {run.info.run_id}")

            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Accuracy: {accuracy:.4f}")
            print("\n✅ Model baseline selesai. Cek hasilnya dengan `python -m mlflow ui`.")
    
    except Exception as e:
        print(f"\n❌ Terjadi error: {e}")

if __name__ == '__main__':
    main()