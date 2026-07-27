import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import mlflow
import mlflow.xgboost
import pickle
import json

def train_fraud_model():
    print("🚀 Starting JPMC Fraud Detection Model Training...")
    print("=" * 50)

    # Load training data
    print("📂 Loading training data...")
    df = pd.read_csv("data/training_data.csv")
    print(f"✅ Loaded {len(df)} transactions")

    # Split features and target
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Train size: {len(X_train)} | Test size: {len(X_test)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost model (same config real banks use)
    print("\n🤖 Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=19,  # handles class imbalance (95% legit, 5% fraud)
        eval_metric="auc",
        random_state=42
    )

    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=50
    )

    # Evaluate model
    print("\n📊 Model Performance:")
    print("=" * 50)
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred,
          target_names=["Legit", "Fraud"]))

    # Save model and scaler
    print("💾 Saving model...")
    with open("models/fraud_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save feature names
    feature_names = list(X.columns)
    with open("models/feature_names.json", "w") as f:
        json.dump(feature_names, f)

    print("✅ Model saved to models/fraud_model.pkl")
    print("✅ Scaler saved to models/scaler.pkl")
    print("✅ Features saved to models/feature_names.json")
    print("\n🎉 Training Complete!")

if __name__ == "__main__":
    train_fraud_model()