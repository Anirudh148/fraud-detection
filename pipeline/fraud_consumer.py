import json
import pickle
import psycopg2
import numpy as np
import pandas as pd
from kafka import KafkaConsumer
from datetime import datetime

# Load trained model and scaler
print("📂 Loading fraud detection model...")
with open("models/fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/feature_names.json", "r") as f:
    import json as j
    feature_names = j.load(f)

print("✅ Model loaded successfully!")

# Connect to PostgreSQL
def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="fraud_db",
        user="user",
        password="pass"
    )

# Create table if not exists
def setup_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fraud_alerts (
            id SERIAL PRIMARY KEY,
            transaction_id VARCHAR(100),
            customer_id VARCHAR(50),
            amount FLOAT,
            merchant_category VARCHAR(50),
            location_city VARCHAR(100),
            fraud_score FLOAT,
            is_fraud_predicted BOOLEAN,
            actual_is_fraud BOOLEAN,
            timestamp TIMESTAMP,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Database table ready!")

def extract_features(transaction):
    """Extract features from transaction for model"""
    features = {
        "amount": transaction["amount"],
        "hour_of_day": int(transaction["timestamp"][11:13]),
        "day_of_week": datetime.fromisoformat(
                       transaction["timestamp"]).weekday(),
        "transaction_count_1h": np.random.randint(1, 10),
        "transaction_count_24h": np.random.randint(1, 30),
        "distance_from_home": np.random.uniform(0, 1000),
        "distance_from_last_transaction": np.random.uniform(0, 500),
        "is_foreign_transaction": 1 if transaction[
                                  "location_country"] != "US" else 0,
        "is_online_transaction": 1 if transaction[
                                 "device_type"] == "web" else 0,
        "merchant_category_encoded": hash(
                                     transaction["merchant_category"]) % 10,
        "card_type_encoded": 1 if transaction["card_type"] == "credit" else 0,
        "device_type_encoded": ["mobile","web","atm","pos"].index(
                                transaction["device_type"]) 
                                if transaction["device_type"] in 
                                ["mobile","web","atm","pos"] else 0,
    }
    return pd.DataFrame([features])[feature_names]

def save_alert(transaction, fraud_score, is_fraud_predicted):
    """Save fraud alert to PostgreSQL"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO fraud_alerts 
        (transaction_id, customer_id, amount, merchant_category,
         location_city, fraud_score, is_fraud_predicted, 
         actual_is_fraud, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        transaction["transaction_id"],
        transaction["customer_id"],
        transaction["amount"],
        transaction["merchant_category"],
        transaction["location_city"],
        float(fraud_score),
        bool(is_fraud_predicted),
        bool(transaction["is_fraud"]),
        transaction["timestamp"]
    ))
    conn.commit()
    cursor.close()
    conn.close()

def start_consumer():
    """Start consuming transactions from Kafka"""
    setup_database()

    consumer = KafkaConsumer(
        'jpmc_transactions',
        bootstrap_servers=['localhost:9092'],
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        auto_offset_reset='latest',
        group_id='fraud_detection_group'
    )

    print("\n🚀 JPMC Fraud Detection Consumer Started!")
    print("=" * 60)
    print("Listening for transactions...\n")

    for message in consumer:
        transaction = message.value

        # Extract features and score
        features = extract_features(transaction)
        features_scaled = scaler.transform(features)
        fraud_score = model.predict_proba(features_scaled)[0][1]
        is_fraud_predicted = fraud_score > 0.5

        # Save to database
        save_alert(transaction, fraud_score, is_fraud_predicted)

        # Print result
        if is_fraud_predicted:
            print(f"🚨 FRAUD DETECTED!")
            print(f"   Customer: {transaction['customer_id']}")
            print(f"   Amount: ${transaction['amount']}")
            print(f"   Location: {transaction['location_city']}")
            print(f"   Fraud Score: {fraud_score:.2%}")
            print(f"   Merchant: {transaction['merchant_category']}")
            print("-" * 60)
        else:
            print(f"✅ LEGIT | ${transaction['amount']:.2f} | "
                  f"{transaction['merchant_category']} | "
                  f"Score: {fraud_score:.2%}")

if __name__ == "__main__":
    start_consumer()