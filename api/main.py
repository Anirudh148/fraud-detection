from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import json

app = FastAPI(
    title="JPMC Fraud Detection API",
    description="Real-time fraud detection and risk alert service",
    version="1.0.0"
)

# Load model
with open("models/fraud_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("models/feature_names.json", "r") as f:
    feature_names = json.load(f)

# DB connection
def get_db():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="fraud_db",
        user="user",
        password="pass"
    )

# Request model
class Transaction(BaseModel):
    customer_id: str
    amount: float
    merchant_category: str
    location_country: str
    device_type: str
    card_type: str
    timestamp: str = None

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "JPMC Fraud Detection API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/score")
def score_transaction(transaction: Transaction):
    """Score a single transaction for fraud in real time"""
    try:
        ts = transaction.timestamp or datetime.now().isoformat()
        hour = int(ts[11:13]) if len(ts) > 11 else datetime.now().hour

        features = {
            "amount": transaction.amount,
            "hour_of_day": hour,
            "day_of_week": datetime.now().weekday(),
            "transaction_count_1h": np.random.randint(1, 10),
            "transaction_count_24h": np.random.randint(1, 30),
            "distance_from_home": np.random.uniform(0, 1000),
            "distance_from_last_transaction": np.random.uniform(0, 500),
            "is_foreign_transaction": 1 if transaction.location_country != "US" else 0,
            "is_online_transaction": 1 if transaction.device_type == "web" else 0,
            "merchant_category_encoded": hash(transaction.merchant_category) % 10,
            "card_type_encoded": 1 if transaction.card_type == "credit" else 0,
            "device_type_encoded": ["mobile","web","atm","pos"].index(
                                    transaction.device_type)
                                    if transaction.device_type in
                                    ["mobile","web","atm","pos"] else 0,
        }

        df = pd.DataFrame([features])[feature_names]
        scaled = scaler.transform(df)
        fraud_score = float(model.predict_proba(scaled)[0][1])
        is_fraud = fraud_score > 0.5

        risk_level = (
            "HIGH" if fraud_score > 0.8 else
            "MEDIUM" if fraud_score > 0.5 else
            "LOW"
        )

        return {
            "customer_id": transaction.customer_id,
            "amount": transaction.amount,
            "fraud_score": round(fraud_score, 4),
            "is_fraud": is_fraud,
            "risk_level": risk_level,
            "recommendation": "BLOCK" if is_fraud else "APPROVE",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fraud-alerts")
def get_fraud_alerts(limit: int = 20):
    """Get latest fraud alerts from database"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT transaction_id, customer_id, amount,
                   merchant_category, location_city,
                   fraud_score, is_fraud_predicted, timestamp
            FROM fraud_alerts
            WHERE is_fraud_predicted = true
            ORDER BY created_at DESC
            LIMIT %s
        """, (limit,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        alerts = []
        for row in rows:
            alerts.append({
                "transaction_id": row[0],
                "customer_id": row[1],
                "amount": row[2],
                "merchant_category": row[3],
                "location_city": row[4],
                "fraud_score": row[5],
                "is_fraud_predicted": row[6],
                "timestamp": str(row[7])
            })
        return {"total": len(alerts), "alerts": alerts}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transactions")
def get_transactions(limit: int = 20):
    """Get latest transactions"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT transaction_id, customer_id, amount,
                   merchant_category, location_city,
                   fraud_score, is_fraud_predicted, timestamp
            FROM fraud_alerts
            ORDER BY created_at DESC
            LIMIT %s
        """, (limit,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        transactions = []
        for row in rows:
            transactions.append({
                "transaction_id": row[0],
                "customer_id": row[1],
                "amount": row[2],
                "merchant_category": row[3],
                "location_city": row[4],
                "fraud_score": row[5],
                "is_fraud_predicted": row[6],
                "timestamp": str(row[7])
            })
        return {"total": len(transactions), "transactions": transactions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def get_stats():
    """Get live fraud statistics"""
    try:
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM fraud_alerts")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM fraud_alerts WHERE is_fraud_predicted = true")
        total_fraud = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(amount) FROM fraud_alerts")
        avg_amount = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(amount) FROM fraud_alerts WHERE is_fraud_predicted = true")
        avg_fraud_amount = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        return {
            "total_transactions": total,
            "total_fraud_detected": total_fraud,
            "fraud_rate": f"{(total_fraud/total*100):.2f}%" if total > 0 else "0%",
            "avg_transaction_amount": round(float(avg_amount or 0), 2),
            "avg_fraud_amount": round(float(avg_fraud_amount or 0), 2),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))