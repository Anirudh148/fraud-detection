import json
import time
import random
from datetime import datetime
from faker import Faker
from kafka import KafkaProducer

fake = Faker()

# Realistic merchant categories like real banks use
MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "gas_station", "online_shopping",
    "atm_withdrawal", "hotel", "airline", "pharmacy",
    "electronics", "luxury_retail"
]

# Suspicious patterns that fraud looks like
FRAUD_PATTERNS = [
    "multiple_small_transactions",
    "unusual_location",
    "high_value_online",
    "atm_rapid_withdrawal"
]

def generate_transaction():
    """Generate one realistic bank transaction"""
    
    is_fraud = random.random() < 0.05  # 5% fraud rate like real banks
    
    transaction = {
        "transaction_id": fake.uuid4(),
        "timestamp": datetime.now().isoformat(),
        "customer_id": f"CUST_{random.randint(1000, 9999)}",
        "account_id": f"ACC_{random.randint(10000, 99999)}",
        "amount": round(random.uniform(500, 50000), 2) if is_fraud 
                  else round(random.uniform(1, 5000), 2),
        "merchant_category": random.choice(MERCHANT_CATEGORIES),
        "merchant_name": fake.company(),
        "location_city": fake.city(),
        "location_country": fake.country_code(),
        "card_type": random.choice(["credit", "debit"]),
        "transaction_type": random.choice(["purchase", "withdrawal", "transfer"]),
        "device_type": random.choice(["mobile", "web", "atm", "pos"]),
        "ip_address": fake.ipv4(),
        "is_fraud": is_fraud,  # In real JPMC this is what model predicts
        "fraud_pattern": random.choice(FRAUD_PATTERNS) if is_fraud else None
    }
    
    return transaction

def start_producer():
    """Start streaming transactions to Kafka"""
    
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )
    
    print("🚀 JPMC Transaction Stream Started...")
    print("=" * 50)
    
    transaction_count = 0
    
    while True:
        transaction = generate_transaction()
        
        # Send to Kafka topic
        producer.send('jpmc_transactions', value=transaction)
        
        transaction_count += 1
        
        # Print to console so we can see it working
        fraud_flag = "🚨 FRAUD" if transaction['is_fraud'] else "✅ LEGIT"
        print(f"[{transaction_count}] {fraud_flag} | "
              f"${transaction['amount']} | "
              f"{transaction['merchant_category']} | "
              f"{transaction['location_city']}")
        
        # Send 1 transaction per second like real stream
        time.sleep(1)

if __name__ == "__main__":
    start_producer()