import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()

def generate_training_data(n_samples=50000):
    """Generate 50,000 realistic bank transactions for training"""
    
    print(f"Generating {n_samples} transactions...")
    
    data = []
    
    for i in range(n_samples):
        is_fraud = random.random() < 0.05  # 5% fraud rate
        
        # Fraud transactions look different from legit ones
        if is_fraud:
            amount = round(random.uniform(2000, 50000), 2)
            hour = random.choice([0, 1, 2, 3, 4])  # Late night
            transaction_count_1h = random.randint(5, 20)  # Many transactions
            distance_from_home = random.uniform(500, 5000)  # Far from home
        else:
            amount = round(random.uniform(1, 3000), 2)
            hour = random.randint(8, 22)  # Normal hours
            transaction_count_1h = random.randint(1, 3)  # Few transactions
            distance_from_home = random.uniform(0, 100)  # Near home

        transaction = {
            # Features our model will learn from
            "amount": amount,
            "hour_of_day": hour,
            "day_of_week": random.randint(0, 6),
            "transaction_count_1h": transaction_count_1h,
            "transaction_count_24h": random.randint(1, 30),
            "distance_from_home": distance_from_home,
            "distance_from_last_transaction": random.uniform(0, 1000),
            "is_foreign_transaction": random.choice([0, 1]),
            "is_online_transaction": random.choice([0, 1]),
            "merchant_category_encoded": random.randint(0, 9),
            "card_type_encoded": random.randint(0, 1),
            "device_type_encoded": random.randint(0, 3),
            
            # Target variable
            "is_fraud": int(is_fraud)
        }
        
        data.append(transaction)
        
        if (i + 1) % 10000 == 0:
            print(f"Generated {i + 1} transactions...")
    
    df = pd.DataFrame(data)
    df.to_csv("data/training_data.csv", index=False)
    print(f"✅ Training data saved to data/training_data.csv")
    print(f"Total transactions: {len(df)}")
    print(f"Fraud transactions: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
    print(f"Legit transactions: {(df['is_fraud']==0).sum()}")
    
    return df

if __name__ == "__main__":
    generate_training_data()