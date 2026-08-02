import os
import psycopg2
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# SQLAlchemy engine for pandas
def get_engine():
    return create_engine(
        "postgresql+psycopg2://user:pass@localhost:5432/fraud_db"
    )

# psycopg2 connection for direct queries
def get_db():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="fraud_db",
        user="user",
        password="pass"
    )

def explain_fraud(transaction: dict) -> str:
    """Use GPT-4 to explain why a transaction is fraudulent"""

    prompt = f"""
    You are a senior fraud analyst .
    Analyze this flagged transaction and explain in plain English 
    why it looks fraudulent. Be specific, professional, and concise.
    
    Transaction Details:
    - Customer ID: {transaction.get('customer_id')}
    - Amount: ${transaction.get('amount')}
    - Merchant Category: {transaction.get('merchant_category')}
    - Location: {transaction.get('location_city')}
    - Fraud Score: {transaction.get('fraud_score')}
    - Timestamp: {transaction.get('timestamp')}
    
    Provide:
    1. Why this transaction looks suspicious (3 specific reasons)
    2. Risk level assessment (LOW/MEDIUM/HIGH/CRITICAL)
    3. Recommended action (APPROVE/REVIEW/BLOCK)
    4. One sentence summary for the analyst dashboard
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content":
             "You are a senior fraud analyst. "
             "Always respond in a structured, professional format."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
        temperature=0.3
    )

    return response.choices[0].message.content

def answer_analyst_question(question: str) -> str:
    """GPT-4 powered analyst chatbot"""

    engine = get_engine()
    df = pd.read_sql("""
        SELECT customer_id, amount, merchant_category,
               location_city, fraud_score, is_fraud_predicted,
               timestamp
        FROM fraud_alerts
        ORDER BY timestamp DESC
        LIMIT 50
    """, engine)

    data_summary = df.to_string(index=False)

    prompt = f"""
    You are a fraud analytics assistant at JPMorgan Chase.
    You have access to the latest 50 transactions from our 
    fraud detection system.
    
    Transaction Data:
    {data_summary}
    
    Analyst Question: {question}
    
    Answer clearly and professionally based on the data above.
    Include specific numbers and transaction details where relevant.
    If no fraud is detected in the data, analyze suspicious patterns
    based on high amounts, unusual locations, or odd merchant categories.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content":
             "You are a fraud analytics assistant at JPMorgan Chase. "
             "Answer questions based on the provided transaction data. "
             "Even if no fraud is predicted, analyze suspicious patterns."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.3
    )

    return response.choices[0].message.content

def get_latest_fraud_and_explain():
    """Get latest fraud alert and explain it"""

    engine = get_engine()
    df = pd.read_sql("""
        SELECT transaction_id, customer_id, amount,
               merchant_category, location_city,
               fraud_score, timestamp
        FROM fraud_alerts
        ORDER BY fraud_score DESC
        LIMIT 1
    """, engine)

    if df.empty:
        print("No transactions found yet.")
        return "No transactions found yet."

    row = df.iloc[0]
    transaction = {
        "transaction_id": row["transaction_id"],
        "customer_id": row["customer_id"],
        "amount": row["amount"],
        "merchant_category": row["merchant_category"],
        "location_city": row["location_city"],
        "fraud_score": row["fraud_score"],
        "timestamp": row["timestamp"]
    }

    print(f"\n🚨 Highest Risk Transaction:")
    print(f"   Customer: {transaction['customer_id']}")
    print(f"   Amount: ${transaction['amount']}")
    print(f"   Location: {transaction['location_city']}")
    print(f"   Fraud Score: {transaction['fraud_score']}")
    print(f"\n🤖 GPT-4 Analysis:")
    print("=" * 60)

    explanation = explain_fraud(transaction)
    print(explanation)
    print("=" * 60)

    return explanation

if __name__ == "__main__":
    print("🚀 JPMC GPT-4 Fraud Explainer")
    print("=" * 60)

    # Test 1 - Explain highest risk transaction
    print("\n📋 TEST 1: Explain Highest Risk Transaction")
    get_latest_fraud_and_explain()

    # Test 2 - Analyst chatbot
    print("\n📋 TEST 2: Analyst Chatbot")
    questions = [
        "Which merchant category has the highest average transaction amount?",
        "What is the average transaction amount overall?",
        "Which city has the most transactions?"
    ]

    for q in questions:
        print(f"\n❓ Question: {q}")
        print(f"🤖 Answer: {answer_analyst_question(q)}")
        print("-" * 60)