import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
import psycopg2
import pandas as pd

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "JPMC Fraud Detection Dashboard"

# DB connection
def get_db():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="fraud_db",
        user="user",
        password="pass"
    )

def get_data():
    try:
        conn = get_db()
        df = pd.read_sql("""
            SELECT customer_id, amount, merchant_category,
                   location_city, fraud_score, 
                   is_fraud_predicted, timestamp
            FROM fraud_alerts
            ORDER BY timestamp DESC
            LIMIT 500
        """, conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

# Layout
app.layout = html.Div(style={
    "backgroundColor": "#0a0a0a",
    "minHeight": "100vh",
    "fontFamily": "Arial",
    "padding": "20px"
}, children=[

    # Header
    html.Div([
        html.H1("JPMC Fraud Detection Dashboard",
                style={"color": "#00ff88", "textAlign": "center",
                       "margin": "0", "fontSize": "28px"}),
        html.P("Real-Time Transaction Monitoring System",
               style={"color": "#888", "textAlign": "center",
                      "margin": "5px 0 20px 0"})
    ]),

    # Stats Cards
    html.Div(id="stats-cards", style={
        "display": "flex",
        "justifyContent": "space-around",
        "marginBottom": "20px"
    }),

    # Charts Row 1
    html.Div([
        html.Div([
            dcc.Graph(id="fraud-gauge")
        ], style={"width": "30%"}),

        html.Div([
            dcc.Graph(id="amount-chart")
        ], style={"width": "35%"}),

        html.Div([
            dcc.Graph(id="merchant-chart")
        ], style={"width": "35%"}),
    ], style={"display": "flex", "gap": "10px", "marginBottom": "20px"}),

    # Live Transaction Table
    html.Div([
        html.H3("Live Transaction Feed",
                style={"color": "#00ff88", "margin": "0 0 10px 0"}),
        html.Div(id="transaction-table")
    ], style={
        "backgroundColor": "#111",
        "padding": "20px",
        "borderRadius": "10px",
        "border": "1px solid #222"
    }),

    # Auto refresh every 3 seconds
    dcc.Interval(id="interval", interval=3000, n_intervals=0)
])

# ─────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────

@app.callback(
    [Output("stats-cards", "children"),
     Output("fraud-gauge", "figure"),
     Output("amount-chart", "figure"),
     Output("merchant-chart", "figure"),
     Output("transaction-table", "children")],
    [Input("interval", "n_intervals")]
)
def update_dashboard(n):
    df = get_data()

    if df.empty:
        empty = html.P("No data yet...",
                       style={"color": "#888"})
        empty_fig = go.Figure()
        return [empty], empty_fig, empty_fig, empty_fig, empty

    total = len(df)
    fraud_count = df["is_fraud_predicted"].sum()
    fraud_rate = (fraud_count / total * 100) if total > 0 else 0
    avg_amount = df["amount"].mean()

    # Stats Cards
    cards = [
        stat_card("Total Transactions", f"{total:,}", "#00ff88"),
        stat_card("Fraud Detected", f"{int(fraud_count):,}", "#ff4444"),
        stat_card("Fraud Rate", f"{fraud_rate:.2f}%", "#ffaa00"),
        stat_card("Avg Amount", f"${avg_amount:,.2f}", "#4488ff"),
    ]

    # Fraud Gauge
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fraud_rate,
        title={"text": "Fraud Rate %",
               "font": {"color": "white"}},
        gauge={
            "axis": {"range": [0, 20],
                     "tickcolor": "white"},
            "bar": {"color": "#ff4444"},
            "steps": [
                {"range": [0, 5], "color": "#1a3a1a"},
                {"range": [5, 10], "color": "#3a3a1a"},
                {"range": [10, 20], "color": "#3a1a1a"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 10
            }
        },
        number={"font": {"color": "white"},
                "suffix": "%"}
    ))
    gauge.update_layout(
        paper_bgcolor="#111",
        font={"color": "white"},
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # Amount Distribution
    amount_fig = px.histogram(
        df, x="amount", nbins=30,
        title="Transaction Amount Distribution",
        color_discrete_sequence=["#00ff88"]
    )
    amount_fig.update_layout(
        paper_bgcolor="#111",
        plot_bgcolor="#111",
        font={"color": "white"},
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis={"gridcolor": "#222"},
        yaxis={"gridcolor": "#222"}
    )

    # Merchant Category
    merchant_counts = df["merchant_category"].value_counts().head(6)
    merchant_fig = px.bar(
        x=merchant_counts.values,
        y=merchant_counts.index,
        orientation="h",
        title="Top Merchant Categories",
        color_discrete_sequence=["#4488ff"]
    )
    merchant_fig.update_layout(
        paper_bgcolor="#111",
        plot_bgcolor="#111",
        font={"color": "white"},
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis={"gridcolor": "#222"},
        yaxis={"gridcolor": "#222"}
    )

    # Transaction Table
    recent = df.head(10)
    rows = []
    for _, row in recent.iterrows():
        is_fraud = row["is_fraud_predicted"]
        color = "#ff4444" if is_fraud else "#00ff88"
        status = "🚨 FRAUD" if is_fraud else "✅ LEGIT"
        rows.append(html.Tr([
            html.Td(row["customer_id"],
                    style={"color": "white", "padding": "8px"}),
            html.Td(f"${row['amount']:,.2f}",
                    style={"color": "white", "padding": "8px"}),
            html.Td(row["merchant_category"],
                    style={"color": "#888", "padding": "8px"}),
            html.Td(row["location_city"],
                    style={"color": "#888", "padding": "8px"}),
            html.Td(f"{row['fraud_score']:.4f}",
                    style={"color": "#ffaa00", "padding": "8px"}),
            html.Td(status, style={"color": color, "padding": "8px"}),
        ]))

    table = html.Table([
        html.Thead(html.Tr([
            html.Th("Customer", style={"color": "#00ff88",
                    "padding": "8px", "textAlign": "left"}),
            html.Th("Amount", style={"color": "#00ff88",
                    "padding": "8px", "textAlign": "left"}),
            html.Th("Merchant", style={"color": "#00ff88",
                    "padding": "8px", "textAlign": "left"}),
            html.Th("City", style={"color": "#00ff88",
                    "padding": "8px", "textAlign": "left"}),
            html.Th("Fraud Score", style={"color": "#00ff88",
                    "padding": "8px", "textAlign": "left"}),
            html.Th("Status", style={"color": "#00ff88",
                    "padding": "8px", "textAlign": "left"}),
        ])),
        html.Tbody(rows)
    ], style={"width": "100%", "borderCollapse": "collapse"})

    return cards, gauge, amount_fig, merchant_fig, table

def stat_card(title, value, color):
    return html.Div([
        html.P(title, style={"color": "#888", "margin": "0",
                             "fontSize": "12px"}),
        html.H2(value, style={"color": color, "margin": "5px 0 0 0"})
    ], style={
        "backgroundColor": "#111",
        "padding": "20px",
        "borderRadius": "10px",
        "border": f"1px solid {color}",
        "minWidth": "150px",
        "textAlign": "center"
    })

if __name__ == "__main__":
    app.run(debug=True, port=8050)