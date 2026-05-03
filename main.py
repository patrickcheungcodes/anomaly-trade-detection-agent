import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from openai import OpenAI

# --- Page Config ---
st.set_page_config(page_title="Agentic Trade Anomaly Detector", layout="wide")

# --- Initialize Environment & DeepSeek ---
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    st.error("DEEPSEEK_API_KEY not found in .env file. Please add it.")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# --- Core Logic ---
class AnomalyDetector:
    def __init__(self, contamination: float = 0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        features = ['amount', 'position_value']
        X = df[features].fillna(0)
        df['anomaly_score'] = self.model.fit_predict(X)
        df['is_anomaly'] = df['anomaly_score'] == -1
        return df

class RiskAgent:
    def generate_report(self, trade_data: pd.Series) -> str:
        prompt = f"""
        You are a Senior Quantitative Risk Officer at a Hedge Fund in Hong Kong.
        Write a short (max 3 sentences), professional risk narrative for this flagged anomalous trade.

        Trade Details:
        - Asset: {trade_data.get('asset', 'Unknown')}
        - Amount Traded: ${trade_data.get('amount', 0):,.2f}
        - Position Value: ${trade_data.get('position_value', 0):,.2f}
        """

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating report: {str(e)}"

# --- Streamlit UI ---
def main():
    st.title("📈 Hedge Fund Agentic Anomaly Detection")
    st.markdown("Isolation Forest + DeepSeek AI for explainable risk insights.")

    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload Trade Data (CSV)", type="csv")
        contamination = st.slider("Anomaly Sensitivity", 0.01, 0.10, 0.05)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Basic cleaning
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df['position_value'] = pd.to_numeric(df['position_value'], errors='coerce')
        df = df.dropna(subset=['amount', 'position_value'])

        # Run Detection
        with st.spinner("Running Anomaly Detection..."):
            detector = AnomalyDetector(contamination=contamination)
            df = detector.detect(df)

        anomalies = df[df['is_anomaly']]

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Trades", len(df))
        col2.metric("Anomalies Detected", len(anomalies))
        col3.metric("Sensitivity", f"{contamination*100:.1f}%")

        # Scatter Plot
        st.subheader("Amount vs Position Value")
        st.scatter_chart(df, x='amount', y='position_value', color='is_anomaly')

        # Anomalies Table + AI Reports
        st.subheader("🚨 Detected Anomalies & AI Risk Reports")
        if not anomalies.empty:
            st.dataframe(anomalies[['trade_id', 'asset', 'amount', 'position_value']], use_container_width=True)

            selected_trade_id = st.selectbox("Select Trade ID to investigate:", anomalies['trade_id'])
            if st.button("Generate Risk Report"):
                trade_row = anomalies[anomalies['trade_id'] == selected_trade_id].iloc[0]
                agent = RiskAgent()
                with st.spinner("DeepSeek Agent is analyzing..."):
                    report = agent.generate_report(trade_row)
                    st.info(report)
        else:
            st.success("No anomalies detected at current sensitivity.")

    else:
        st.info("👈 Upload your trade data CSV to begin analysis.")

if __name__ == "__main__":
    main()
