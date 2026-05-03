import pandas as pd
import numpy as np
import logging
import os
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest

# 1. Production-ready logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TradeAnomalyDetector")

class AnomalyDetector:
    def __init__(self, data_path: str, output_path: str):
        self.data_path = data_path
        self.output_path = output_path
        
        # 2. Use dotenv for configurations
        load_dotenv()
        # Default to 5% contamination if not set in .env
        self.contamination = float(os.getenv('ANOMALY_CONTAMINATION', 0.05)) 
        
        # Initialize the ML Model
        self.model = IsolationForest(
            contamination=self.contamination, 
            random_state=42
        )

    def load_and_clean_data(self) -> pd.DataFrame:
        """Loads data with basic error handling."""
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Successfully loaded {len(df)} trades.")
            
            # Feature Engineering
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            
            # Drop rows where critical financial data couldn't be parsed
            df = df.dropna(subset=['amount', 'position_value'])
            
            return df
        except FileNotFoundError:
            logger.error(f"Data file not found at {self.data_path}. Please check the path.")
            raise
        except Exception as e:
            logger.error(f"An error occurred during data loading: {str(e)}")
            raise

    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies Isolation Forest for multivariate anomaly detection."""
        logger.info("Running Isolation Forest algorithm...")
        
        # Select features for the model
        features = ['amount', 'position_value']
        X = df[features]
        
        # Fit model and predict (-1 for anomalies, 1 for normal)
        df['anomaly_score'] = self.model.fit_predict(X)
        
        # Map back to boolean for easier querying (True if anomalous)
        df['is_anomaly'] = df['anomaly_score'] == -1
        
        anomalies_count = df['is_anomaly'].sum()
        logger.info(f"Detected {anomalies_count} potential anomalies.")
        
        return df

    def save_results(self, df: pd.DataFrame):
        """Saves the processed dataframe."""
        df.to_csv(self.output_path, index=False)
        logger.info(f"Results saved to {self.output_path}")

    def run_pipeline(self):
        """Executes the end-to-end detection pipeline."""
        logger.info("=== Starting Anomaly Detection Pipeline ===")
        df = self.load_and_clean_data()
        df_processed = self.detect_anomalies(df)
        self.save_results(df_processed)
        logger.info("=== Pipeline Completed ===")

if __name__ == "__main__":
    # Execution block
    detector = AnomalyDetector(
        data_path='data/sample_trades.csv', 
        output_path='data/processed_trades.csv'
    )
    detector.run_pipeline()
