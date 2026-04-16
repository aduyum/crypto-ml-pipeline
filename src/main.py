import os
import pandas as pd
import logging
from data_fetcher import fetch_binance_data
from features import add_features
from labels import add_labels
from walk_forward import run_walk_forward

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

def main():
    logging.info(">>> INITIALIZING QUANTITATIVE ML PIPELINE v1.0 <<<")
    data_path = 'data/btc_1h_raw.parquet'
    
    if not os.path.exists(data_path):
        logging.info("No cached data found. Fetching from Binance...")
        df_raw = fetch_binance_data()
    else:
        logging.info(f"Loading cached dataset: {data_path}")
        df_raw = pd.read_parquet(data_path)
        
    logging.info("Extracting Time-Series Features & Signals...")
    df_features = add_features(df_raw)
    df_labeled = add_labels(df_features)
    
    logging.info("Commencing Walk-Forward Backtest...")
    run_walk_forward(df_labeled)
    
    logging.info(">>> PIPELINE EXECUTION FINISHED <<<")

if __name__ == '__main__':
    main()