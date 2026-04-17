import pandas as pd
import os
import joblib
import logging
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from data_fetcher import fetch_binance_data
from features import add_features
from labels import add_labels

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def train_production_model():
    logging.info("--- Starting Production Model Training ---")
    data_path = 'data/btc_1h_raw.parquet'
    
    if not os.path.exists(data_path):
        df_raw = fetch_binance_data()
    else:
        df_raw = pd.read_parquet(data_path)
        
    df_features = add_features(df_raw)
    df_labeled = add_labels(df_features)
    
    # Drop the lookahead rows
    df_labeled = df_labeled.iloc[:-12]
    
    drop_cols =['open', 'high', 'low', 'close', 'volume', 'Target']
    feature_cols =[col for col in df_labeled.columns if col not in drop_cols]
    
    X = df_labeled[feature_cols]
    y = df_labeled['Target'].values
    
    sample_weights = compute_sample_weight('balanced', y=y)
    
    # Best parameters discovered from walk-forward RandomizedSearchCV
    best_xgb = XGBClassifier(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.05, 
        random_state=42, 
        n_jobs=-1
    )
    
    logging.info("Fitting Final XGBoost Model on all historical data...")
    best_xgb.fit(X, y, sample_weight=sample_weights)
    
    os.makedirs('models', exist_ok=True)
    model_path = 'models/xgb_prod.pkl'
    joblib.dump(best_xgb, model_path)
    
    logging.info(f"Production model saved successfully to {model_path}!")

if __name__ == "__main__":
    train_production_model()