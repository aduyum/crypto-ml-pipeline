import pandas as pd
import warnings
import logging
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

def run_walk_forward(df, train_window=1000, test_size=200):
    logging.info("Starting Walk-Forward Validation...")
    
    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'Target']
    feature_cols = [col for col in df.columns if col not in drop_cols]
    
    all_y_true, all_y_pred = [], []
    total_steps = (len(df) - train_window) // test_size
    
    for step in range(total_steps):
        train_start = step * test_size
        train_end = train_start + train_window
        test_end = train_end + test_size
        
        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[train_end:test_end]
        
        model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42, n_jobs=-1)
        model.fit(train_df[feature_cols], train_df['Target'])
        preds = model.predict(test_df[feature_cols])
        
        all_y_true.extend(test_df['Target'].values)
        all_y_pred.extend(preds)
        
        logging.info(f"Fold {step+1:02d}/{total_steps} | Tested until {test_df.index[-1].date()}")
        
    logging.info("\n=== Final Backtest Report ===\n" + classification_report(all_y_true, all_y_pred, target_names=['Hold', 'Buy', 'Sell']))
    return all_y_true, all_y_pred