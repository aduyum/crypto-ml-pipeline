import pandas as pd
import numpy as np
import warnings
import logging
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from portfolio import calculate_position_size

warnings.filterwarnings('ignore')

def run_walk_forward(df, train_window=1000, test_size=200):
    logging.info("Starting Walk-Forward Validation & Risk Simulation...")
    
    drop_cols =['open', 'high', 'low', 'close', 'volume', 'Target']
    feature_cols =[col for col in df.columns if col not in drop_cols]
    
    all_y_true, all_y_pred =[],[]
    total_steps = (len(df) - train_window) // test_size
    
    total_trades = 0
    avg_confidence = []
    avg_position_size =[]
    
    for step in range(total_steps):
        train_start = step * test_size
        train_end = train_start + train_window
        test_end = train_end + test_size
        
        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[train_end:test_end]
        
        model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42, n_jobs=-1)
        model.fit(train_df[feature_cols], train_df['Target'])
        
        # Use probabilities for trading
        probs = model.predict_proba(test_df[feature_cols])
        preds = np.argmax(probs, axis=1)
        
        all_y_true.extend(test_df['Target'].values)
        all_y_pred.extend(preds)
        
        # Simulate position sizing
        for i in range(len(preds)):
            pred = preds[i]
            prob = probs[i]
            row = test_df.iloc[i]
            
            if pred in [1, 2]:  # If buy or sell
                size = calculate_position_size(
                    prediction=pred, 
                    probabilities=prob, 
                    atr_14=row['ATR_14'], 
                    close_price=row['close'], 
                    capital=10000, 
                    risk_pct=0.02
                )
                total_trades += 1
                avg_confidence.append(prob[pred])
                avg_position_size.append(size)
                
        logging.info(f"Fold {step+1:02d}/{total_steps} | Tested until {test_df.index[-1].date()}")
        
    logging.info("\n=== Final Classification Report ===\n" + classification_report(all_y_true, all_y_pred, target_names=['Hold', 'Buy', 'Sell']))
    
    if total_trades > 0:
        logging.info("=== Risk Management Simulation ===")
        logging.info(f"Total Actionable Signals: {total_trades}")
        logging.info(f"Average Model Confidence: {np.mean(avg_confidence):.2%}")
        logging.info(f"Average Position Sized : ${np.mean(avg_position_size):.2f} (on $10k Capital base)")
        
    return all_y_true, all_y_pred