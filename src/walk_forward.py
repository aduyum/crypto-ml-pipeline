import pandas as pd
import numpy as np
import warnings
import logging
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from portfolio import calculate_position_size
from models import PyTorchSequenceClassifier

warnings.filterwarnings('ignore')

def run_walk_forward(df, train_window=1000, test_size=200):
    logging.info("Starting Walk-Forward Validation: XGBoost vs PyTorch NN...")
    
    drop_cols =['open', 'high', 'low', 'close', 'volume', 'Target']
    feature_cols =[col for col in df.columns if col not in drop_cols]
    
    all_y_true, all_y_pred_xgb, all_y_pred_nn = [],[],[]
    total_steps = (len(df) - train_window) // test_size
    
    for step in range(total_steps):
        train_start = step * test_size
        train_end = train_start + train_window
        test_end = train_end + test_size
        
        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[train_end:test_end]
        
        # Neural Networks require scaled data to converge properly
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_df[feature_cols])
        X_test_scaled = scaler.transform(test_df[feature_cols])
        y_train = train_df['Target'].values
        
        # XGBoost
        xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42, n_jobs=-1)
        xgb_model.fit(X_train_scaled, y_train)
        xgb_preds = xgb_model.predict(X_test_scaled)
        
        # PyTorch Sequence Neural Network (GRU)
        nn_model = PyTorchSequenceClassifier(input_dim=len(feature_cols), seq_length=12, epochs=25)
        nn_model.fit(X_train_scaled, y_train)
        nn_preds = nn_model.predict(X_test_scaled)
        
        all_y_true.extend(test_df['Target'].values)
        all_y_pred_xgb.extend(xgb_preds)
        all_y_pred_nn.extend(nn_preds)
        
        logging.info(f"Fold {step+1:02d}/{total_steps} | Tested until {test_df.index[-1].date()}")
        
    logging.info("\n=== XGBoost Backtest Report ===\n" + classification_report(all_y_true, all_y_pred_xgb, target_names=['Hold', 'Buy', 'Sell']))
    logging.info("\n=== PyTorch NN Backtest Report ===\n" + classification_report(all_y_true, all_y_pred_nn, target_names=['Hold', 'Buy', 'Sell']))
    
    return all_y_true, all_y_pred_xgb, all_y_pred_nn