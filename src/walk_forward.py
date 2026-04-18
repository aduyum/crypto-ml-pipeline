import numpy as np
import warnings
import logging
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from xgboost import XGBClassifier
from models import PyTorchSequenceClassifier
from backtest_metrics import calculate_financial_metrics
import pandas as pd

warnings.filterwarnings('ignore')

def run_walk_forward(df, train_window=1000, test_size=200, lookahead=12):
    logging.info("Starting Nested Walk-Forward Validation with Purging & Class Balancing...")
    
    drop_cols =['open', 'high', 'low', 'close', 'volume', 'Target', 'Future_Return']
    feature_cols =[col for col in df.columns if col not in drop_cols]
    
    models_to_test =['Baseline_LogReg', 'RandomForest', 'Tuned_XGBoost', 'PyTorch_GRU']
    all_y_true =[]
    all_future_returns = []
    all_dates =[]
    predictions = {model:[] for model in models_to_test}
    
    total_steps = (len(df) - train_window) // test_size
    
    for step in range(total_steps):
        train_start = step * test_size
        train_end = train_start + train_window
        test_end = train_end + test_size
        
        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[train_end:test_end]
        
        # The last 'lookahead' rows of the train_df use future data that overlaps with test_df.
        # Drop them to prevent data leakage.
        train_df = train_df.iloc[:-lookahead]
        
        y_train = train_df['Target'].values
        
        # Compute weights to penalize the majority class ("Hold")
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        sample_weights = compute_sample_weight('balanced', y=y_train)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_df[feature_cols])
        X_test_scaled = scaler.transform(test_df[feature_cols])
        
        # Linear Baseline
        logreg = LogisticRegression(class_weight='balanced', max_iter=200, random_state=42)
        logreg.fit(X_train_scaled, y_train)
        predictions['Baseline_LogReg'].extend(logreg.predict(X_test_scaled))
        
        # Random Forest Baseline
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        predictions['RandomForest'].extend(rf.predict(X_test_scaled))
        
        # Dynamically Tuned XGBoost
        tscv = TimeSeriesSplit(n_splits=3)
        param_grid = {
            'max_depth':[3, 5, 7],
            'learning_rate':[0.01, 0.05, 0.1],
            'n_estimators':[50, 100, 200]
        }
        base_xgb = XGBClassifier(random_state=42, n_jobs=-1)
        tuned_xgb = RandomizedSearchCV(
            base_xgb, param_distributions=param_grid, n_iter=5, 
            cv=tscv, scoring='f1_macro', n_jobs=-1, random_state=42
        )
        # Pass sample_weights to XGBoost so it focuses on Buy/Sell classes
        tuned_xgb.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        predictions['Tuned_XGBoost'].extend(tuned_xgb.predict(X_test_scaled))
        
        # PyTorch Sequence Network (GRU)
        nn_model = PyTorchSequenceClassifier(input_dim=len(feature_cols), seq_length=12, epochs=20, class_weights=class_weights)
        nn_model.fit(X_train_scaled, y_train)
        predictions['PyTorch_GRU'].extend(nn_model.predict(X_test_scaled))
        
        all_y_true.extend(test_df['Target'].values)
        all_future_returns.extend(test_df['Future_Return'].values)
        all_dates.extend(test_df.index)
        
        logging.info(f"Fold {step+1:02d}/{total_steps} | Tested until {test_df.index[-1].date()} | Best XGB Params: {tuned_xgb.best_params_}")
        
    logging.info("\n================ MODEL ZOO PERFORMANCE ================")
    for model_name in models_to_test:
        f1_macro = f1_score(all_y_true, predictions[model_name], average='macro')
        logging.info(f"--- {model_name} (Macro F1: {f1_macro:.3f}) ---")
        logging.info("\n" + classification_report(all_y_true, predictions[model_name], target_names=['Hold', 'Buy', 'Sell']))
        
    # Run financial backtest on the best model (Tuned_XGBoost)
    df_results = pd.DataFrame({
        'Future_Return': all_future_returns,
        'Prediction': predictions['Tuned_XGBoost']
    }, index=all_dates)
    
    calculate_financial_metrics(df_results, fee_pct=0.001)

    return all_y_true, predictions