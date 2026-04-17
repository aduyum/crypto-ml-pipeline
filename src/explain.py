import joblib
import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

def explain_model():
    logging.info("Generating Global Feature Importance...")
    
    # 1. Load the production model
    try:
        model = joblib.load('models/xgb_prod.pkl')
    except FileNotFoundError:
        logging.error("Model file not found. Please run src/train_prod.py first.")
        return

    # 2. Get feature importance (Gain)
    importance = model.get_booster().get_score(importance_type='gain')
    
    if not importance:
        logging.warning("No importance scores found. Ensure the model was trained on more than 1 feature.")
        return

    # 3. Convert to Series and Sort
    series = pd.Series(importance).sort_values(ascending=True)
    
    # 4. Plotting
    plt.figure(figsize=(12, 8))
    # Top 15 features 
    series.tail(15).plot(kind='barh', color='teal')
    plt.title("Top Feature Importance: XGBoost Gain (Alpha Drivers)")
    plt.xlabel("Average Gain per Split")
    plt.ylabel("Features")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    save_path = 'assets/feature_importance.png'
    plt.savefig(save_path)
    logging.info(f"Chart saved as {save_path}")
    
    # 5. Insight Log
    top_3 = series.tail(3).index.tolist()[::-1]
    logging.info(f"Top 3 Features driving the model: {', '.join(top_3)}")

if __name__ == "__main__":
    explain_model()