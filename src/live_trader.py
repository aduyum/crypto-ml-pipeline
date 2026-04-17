import ccxt
import pandas as pd
import joblib
import time
import logging
from features import add_features
from portfolio import calculate_position_size

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_live_loop(symbol='BTC/USDT', timeframe='1h', capital=10000):
    logging.info(f"--- INITIALIZING LIVE TRADING ENGINE: {symbol} ---")
    
    try:
        model = joblib.load('models/xgb_prod.pkl')
        logging.info("Production Model Loaded.")
    except Exception as e:
        logging.error(f"Failed to load model. Did you run train_prod.py? Error: {e}")
        return

    exchange = ccxt.binance()
    
    # Run it once for the demonstration (normally would be an infinite while True: loop)
    logging.info("Fetching latest market state from Binance...")
    
    try:
        # Fetch enough candles to warm up the 20-period GMM and 14-period RSI
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=50)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Calculate features on the fly
        df_features = add_features(df)
        
        # Get the latest candle
        latest_candle = df_features.iloc[-1:]
        current_price = latest_candle['close'].values[0]
        current_atr = latest_candle['ATR_14'].values[0]
        
        drop_cols =['open', 'high', 'low', 'close', 'volume']
        X_live = latest_candle.drop(columns=[c for c in drop_cols if c in latest_candle.columns])
        
        # Predict
        probs = model.predict_proba(X_live)[0]
        prediction = int(model.predict(X_live)[0])
        
        classes = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        signal = classes[prediction]
        confidence = probs[prediction]
        
        # Risk Management
        target_position = calculate_position_size(
            prediction=prediction,
            probabilities=probs,
            atr_14=current_atr,
            close_price=current_price,
            capital=capital
        )
        
        logging.info("================ LIVE MARKET UPDATE ================")
        logging.info(f"Current Price:  ${current_price:,.2f}")
        logging.info(f"Market Regime:  {'High Volatility' if latest_candle['Vol_Regime'].values[0] == 1 else 'Low Volatility'}")
        logging.info(f"Model Signal:   {signal} (Confidence: {confidence:.1%})")
        
        if signal in ['BUY', 'SELL']:
            logging.info(f"Action: Execution engine requesting {signal} order.")
            logging.info(f"Capital Allocation: ${target_position:,.2f} based on Volatility/Confidence.")
        else:
            logging.info("Action: Staying flat. Waiting for next candle.")
        logging.info("====================================================")

    except Exception as e:
        logging.error(f"Live Execution Error: {e}")

if __name__ == "__main__":
    run_live_loop()