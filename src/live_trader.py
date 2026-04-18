import ccxt
import pandas as pd
import joblib
import time
import logging
from typing import Optional, Dict
from features import add_features
from portfolio import calculate_position_size

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class LiveExecutionEngine:
    def __init__(self, symbol: str = 'BTC/USDT', timeframe: str = '1h', capital: float = 10000.0):
        self.symbol = symbol
        self.timeframe = timeframe
        self.capital = capital
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.current_inventory_usd = 0.0
        
        try:
            self.model = joblib.load('models/xgb_prod.pkl')
            logging.info("Production Model Loaded Successfully.")
        except Exception as e:
            logging.error(f"Failed to load model. Run train_prod.py first. Error: {e}")
            raise

    def get_market_state(self) -> Optional[pd.DataFrame]:
        """Fetches OHLCV and calculates features with retry logic."""
        retries = 3
        for attempt in range(retries):
            try:
                # Fetch enough candles to warm up the 20-period GMM and 14-period RSI
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=50)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                return add_features(df)
            except ccxt.NetworkError as e:
                wait_time = 2 ** attempt
                logging.warning(f"Network error fetching market state. Retrying in {wait_time}s... ({e})")
                time.sleep(wait_time)
            except Exception as e:
                logging.error(f"Unexpected error fetching market data: {e}")
                return None
        return None

    def execute_trade(self, target_position_usd: float, current_price: float, signal: str):
        """Calculates the delta and executes the required trade to reach the target inventory."""
        # Calculate Delta: How much do we need to buy/sell to reach the target?
        trade_delta_usd = target_position_usd - self.current_inventory_usd
        
        # Minimum trade size check
        if abs(trade_delta_usd) < 10.0:
            logging.info("Trade delta too small. No execution required.")
            return

        trade_amount_btc = abs(trade_delta_usd) / current_price
        side = "BUY" if trade_delta_usd > 0 else "SELL"

        logging.info(f"*** EXECUTING {side} ORDER: {trade_amount_btc:.5f} BTC (${abs(trade_delta_usd):.2f}) ***")
        
        # Normally, self.exchange.create_market_order() but for here:
        self.exchange.create_market_order(self.symbol, side.lower(), trade_amount_btc)
        
        # Update local state
        self.current_inventory_usd = target_position_usd

    def run(self, poll_interval_seconds: int = 60):
        logging.info(f"--- INITIALIZING LIVE TRADING ENGINE: {self.symbol} ---")
        logging.info(f"Starting capital: ${self.capital:,.2f} | Polling every {poll_interval_seconds}s")
        
        while True:
            try:
                df_features = self.get_market_state()
                if df_features is None or df_features.empty:
                    time.sleep(poll_interval_seconds)
                    continue

                latest_candle = df_features.iloc[-1:]
                current_price = latest_candle['close'].values[0]
                current_atr = latest_candle['ATR_14'].values[0]
                
                drop_cols = ['open', 'high', 'low', 'close', 'volume', 'Target']
                X_live = latest_candle.drop(columns=[c for c in drop_cols if c in latest_candle.columns])
                
                # Predict
                probs = self.model.predict_proba(X_live)[0]
                prediction = int(self.model.predict(X_live)[0])
                
                classes = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
                signal = classes[prediction]
                confidence = probs[prediction]
                
                # Calculate absolute target portfolio exposure based on risk rules
                target_position = calculate_position_size(
                    prediction=prediction,
                    probabilities=probs,
                    atr_14=current_atr,
                    close_price=current_price,
                    capital=self.capital
                )
                
                # If signal is SELL, target exposure is technically 0 (or short if allowed). 
                # Assuming long-only for crypto spot markets:
                if signal == 'SELL':
                    target_position = 0.0

                logging.info("================ LIVE MARKET UPDATE ================")
                logging.info(f"Current Price:      ${current_price:,.2f}")
                logging.info(f"Current Inventory:  ${self.current_inventory_usd:,.2f}")
                logging.info(f"Model Signal:       {signal} (Confidence: {confidence:.1%})")
                logging.info(f"Target Exposure:    ${target_position:,.2f}")
                
                # Reconcile inventory delta
                self.execute_trade(target_position, current_price, signal)
                
                logging.info("====================================================")
                
                # Wait for the next candle/interval
                time.sleep(poll_interval_seconds)

            except KeyboardInterrupt:
                logging.info("Live engine stopped by user.")
                break
            except Exception as e:
                logging.error(f"Live Execution Loop Error: {e}")
                time.sleep(poll_interval_seconds)

if __name__ == "__main__":
    # In production, poll_interval_seconds would be 3600 (1 hour) to match the 1h timeframe
    # For testing purposes, here it's 10 seconds.
    engine = LiveExecutionEngine()
    engine.run(poll_interval_seconds=10)