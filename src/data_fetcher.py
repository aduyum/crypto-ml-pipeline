import ccxt
import pandas as pd
import os
import time
import logging

def fetch_binance_data(symbol='BTC/USDT', timeframe='1h', limit=4000):
    logging.info(f"Fetching {limit} candles of {symbol} from Binance...")
    
    exchange = ccxt.binance({'enableRateLimit': True})
    all_ohlcv = []
    now_ms = exchange.milliseconds()
    since_ms = now_ms - (limit * 60 * 60 * 1000)

    while len(all_ohlcv) < limit:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since_ms = ohlcv[-1][0] + 1
            logging.info(f"Progress: {len(all_ohlcv)}/{limit} candles...")
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            logging.error(f"API Error: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')].tail(limit)
    
    os.makedirs('data', exist_ok=True)
    filepath = 'data/btc_1h_raw.parquet'
    df.to_parquet(filepath)
    logging.info(f"Data saved to {filepath}")
    return df