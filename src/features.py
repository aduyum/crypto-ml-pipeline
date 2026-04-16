import pandas as pd
import pandas_ta as ta
import logging

def add_features(df):
    logging.info("Engineering technical indicators...")
    df = df.copy()
    
    df['RSI_14'] = df.ta.rsi(length=14)
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)
    df['ATR_14'] = df.ta.atr(length=14)
    df['OBV'] = df.ta.obv()
    df['Log_Return'] = df.ta.log_return(length=1)
    
    df.dropna(inplace=True)
    logging.info(f"Feature set complete. Shape: {df.shape}")
    return df