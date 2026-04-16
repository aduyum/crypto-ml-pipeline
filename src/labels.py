import pandas as pd
import numpy as np
import logging

def add_labels(df, lookahead=12, atr_multiplier=1.5):
    logging.info(f"Labeling data with {lookahead}-hour lookahead...")
    df = df.copy()
    
    df['Future_Close'] = df['close'].shift(-lookahead)
    df['Future_Return'] = (df['Future_Close'] - df['close']) / df['close']
    df['ATR_Pct'] = df['ATR_14'] / df['close']
    
    buy_condition = df['Future_Return'] > (df['ATR_Pct'] * atr_multiplier)
    sell_condition = df['Future_Return'] < -(df['ATR_Pct'] * atr_multiplier)
    
    df['Target'] = 0
    df.loc[buy_condition, 'Target'] = 1
    df.loc[sell_condition, 'Target'] = 2
    
    df.dropna(subset=['Future_Close'], inplace=True)
    df.drop(columns=['Future_Close', 'Future_Return', 'ATR_Pct'], inplace=True)
    
    dist = df['Target'].value_counts(normalize=True) * 100
    logging.info(f"Labels generated. Class Dist: Hold={dist.get(0,0):.1f}%, Buy={dist.get(1,0):.1f}%, Sell={dist.get(2,0):.1f}%")
    
    return df