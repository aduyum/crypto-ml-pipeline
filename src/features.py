import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.mixture import GaussianMixture
import logging

def add_features(df):
    
    logging.info("Engineering technical, statistical, and regime features...")
    df = df.copy()
    
    # 1. Standard Technical Indicators (Momentum & Volume)
    df['RSI_14'] = df.ta.rsi(length=14)
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)
    df['ATR_14'] = df.ta.atr(length=14)
    df['OBV'] = df.ta.obv()
    df['Log_Return'] = df.ta.log_return(length=1)
    
    # 2. Statistical Moments (Shape of the distribution)
    logging.info("Calculating statistical moments (Skewness & Kurtosis)...")
    df['Roll_Skew_14'] = df['Log_Return'].rolling(14).skew()
    df['Roll_Kurt_14'] = df['Log_Return'].rolling(14).kurt()
    df['Returns_Vol_20'] = df['Log_Return'].rolling(20).std()
    
    df.dropna(inplace=True)
    
    # 3. Market Regime Detection (Unsupervised ML)
    # Using a Gaussian Mixture Model to classify the market into Low (0) or High (1) Volatility
    logging.info("Fitting Gaussian Mixture Model for Volatility Regimes...")
    vol_data = df['Returns_Vol_20'].values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42, n_init=3)
    gmm.fit(vol_data)
    
    regimes = gmm.predict(vol_data)
    
    # Ensure '1' is consistently the High Volatility regime
    if gmm.means_[0][0] > gmm.means_[1][0]:
        regimes = 1 - regimes
        
    df['Vol_Regime'] = regimes
    
    logging.info(f"Feature set complete. High-Vol Regime instances: {df['Vol_Regime'].sum()}")
    logging.info(f"Final Feature Shape: {df.shape}")
    
    return df