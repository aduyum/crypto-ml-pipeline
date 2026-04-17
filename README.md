# Quantitative ML Trading Pipeline (BTC/USDT)

## Overview
This repository implements a production-grade machine learning pipeline for classifying cryptocurrency trading signals. Developed with a focus on **industrial reliability** and **rigorous validation**, the system addresses the primary challenges in quantitative finance: data leakage, lookahead bias, and non-stationary market regimes.

## Key Features
- **Data Ingestion**: Robust pagination through the Binance API via `ccxt`, handling rate limits and data integrity.
- **Walk-Forward Validation**: A strict out-of-sample backtesting framework that simulates real-world trading by moving the training window forward in time.
- **Leakage Prevention**: Implementation of volatility-adjusted labeling (ATR-based) with strict point-in-time feature engineering.
- **Industrial Stack**: Built using **XGBoost** for high-performance inference and **Docker** for cloud-agnostic deployment.
- **Risk Management & Position Sizing**: Translates model confidence (`predict_proba`) into dynamic position sizes. Capital allocation is scaled inversely by volatility (ATR) and directly by the model's probabilistic certainty, mimicking institutional Kelly-criterion implementations.

## Technical Architecture
1. `data_fetcher.py`: Automated OHLCV retrieval and Parquet storage.
2. `features.py`: Advanced feature engineering including:
    - **Momentum/Volume**: RSI, MACD, OBV.
    - **Statistical Moments**: Rolling Skewness & Kurtosis to capture non-normal return distributions.
    - **Market Regimes**: Unsupervised Gaussian Mixture Models (GMM) to classify high vs. low volatility states.
3. `labels.py`: Classification targeting using a triple-barrier-style volatility adjusted approach.
4. `models.py`: Deep Learning sequence models (Gated Recurrent Units / GRU) with sliding lookback windows to capture temporal market memory, optimized with AdamW and gradient clipping.
5. `portfolio.py`: Capital allocation scaling via ATR and model probability confidence.
6. `walk_forward.py`: The cross-validation engine.
7. `Dockerfile`: Containerization for deployment to institutional-grade compute (e.g., OVH Cloud).

## How to Run
Ensure you have Docker installed:

```bash
docker build -t crypto-pipeline .
docker run crypto-pipeline
```

## Performance Note
The model achieves a **~48% accuracy** on a 3-class classification problem. In high-frequency volatile markets, this represents a strong signal-to-noise ratio, prioritizing risk management (the 'Hold' class) over aggressive over-trading.