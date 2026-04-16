# Quantitative ML Trading Pipeline (BTC/USDT)

## Overview
This repository implements a production-grade machine learning pipeline for classifying cryptocurrency trading signals. Developed with a focus on **industrial reliability** and **rigorous validation**, the system addresses the primary challenges in quantitative finance: data leakage, lookahead bias, and non-stationary market regimes.

## Key Features
- **Data Ingestion**: Robust pagination through the Binance API via `ccxt`, handling rate limits and data integrity.
- **Walk-Forward Validation**: A strict out-of-sample backtesting framework that simulates real-world trading by moving the training window forward in time.
- **Leakage Prevention**: Implementation of volatility-adjusted labeling (ATR-based) with strict point-in-time feature engineering.
- **Industrial Stack**: Built using **XGBoost** for high-performance inference and **Docker** for cloud-agnostic deployment.

## Technical Architecture
1. `data_fetcher.py`: Automated OHLCV retrieval and Parquet storage.
2. `features.py`: Time-series signal extraction (Momentum, Volatility, Volume).
3. `labels.py`: Classification targeting using a triple-barrier-style volatility adjusted approach.
4. `walk_forward.py`: The cross-validation engine.
5. `Dockerfile`: Containerization for deployment to institutional-grade compute (e.g., OVH Cloud).

## How to Run
Ensure you have Docker installed:

```bash
docker build -t crypto-pipeline .
docker run crypto-pipeline
```

## Performance Note
The model achieves a **~48% accuracy** on a 3-class classification problem. In high-frequency volatile markets, this represents a strong signal-to-noise ratio, prioritizing risk management (the 'Hold' class) over aggressive over-trading.
```