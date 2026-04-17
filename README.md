# Quantitative ML Trading Pipeline (BTC/USDT)

## Overview
This repository implements a machine learning pipeline for classifying cryptocurrency trading signals. Developed with a focus on **industrial reliability** and **rigorous validation**, the system addresses the primary challenges in quantitative finance: data leakage, lookahead bias, class imbalance, and non-stationary market regimes.

## Key Features
- **Data Ingestion**: Robust pagination through the Binance API via `ccxt`, handling rate limits and data integrity.
- **Walk-Forward Validation**: A strict out-of-sample backtesting framework that simulates real-world trading by moving the training window forward in time.
- **Nested Cross-Validation & Model Zoo**: Hyperparameter tuning is strictly isolated within the training folds using `TimeSeriesSplit`. Compares Linear Baselines, Random Forest, dynamically tuned XGBoost, and SOTA PyTorch Sequence models.
- **Leakage Prevention (Purging)**: Implements combinatorial purging (inspired by Marcos López de Prado) to drop training samples whose lookahead windows overlap with the out-of-sample test set.
- **Class Imbalance Handling**: Dynamic class and sample weighting force the models to focus on actionable minority classes (Buy/Sell) rather than defaulting to the majority class (Hold).
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
6. `walk_forward.py`: Nested Cross-Validation engine evaluating the Model Zoo using strictly out-of-sample metrics.
7. `Dockerfile`: Containerization for deployment to institutional-grade compute (e.g., OVH Cloud).

## How to Run
Ensure you have Docker installed:

```bash
docker build -t crypto-pipeline .
docker run crypto-pipeline
```

## Performance Note
The pipeline deliberately sacrifices raw accuracy (stabilizing around **~34-35%**) to achieve realistic, actionable trading metrics. 

Initially, naive models achieved ~50% accuracy by blindly predicting "Hold" (the majority class), yielding a "Buy/Sell" recall of just 2%—a classic trap in financial machine learning. By implementing **Combinatorial Purging** (removing overlapping label leakage) and **Balanced Sample Weighting**, the model's actionable minority-class recall jumped to **~35-40%**. 