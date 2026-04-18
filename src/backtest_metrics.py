import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s[%(levelname)s] %(message)s')

def calculate_financial_metrics(df_results, fee_pct=0.001):
    """
    Simulates portfolio PnL based on model predictions.
    df_results must contain: 'Future_Return' and 'Prediction' (0=Hold, 1=Buy, 2=Sell)
    """
    logging.info("Calculating Financial Metrics (Simulating Live PnL)...")

    # Map predictions to positions: 1 for Long, -1 for Short, 0 for Flat
    df_results['Position'] = df_results['Prediction'].map({0: 0, 1: 1, 2: -1})

    # Apply fees only when the position changes (Turnover)
    df_results['Trade_Fee'] = np.where(df_results['Position'] != df_results['Position'].shift(1), fee_pct, 0)
    
    # Strategy Return = (Position * Asset Return) - Trading Fees
    df_results['Strategy_Return'] = (df_results['Position'] * df_results['Future_Return']) - df_results['Trade_Fee']
    
    df_results['Cumulative_Market'] = (1 + df_results['Future_Return']).cumprod()
    df_results['Cumulative_Strategy'] = (1 + df_results['Strategy_Return']).cumprod()

    # Calculate Metrics (Assuming 1h timeframe -> 8760 periods/year)
    periods_per_year = 8760
    mean_return = df_results['Strategy_Return'].mean()
    std_return = df_results['Strategy_Return'].std()
    
    sharpe_ratio = (mean_return / std_return) * np.sqrt(periods_per_year) if std_return > 0 else 0
    
    # Max Drawdown
    rolling_max = df_results['Cumulative_Strategy'].cummax()
    drawdown = df_results['Cumulative_Strategy'] / rolling_max - 1
    max_drawdown = drawdown.min()
    
    # Win Rate
    winning_trades = len(df_results[df_results['Strategy_Return'] > 0])
    total_active_trades = len(df_results[df_results['Position'] != 0])
    win_rate = winning_trades / total_active_trades if total_active_trades > 0 else 0

    logging.info("\n================ FINANCIAL METRICS (OOS) ================")
    logging.info(f"Total Strategy Return: {(df_results['Cumulative_Strategy'].iloc[-1] - 1)*100:.2f}%")
    logging.info(f"Total Market Return:   {(df_results['Cumulative_Market'].iloc[-1] - 1)*100:.2f}%")
    logging.info(f"Sharpe Ratio:          {sharpe_ratio:.2f}")
    logging.info(f"Max Drawdown:          {max_drawdown*100:.2f}%")
    logging.info(f"Win Rate (Active):     {win_rate*100:.2f}%")
    logging.info("=========================================================\n")

    # Plotting
    os.makedirs('assets', exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(df_results.index, df_results['Cumulative_Market'], label='Market (Buy & Hold)', color='gray', alpha=0.7)
    plt.plot(df_results.index, df_results['Cumulative_Strategy'], label='Strategy (Tuned XGBoost)', color='teal', linewidth=2)
    plt.title("Out-of-Sample Equity Curve vs Market (Includes 0.1% Fees)")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    save_path = 'assets/equity_curve.png'
    plt.savefig(save_path)
    logging.info(f"Equity curve saved to {save_path}")

    return df_results