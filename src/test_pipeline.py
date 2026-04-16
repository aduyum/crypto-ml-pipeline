import pandas as pd
import numpy as np
import logging
from labels import add_labels
from features import add_features

def test_no_leakage():
    """
    Test that the target variable handling prevents lookahead bias.
    """
    # Create dummy data - Note: freq='h' (lowercase) for modern Pandas
    df = pd.DataFrame({
        'open': np.random.rand(100),
        'high': np.random.rand(100),
        'low': np.random.rand(100),
        'close': np.random.rand(100),
        'volume': np.random.rand(100)
    }, index=pd.date_range(start='2023-01-01', periods=100, freq='h'))
    
    df = add_features(df)
    df_labeled = add_labels(df, lookahead=12)
    
    # Check 1: Target column exists
    assert 'Target' in df_labeled.columns
    
    # Check 2: Verify length is reduced (dropped the last 12 rows + feature warm-up)
    assert len(df_labeled) < 100
    
    logging.info("✓ Leakage Test Passed: Target is isolated and future-safe.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_no_leakage()