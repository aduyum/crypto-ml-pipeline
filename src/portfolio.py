import logging

def calculate_position_size(prediction, probabilities, atr_14, close_price, capital=10000, risk_pct=0.02):
    """
    Calculates dynamic position size based on prediction confidence and volatility.
    
    Args:
        prediction (int): 0 (Hold), 1 (Buy), 2 (Sell)
        probabilities (array): Probabilities for each class[P(Hold), P(Buy), P(Sell)]
        atr_14 (float): Average True Range for volatility scaling
        close_price (float): Current asset price
        capital (float): Total portfolio capital available
        risk_pct (float): Maximum risk percentage of total capital per trade (default 2%)
    """
    if prediction == 0:
        return 0.0  # Hold means we don't allocate capital
        
    confidence = probabilities[prediction]
    
    # 1. Volatility Scaling (Risk Amount)
    # Stop loss distance is 1.5 * ATR (Aligning with our label generation threshold)
    stop_loss_distance = 1.5 * atr_14
    if stop_loss_distance <= 0:
        return 0.0
        
    # Maximum dollar risk based on portfolio size
    max_risk_usd = capital * risk_pct
    
    # Base position size (how much of the asset to buy so a hit to stop_loss loses max_risk_usd)
    base_position_size_usd = (max_risk_usd / stop_loss_distance) * close_price
    
    # 2. Probability Scaling (Confidence Factor)
    # We penalize the position size if the model isn't highly confident
    # e.g., if confidence is 45%, we take 45% of the base position. 
    scaled_position_size = base_position_size_usd * confidence
    
    # Cap the allocation so we never use leverage (exceed capital)
    final_allocation = min(scaled_position_size, capital)
    
    return final_allocation