import numpy as np

def calculate_risk(data):
    """
    Returns volatility metrics for the dashboard.
    """
    returns = data["Close"].pct_change().dropna()
    volatility = np.std(returns) * (252 ** 0.5)
    
    if volatility < 0.2:
        level = "Low Risk"
    elif volatility < 0.4:
        level = "Medium Risk"
    else:
        level = "High Risk"
        
    return volatility, level

def get_stop_loss_price(df, direction="long", atr_multiplier=2.0):
    """
    Calculates the exact price where you MUST sell to prevent big losses.
    """
    current_price = df['Close'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    
    if direction == "long":
        stop_loss = current_price - (atr * atr_multiplier)
    else:
        stop_loss = current_price + (atr * atr_multiplier)
        
    return stop_loss

def calculate_position_size(account_balance, entry_price, stop_loss_price, risk_per_trade=0.02):
    """
    Calculates how many shares to buy so you never lose more than 2% of your account.
    """
    # 1. How much cash are we willing to risk? (e.g., $1000 * 0.02 = $20 risk)
    risk_amount = account_balance * risk_per_trade
    
    # 2. Risk per share
    risk_per_share = abs(entry_price - stop_loss_price)
    
    if risk_per_share == 0:
        return 0
        
    # 3. Position Size
    position_size = risk_amount / risk_per_share
    
    # Return as integer (can't buy 0.5 shares usually)
    return int(position_size)