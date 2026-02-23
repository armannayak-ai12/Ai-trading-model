import numpy as np
import pandas as pd

def calculate_metrics(equity_curve, trade_log):
    
    # 1. Basic Returns
    returns = np.diff(equity_curve) / equity_curve[:-1]
    total_return = (equity_curve[-1] / equity_curve[0]) - 1
    
    # 2. Risk Metrics (Sharpe & Drawdown)
    if np.std(returns) != 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe = 0
        
    cumulative = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cumulative) / cumulative
    max_drawdown = drawdown.min()
    
    # 3. Trade Analysis (Win Rate, Profit Factor, etc.)
    if len(trade_log) > 0:
        trades = pd.DataFrame(trade_log)
        wins = trades[trades["PnL"] > 0]
        losses = trades[trades["PnL"] <= 0]
        
        win_rate = len(wins) / len(trades)
        
        avg_win = wins["PnL"].mean() if len(wins) > 0 else 0
        avg_loss = losses["PnL"].mean() if len(losses) > 0 else 0
        
        if abs(losses["PnL"].sum()) > 0:
            profit_factor = wins["PnL"].sum() / abs(losses["PnL"].sum())
        else:
            profit_factor = 10.0 # Infinite profit factor handling
            
        trade_count = len(trades)
    else:
        win_rate = 0
        profit_factor = 0
        avg_win = 0
        avg_loss = 0
        trade_count = 0

    return {
        "Total Return": total_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate,
        "Profit Factor": profit_factor,
        "Trade Count": trade_count,
        "Avg Win": avg_win,
        "Avg Loss": avg_loss
    }