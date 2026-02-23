import pandas as pd
import numpy as np

def run_backtest(data, model, features, initial_capital=10000, risk_per_trade=0.02):
    
    # 1. Setup Data
    df = data.copy()
    df["Prediction"] = model.predict(df[features])
    
    # 2. Simulation Variables
    capital = initial_capital
    position = 0       
    entry_price = 0
    stop_loss_price = 0
    
    commission = 0.001 
    slippage = 0.001   
    
    equity_curve = [initial_capital]
    trade_log = [] 
    
    # 3. Loop through candles
    for i in range(len(df) - 1):
        
        current_date = df.index[i]
        prediction = df.iloc[i]["Prediction"]
        current_close = df.iloc[i]["Close"]
        atr = df.iloc[i]["ATR"]
        
        # --- NEW FILTERS ---
        # 1. Trend Filter: Only Buy if Price > EMA200
        # (We use .get to avoid crashing if EMA200 isn't there yet)
        ema200 = df.iloc[i].get("EMA200", 0) 
        trend_filter_pass = current_close > ema200
        
        # 2. Volatility Filter: Avoid dead markets
        # We only trade if ATR is at least 0.5% of price
        vol_filter_pass = atr > (current_close * 0.004) 
        
        next_open = df.iloc[i+1]["Open"]
        
     # --- EXIT LOGIC (Updated with Take Profit) ---
        if position > 0:
            exit_price = 0
            reason = ""
            
            # 1. Define Take Profit Price (Dynamic)
            # We target a win that is 4 times bigger than the volatility (Risk:Reward Management)
            take_profit_price = entry_price + (atr * 4.0)

            # 2. Check Stop Loss FIRST (Safety First)
            if df.iloc[i+1]["Low"] < stop_loss_price:
                exit_price = stop_loss_price
                reason = "Stop Loss"
            
            # 3. Check Take Profit (Greed Management)
            elif df.iloc[i+1]["High"] > take_profit_price:
                exit_price = take_profit_price
                reason = "Take Profit 💰"
            
            # 4. Check Signal (Sell)
            elif prediction == 0:
                exit_price = next_open * (1 - slippage) 
                reason = "Signal"
            
            # If we exited...
            if exit_price > 0:
                sell_value = position * exit_price
                capital += sell_value
                capital -= (sell_value * commission)
                
                pnl = (exit_price - entry_price) * position
                trade_log.append({
                    "Date": current_date,
                    "Type": "Sell",
                    "Entry Price": entry_price,
                    "Exit Price": exit_price,
                    "PnL": pnl,
                    "Return Pct": (exit_price - entry_price) / entry_price,
                    "Reason": reason
                })
                position = 0
        # --- ENTRY LOGIC (With New Filters!) ---
        # We now check: Prediction AND Trend Filter AND Volatility Filter
        if position == 0 and prediction == 1 and trend_filter_pass and vol_filter_pass:
            
            buy_price = next_open * (1 + slippage)
            sl_dist = atr * 3
            temp_sl = buy_price - sl_dist
            
            risk_per_share = buy_price - temp_sl
            if risk_per_share > 0:
                risk_amt = capital * risk_per_trade
                shares_to_buy = int(risk_amt / risk_per_share)
                
                cost = shares_to_buy * buy_price
                if cost < capital and shares_to_buy > 0:
                    capital -= cost
                    capital -= (cost * commission)
                    
                    position = shares_to_buy
                    entry_price = buy_price
                    stop_loss_price = temp_sl

        # --- UPDATE EQUITY ---
        mark_price = df.iloc[i+1]["Close"]
        if position > 0:
            current_equity = capital + (position * mark_price)
        else:
            current_equity = capital
            
        equity_curve.append(current_equity)

    df["Equity"] = equity_curve
    return df, trade_log