import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import model as mdl  # Import your existing model logic
import backtest as bt # Import your existing backtest logic

def walk_forward_test(data, features, train_window_days=90, test_window_days=30):
    """
    Performs Walk-Forward Validation.
    1. Trains on a rolling window (e.g., 90 days).
    2. Tests on the NEXT window (e.g., 30 days).
    3. Slides forward and repeats.
    """
    
    # Sort data just in case
    data = data.sort_index()
    
    results = []
    equity_curve = []
    
    # Calculate indices
    # We need enough data for the first training block
    start_index = 0
    total_days = (data.index[-1] - data.index[0]).days
    
    # We iterate through time
    current_date = data.index[0] + pd.Timedelta(days=train_window_days)
    
    full_trade_log = []
    
    print(f"🔄 Starting Walk-Forward: Train {train_window_days}d / Test {test_window_days}d")
    
    while current_date < data.index[-1]:
        # 1. Define Time Ranges
        train_start = current_date - pd.Timedelta(days=train_window_days)
        train_end = current_date
        
        test_start = current_date
        test_end = current_date + pd.Timedelta(days=test_window_days)
        
        # 2. Slice Data
        train_data = data[(data.index >= train_start) & (data.index < train_end)].copy()
        test_data = data[(data.index >= test_start) & (data.index < test_end)].copy()
        
        if len(test_data) < 10: # Skip if not enough data
            current_date += pd.Timedelta(days=test_window_days)
            continue
            
        # 3. Train Model (Freshly trained for this specific period)
        # We assume your model.py has a train function. We adapt it here.
        model, acc, _, _ = mdl.train_model(train_data)
        
        # 4. Test Model (Backtest on unseen future data)
        # We need to predict first
        test_data['Prediction'] = model.predict(test_data[features])
        
        # Run Backtest logic
        # We reuse your existing backtest engine, but we need to trick it 
        # into returning just the log, not the full graph data usually.
        # Note: We need to pass the model, but we already added predictions.
        bt_df, log = bt.run_backtest(test_data, model, features, initial_capital=10000)
        
        # 5. Store Results
        # Accumulate trades
        full_trade_log.extend(log)
        
        # Store metrics for this window
        window_return = (bt_df['Equity'].iloc[-1] - bt_df['Equity'].iloc[0])
        results.append({
            "Period Start": test_start.date(),
            "Period End": test_end.date(),
            "Train Accuracy": acc,
            "Profit/Loss": window_return,
            "Trades": len(log)
        })
        
        # Move Forward
        current_date += pd.Timedelta(days=test_window_days)
        
    return pd.DataFrame(results), full_trade_log