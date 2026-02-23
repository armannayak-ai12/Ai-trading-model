import pandas as pd
import pandas_ta as ta
import numpy as np  # <--- Make sure this import is here!

def add_indicators(data):
    df = data.copy()
    
    # --- 1. Momentum & Trend ---
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['EMA20'] = ta.ema(df['Close'], length=20)
    df['EMA50'] = ta.ema(df['Close'], length=50)

    # Add this line with your other EMAs
   # Correct syntax for pandas_ta
    data['EMA200'] = ta.ema(data['Close'], length=200)

    df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
    
    # --- 2. Volatility ---
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # --- 3. Advanced Features ---
    # FIX: Use 'np.log' instead of 'pd.np.log'
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    df['Volatility_20'] = df['Log_Ret'].rolling(window=20).std()
    
    df['RSI_Lag1'] = df['RSI'].shift(1)
    df['Close_Lag1'] = df['Close'].shift(1)
    
    df.dropna(inplace=True)
    
    return df