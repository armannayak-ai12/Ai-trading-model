from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model(data):
    # Setup Target: 1 if price goes UP next candle, 0 if DOWN
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    
    # Use the NEW Advanced Features
    features = [
        "Close", "RSI", "MACD", "MACD_signal", "EMA20", "EMA50", 
        "ADX", "ATR", "Log_Ret", "Volatility_20", "RSI_Lag1"
    ]
    
    # Filter data to ensure all columns exist
    data = data.dropna()
    
    X = data[features]
    y = data["Target"]
    
    if len(X) < 10:
        raise ValueError("Not enough data to train model")
        
    
    # shuffle=False ensures we train on the PAST (first 80%) and test on the FUTURE (last 20%)
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=10, 
        min_samples_split=5, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    
    
    # Returns 4 items: Model, Score, Test Data (for backtest), and Latest Row (for prediction)
    return model, accuracy, X_test, X.iloc[[-1]]