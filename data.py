import yfinance as yf

def get_data(symbol="AAPL", period="7d", interval="1m"):
    data = yf.download(
        symbol,
        period=period,
        interval=interval
    )
    data.columns = data.columns.map(lambda x: x[0] if isinstance(x, tuple) else x)
    return data
