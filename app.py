import streamlit as st
from backtest import run_backtest
from performance import calculate_metrics
import pandas as pd
import plotly.express as px
import news_analysis  # <--- NEW IMPORT
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go

# --- IMPORT YOUR MODULES ---
from data import get_data
from indicators import add_indicators
from model import train_model


# TO THIS:
from risk import calculate_risk, get_stop_loss_price

# 1. PAGE CONFIGURATION (Must be the first Streamlit command)
st.set_page_config(
    page_title="AI Trading Analysis",
    page_icon="📈",
    layout="wide",  # Uses the full width of the screen
    initial_sidebar_state="expanded"
)

# 2. CUSTOM CSS FOR "AESTHETIC" LOOK
# This hides default menus and styles your metric cards
st.markdown("""
<style>
    /* Main Background adjustments */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Card Styling for Metrics */
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50; /* Default green accent */
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }
    
    /* Remove default top padding */
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# 3. SIDEBAR CONFIGURATION
with st.sidebar:
    st.title("⚡ AI Control Panel")
    st.caption("Live Market Analysis System")
    st.markdown("---")
    
    symbol = st.text_input("Enter Stock Symbol", "AAPL").upper()
    
    timeframe = st.selectbox(
        "Select Timeframe",
        ["1m", "5m", "15m", "1h", "1d"]
    )
    
    chart_type = st.selectbox(
        "Select Chart Type",
        ["Candlestick", "Line", "OHLC", "Area"]
    )
    
    st.markdown("---")
    st.info(f"Refreshes every 30s")
    
    # Auto-refresh logic inside sidebar to keep it clean
    st_autorefresh(interval=30000, key="datarefresh")
# ------------------------------------------------------------------
# 🚀 STRATEGY VALIDATOR (Scanner) - Self-Contained Fix
# ------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🚀 Strategy Validator")
run_scan = st.sidebar.button("Scan Top Stocks")

if run_scan:
    st.sidebar.write("🔄 Scanning market... please wait.")
    
    tickers = ["AAPL", "TSLA", "MSFT", "NVDA", "AMZN", "GOOGL", "AMD"]
    scan_results = []
    
    progress_bar = st.sidebar.progress(0)
    
    for i, ticker in enumerate(tickers):
        try:
            # --- FIX: Define Period LOCALLY inside the loop ---
            # This prevents the "NameError: period is not defined" crash
            if timeframe == "1m":
                scan_period = "7d"
            elif timeframe == "5m" or timeframe == "15m":
                scan_period = "60d"
            elif timeframe == "1h":
                scan_period = "730d"
            else:
                scan_period = "2y"
            # --------------------------------------------------

            # 1. Get Data using the local 'scan_period'
            df_scan = get_data(ticker, period=scan_period, interval=timeframe)
            df_scan = add_indicators(df_scan)
            
            # 2. Train Model 
            # (We use the exact same features list as your main app)
            features_scan = [
                "Close", "RSI", "MACD", "MACD_signal", "EMA20", "EMA50", 
                "ADX", "ATR", "Log_Ret", "Volatility_20", "RSI_Lag1"
            ]
            
            model_scan, acc_scan, X_scan, _ = train_model(df_scan)
            
            # 3. Backtest
            test_scan = df_scan.loc[X_scan.index].copy()
            bt_scan, log_scan = run_backtest(test_scan, model_scan, features_scan)
            
            # 4. Metrics
            metrics_scan = calculate_metrics(bt_scan["Equity"].values, log_scan)
            
            scan_results.append({
                "Ticker": ticker,
                "Return": f"{metrics_scan['Total Return']*100:.1f}%",
                "Win Rate": f"{metrics_scan['Win Rate']*100:.0f}%",
                "Profit Factor": f"{metrics_scan['Profit Factor']:.2f}",
                "Trades": metrics_scan['Trade Count']
            })
            
        except Exception as e:
            st.sidebar.error(f"Failed {ticker}: {e}")
            
        progress_bar.progress((i + 1) / len(tickers))
        
    st.sidebar.success("✅ Scan Complete!")
    if len(scan_results) > 0:
        scan_df = pd.DataFrame(scan_results)
        st.sidebar.markdown("### 🏆 Leaderboard")
        st.sidebar.dataframe(scan_df, hide_index=True)  

# ------------------------------------------------------------------
# 🔬 WALK-FORWARD VALIDATION (The Real-Money Test)
# ------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Robustness Lab")
run_wf = st.sidebar.button("Run Walk-Forward Test")

if run_wf:
    import validation # Import the new file
    
    st.sidebar.info("⏳ Simulating 1 year of trading... (This takes time)")
    
    # 1. Get a LOT of data (2 years)
    wf_data = get_data(symbol, period="2y", interval="1h") 
    wf_data = add_indicators(wf_data)
    
    # 2. Define Features (Must match your training logic)
    wf_features = [
        "Close", "RSI", "MACD", "MACD_signal", "EMA20", "EMA50", 
        "ADX", "ATR", "Log_Ret", "Volatility_20", "RSI_Lag1"
    ]
    
    # 3. Run the Test
    try:
        results_df, full_log = validation.walk_forward_test(
            wf_data, 
            wf_features, 
            train_window_days=90, 
            test_window_days=30
        )
        
        # 4. Display Results
        st.markdown("---")
        st.markdown(f"### 🔬 Walk-Forward Analysis: {symbol}")
        st.caption("This tests how the bot adapts to changing market conditions over time.")
        
        # Summary Metrics
        total_profit = results_df['Profit/Loss'].sum()
        positive_months = len(results_df[results_df['Profit/Loss'] > 0])
        total_months = len(results_df)
        consistency = (positive_months / total_months) * 100 if total_months > 0 else 0
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Profit", f"${total_profit:.2f}")
        m2.metric("Consistency", f"{consistency:.0f}%", f"{positive_months}/{total_months} Months")
        m3.metric("Total Trades", len(full_log))
        
        # Show the Month-by-Month breakdown
        st.markdown("#### 📅 Monthly Performance Breakdown")
        
        def color_profit(val):
            color = 'green' if val > 0 else 'red'
            return f'color: {color}'
            
        st.dataframe(
            results_df.style.map(color_profit, subset=['Profit/Loss']),
            use_container_width=True
        )
        
        if total_profit > 0 and consistency > 60:
            st.success("✅ **PASSED:** Strategy is robust and adaptable!")
        else:
            st.error("❌ **FAILED:** Strategy is unstable.")
            
    except Exception as e:
        st.error(f"Walk-Forward Failed: {e}") 

# 4. MAIN LOGIC & DATA PROCESSING
# Determine period based on timeframe (Keep your existing logic)
if timeframe == "1m":
    period = "7d"
elif timeframe == "5m":
    period = "60d"
elif timeframe == "15m":
    period = "60d"
elif timeframe == "1h":
    period = "730d"
else:
    period = "2y"

# Fetch and Process Data
try:
    data = get_data(symbol, period=period, interval=timeframe)
    data = add_indicators(data)
    
# Train Model & Predict
# ------------------------------------------------------------------
    # 1. Train Model & Get Unseen Test Data
    # ------------------------------------------------------------------
    # We now unpack 4 values instead of 3 (X_test contains the future data indices)
    model, accuracy, X_test, latest_features = train_model(data)
    
    # Get Prediction (0 or 1)
    prediction = model.predict(latest_features)[0]

    # NEW: Get Exact Probability (The "Confidence" Score)
    probs = model.predict_proba(latest_features)[0] 
    # probs[0] = Sell Probability, probs[1] = Buy Probability
    
    if prediction == 1:
        confidence = probs[1]
    else:
        confidence = probs[0]

    # ------------------------------------------------------------------
    # 2. Define Features (Must match model.py)
    # ------------------------------------------------------------------
    features = [
        "Close", "RSI", "MACD", "MACD_signal", "EMA20", "EMA50", 
        "ADX", "ATR", "Log_Ret", "Volatility_20", "RSI_Lag1"
    ]

    # ------------------------------------------------------------------
    # 3. RUN HONEST BACKTEST
    # ------------------------------------------------------------------
    # We grab the original OHLCV data that corresponds to our X_test indices
    # This guarantees the backtest uses the EXACT same "future" data the model was tested on.
    test_data = data.loc[X_test.index].copy()
    
  # UNPACK BOTH VALUES: DataFrame AND Trade Log
    bt_data, trade_log = run_backtest(test_data, model, features)
    
    # Pass both to the metrics calculator
    metrics = calculate_metrics(bt_data["Equity"].values, trade_log)
    # Pass both to the metrics calculator
    metrics = calculate_metrics(bt_data["Equity"].values, trade_log)
    
    # Risk Analysis
    volatility, risk_level = calculate_risk(data)

    # 5. DASHBOARD UI LAYOUT
    
    # -- Header Section --
    st.title(f"📈 {symbol} AI Analysis")
    st.markdown("Real-time technical analysis and prediction based on Random Forest implementation.")
    
    # -- KPI Metrics Row (The "Professional" Look) --
   # -- KPI Metrics Row --
    col1, col2, col3, col4, col5 = st.columns(5) # <--- Changed to 5 columns
    
    # Get current price & changes
    current_price = data['Close'].iloc[-1]
    price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
    
    # Calculate Stop Loss (The new math)
    suggested_stop_loss = get_stop_loss_price(data, direction="long")

    with col1:
        st.metric(label="Current Price", value=f"${current_price:.2f}", delta=f"{price_change:.2f}")
    with col2:
        pred_label = "BUY 🚀" if prediction == 1 else "SELL 📉"
        pred_color = "normal" if prediction == 1 else "inverse"
        
        # Display the Signal with the exact Confidence %
        st.metric(
            label="AI Signal", 
            value=pred_label, 
            delta=f"{confidence*100:.1f}% Confidence", 
            delta_color=pred_color
        )

# ⚖️ POSITION SIZE CALCULATOR (New Feature)
# ------------------------------------------------------------------
# This goes OUTSIDE the columns so it spans the full width
    st.markdown("---")
    st.markdown("### ⚖️ Live Trade Setup & Risk Calculator")

    with st.container(border=True):
        # 1. User Inputs
        r_col1, r_col2 = st.columns([1, 2])
        
        with r_col1:
            account_size = st.number_input("Account Balance ($)", value=10000, step=500)
            risk_pct = st.slider("Risk Per Trade (%)", 0.5, 5.0, 1.0, 0.5)
        
        # 2. Calculations
        # We use the latest ATR from your data to calculate a safe Stop Loss
        atr_value = data['ATR'].iloc[-1]
        stop_loss_dist = atr_value * 2
        
        # Calculate Prices
        suggested_stop_loss = current_price - stop_loss_dist
        risk_per_share = current_price - suggested_stop_loss    
        
        # Calculate Position Size
        max_risk_dollars = account_size * (risk_pct / 100)
        
        if risk_per_share > 0:
            shares_to_buy = int(max_risk_dollars / risk_per_share)
        else:
            shares_to_buy = 0
            
        total_cost = shares_to_buy * current_price

        # 3. Display Results
        with r_col2:
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("🛑 Stop Loss", f"${suggested_stop_loss:.2f}", f"-${stop_loss_dist:.2f} (2ATR)")
            res_col2.metric("📉 Max Risk", f"${max_risk_dollars:.0f}", f"-{risk_pct}%")
            res_col3.metric("📦 Position Size", f"{shares_to_buy} Shares", f"Cost: ${total_cost:,.0f}")

        # Visual Guide
        if total_cost > account_size:
            st.warning(f"⚠️ Insufficient funds. You need ${total_cost:,.0f} but have ${account_size:,.0f}.")
        else:
            st.success(f"✅ Trade Plan: Buy {shares_to_buy} shares. If price hits ${suggested_stop_loss:.2f}, you lose exactly ${max_risk_dollars:.0f} ({risk_pct}%).")

    with col3:
        st.metric(label="Model Accuracy", value=f"{accuracy*100:.1f}%")
        
    with col4:
        st.metric(label="Risk Level", value=risk_level, delta=f"{volatility*100:.2f}% Volatility", delta_color="off")

    with col5:
        # THE NEW CARD: specific price to exit
        st.metric(label="Stop Loss", value=f"${suggested_stop_loss:.2f}", delta="Safety Net", delta_color="off")
    # -- Chart Section --
# -- Chart Section --
    st.subheader("Market Overview")
    
    fig = go.Figure()

    # Custom Hover Template (Makes the tooltip clear and bold)
    hover_template = (
        "<b>Date:</b> %{x}<br>"
        "<b>Open:</b> %{open}<br>"
        "<b>High:</b> %{high}<br>"
        "<b>Low:</b> %{low}<br>"
        "<b>Close:</b> %{close}<extra></extra>"
    )

    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Market Data",
            # TradingView Style Colors (Teal & Soft Red)
            increasing_line_color='#26a69a', 
            decreasing_line_color='#ef5350',
            hovertemplate=hover_template
        ))
    
    elif chart_type == "Line":
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Close'], mode='lines', name='Close Price',
            line=dict(color='#2962FF', width=2)
        ))
    elif chart_type == "OHLC":
        fig.add_trace(go.Ohlc(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close']))
    elif chart_type == "Area":
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Close'], fill='tozeroy', mode='lines', name='Close Price',
            line=dict(color='#2962FF')
        ))

    # Updated Layout with "Crosshair" Hover Mode
    fig.update_layout(
        template="plotly_dark",
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified", # This adds the crosshair line
        xaxis=dict(
            showgrid=False,
            rangebreaks=[dict(bounds=["sat", "mon"])] # Hides weekends
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#1f2937',
            zeroline=False
        ),
        xaxis_rangeslider_visible=False,
        showlegend=False
    )
    
    # Custom hover label styling (Dark background for the tooltip)
   # Calculate zoom range to show only the last 100 candles initially
    # (This fixes the "squished" look)
    initial_range_start = data.index[-100] if len(data) > 100 else data.index[0]
    initial_range_end = data.index[-1]

    fig.update_layout(
        template="plotly_dark",
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        
        # X-Axis: Auto-Zoom to recent data & Hide Weekends
        xaxis=dict(
            showgrid=False,
            range=[initial_range_start, initial_range_end], # <--- THIS FIXES THE OVERLAP
            rangeslider=dict(visible=False),
            rangebreaks=[
                dict(bounds=["sat", "mon"]), # Hide weekends
            ]
        ),
        
        # Y-Axis: Clean grid lines
        yaxis=dict(
            showgrid=True,
            gridcolor='#1f2937',
            zeroline=False,
        ),
        showlegend=False
    )
    
    # ------------------------------------------------------------------
    # 📉 PLOT STOP LOSS ON CHART (Visual Upgrade)
    # ------------------------------------------------------------------
    fig.add_hline(
        y=suggested_stop_loss, 
        line_dash="dash", 
        line_color="red", 
        annotation_text=f"SL: ${suggested_stop_loss:.2f}",
        annotation_position="bottom right"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## 📈 Backtest Equity Curve")

    fig_equity = px.line(
        bt_data,
        x=bt_data.index,
        y="Equity",
        template="plotly_dark"
    )

    st.plotly_chart(fig_equity)

    # ------------------------------------------------------------------
    # 🌊 DRAWDOWN CHART (Risk Visualization)
    # ------------------------------------------------------------------
    # Calculate Drawdown: (Current Equity / Peak Equity) - 1
    bt_data['Drawdown'] = (bt_data['Equity'] / bt_data['Equity'].cummax()) - 1
    
    fig_dd = px.area(
        bt_data, 
        x=bt_data.index, 
        y="Drawdown", 
        template="plotly_dark"
    )
    
    # Style it Red (because Drawdown is "bad")
    fig_dd.update_traces(line_color='#ff4b4b', fillcolor='rgba(255, 75, 75, 0.2)')
    
    # Clean up the layout
    fig_dd.update_layout(
        title="🌊 Maximum Drawdown (Risk)",
        xaxis_title="Date",
        yaxis_title="Drawdown %",
        yaxis_tickformat=".1%", # Show as -1.5% instead of -0.015
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("### 📊 Strategy Performance")
    
    # Create 4 columns for detailed metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    kpi1.metric("Total Return", f"{metrics['Total Return']*100:.2f}%")
    kpi1.metric("Win Rate", f"{metrics['Win Rate']*100:.1f}%")
    
    kpi2.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
    kpi2.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}")
    
    kpi3.metric("Max Drawdown", f"{metrics['Max Drawdown']*100:.2f}%")
    kpi3.metric("Avg Win", f"${metrics['Avg Win']:.2f}")
    
    kpi4.metric("Total Trades", f"{metrics['Trade Count']}")
    kpi4.metric("Avg Loss", f"${metrics['Avg Loss']:.2f}")

    st.markdown("---")
    st.markdown("### 📜 Recent Trade History")
    
    # Check if there are any trades to show
    if len(trade_log) > 0:
        # Convert list of trades to a Dataframe
        trades_df = pd.DataFrame(trade_log)
        
        # Format the numbers to look professional (e.g., $150.20, 1.5%)
        trades_df["Entry Price"] = trades_df["Entry Price"].map("${:,.2f}".format)
        trades_df["Exit Price"] = trades_df["Exit Price"].map("${:,.2f}".format)
        trades_df["PnL"] = trades_df["PnL"].map("${:,.2f}".format)
        trades_df["Return Pct"] = trades_df["Return Pct"].map("{:.2f}%".format)
        
        # Reorder columns for readability
        trades_df = trades_df[["Date", "Type", "Entry Price", "Exit Price", "PnL", "Return Pct", "Reason"]]
        
        # Display the table
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
    else:
        st.info("🟡 No trades generated in this period. The AI is waiting for a high-confidence setup.")

    # -- Footer / Debug Data --
    with st.expander("📊 View Raw Data & Model Details"):
        st.dataframe(data.tail())
        st.write(f"Model used: Random Forest Classifier | Volatility: {volatility}")
    # ------------------------------------------------------------------
    # 🧠 MODEL BRAIN (Feature Importance)
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🧠 Model Logic: What drives the decision?")
    
    # 1. Extract Importance scores from the Random Forest
    importances = model.feature_importances_
    
    # 2. Create a DataFrame for the chart
    feature_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)
    
    # 3. Create the Chart
    fig_feat = px.bar(
        feature_df, 
        x="Importance", 
        y="Feature", 
        orientation='h', 
        template="plotly_dark",
        title="Most Important Indicators"
    )
    
    fig_feat.update_traces(marker_color='#00CC96')
    
    st.plotly_chart(fig_feat, use_container_width=True)
# ------------------------------------------------------------------
    # 📰 REAL-TIME NEWS INTELLIGENCE (Finnhub)
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📰 Market Sentiment (Fundamental Analysis)")
    
    with st.spinner(f"Scanning Finnhub News for {symbol}..."):
        news_df, sentiment_score = news_analysis.get_finnhub_news(symbol)
    
    # Create 2 Columns
    news_col1, news_col2 = st.columns([1, 2])
    
    # LEFT COLUMN: The "Hybrid" Decision
    with news_col1:
        st.markdown("#### 🧠 AI Consensus")
        
        # 1. Determine Sentiment Label
        if sentiment_score > 0.15:
            sent_label = "POSITIVE 🚀"
            sent_color = "green"
        elif sentiment_score < -0.15:
            sent_label = "NEGATIVE 🐻"
            sent_color = "red"
        else:
            sent_label = "NEUTRAL 😐"
            sent_color = "off"
            
        st.metric("News Sentiment", sent_label, f"{sentiment_score:.2f} Score")
        
        # 2. THE HYBRID LOGIC (Technical + Fundamental)
        st.markdown("---")
        st.markdown("**Final Verification:**")
        
        if prediction == 1 and sentiment_score > 0.1:
            st.success("✅ **STRONG BUY**: Technicals and News agree!")
        elif prediction == 1 and sentiment_score < -0.1:
            st.warning("⚠️ **RISKY BUY**: Model says Buy, but News is Negative.")
        elif prediction == 0 and sentiment_score < -0.1:
            st.error("✅ **STRONG SELL**: Technicals and News agree!")
        else:
            st.info("ℹ️ **MIXED SIGNALS**: Trade with caution.")

    # RIGHT COLUMN: The Headlines
    with news_col2:
        st.markdown("#### Latest Global Headlines")
        if not news_df.empty:
            st.dataframe(
                news_df[["Date", "Source", "Headline", "Sentiment"]],
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning("No recent news found on Finnhub.")
except Exception as e:
    st.error(f"Error loading data for {symbol}. Please check the symbol or try again later.")
    st.exception(e)